import torch
import torch.nn as nn
from torch.nn import ParameterList, Parameter
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.distributions.multivariate_normal import MultivariateNormal
from .quadrature import GaussHermiteQuadrature1D
from torch.distributions.kl import kl_divergence


def create_torch_random_gen(value):
    gen = torch.Generator()
    if value is None:
        gen.seed()
        return gen
    if isinstance(value, int):
        gen.manual_seed(value)
        return gen
    if isinstance(value, torch.Generator):
        return value
    raise Exception(
        "torch random generator must be initialized"
        " by int, None or via torch.Generator instance, "
        " got {} instead".format(type(value)))

class CLBaseline(nn.Module):

    def __init__(self, base_model, h_dim, out_dim=1, device='cpu', seed=None):
        '''
        :Parameters:
        base_model: torch.nn.Module: task-agnostic model
        h_dim: int: dimension of base_model output
        out_dim: output dimension of task-specific tensor \omega
        (dimension of loss_function input)
        '''
        super().__init__()

        self.base = base_model
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.device = device
        #replay buffers for the previous tasks (includes torch.utils.data.Subsets)
        self.tasks_replay_buffers = []
        #task-specific tensors (which applied to base model outputs)
        self.tasks_omegas = ParameterList()
        self.input_classes_transformers = []
        self.hidden_classes_transformers = []

        if out_dim == 1:
            # computes loss
            self.loss_func = nn.BCEWithLogitsLoss()

            # predicts distribution over the classes
            def pred_func(input):
                pred = F.sigmoid(input)
                return torch.stack([1. - pred, pred], dim=-1).squeeze()
        
            self.pred_func = pred_func
        
        if out_dim > 1:
            self.loss_func = nn.CrossEntropyLoss()
            self.pred_func = nn.Softmax()
        
        self.torch_gen = create_torch_random_gen(seed)
        self.to(self.device)
    
    def create_new_task(self, classes):
        '''
        Create new task-specific tensor \omega
        :Parameters:
        classes: list: list of target classes
        '''
        w = Parameter(torch.randn(
            (self.out_dim, self.h_dim), generator=self.torch_gen)).to(self.device)
        self.tasks_omegas.append(w)
        classes.sort()

        def input_classes_transform(tsr):
            for i, _cls in enumerate(classes):
                tsr[tsr == _cls] = i
            return tsr
        
        def output_classes_transform(tsr):
            for i in range(len(classes)):
                tsr[tsr == i] = classes[i]
            return tsr
        
        self.input_classes_transformers.append(input_classes_transform)
        self.hidden_classes_transformers.append(output_classes_transform)

    def forward(self, cl_batch):
        '''
        Returns the objective with respect to (x, target)
        '''
        if len(self.tasks_omegas) == 0:
            raise Exception("No tasks have been created yet!")

        # some consistency check
        assert(len(self.tasks_omegas) == 1 + len(self.tasks_replay_buffers))
        assert(len(cl_batch) == len(self.tasks_omegas))

        main_batch_size = cl_batch[-1][0].size(0)
        main_transform = self.input_classes_transformers[-1]
        loss = self.loss_func(
            torch.matmul(self.base(cl_batch[-1][0]), self.tasks_omegas[-1].T).squeeze(),
            main_transform(cl_batch[-1][1]))

        for i in range(len(self.tasks_replay_buffers)):
            omega = self.tasks_omegas[i]
            x = cl_batch[i][0]
            target = cl_batch[i][1]
            transform = self.input_classes_transformers[i]
            curr_unb_loss = self.loss_func(
                torch.matmul(self.base(x), omega.T).squeeze(), 
                transform(target))
            curr_batch_size = cl_batch[i][0].size(0)
            assert(curr_unb_loss >= 0.)
            loss += main_batch_size/float(curr_batch_size) * curr_unb_loss
        assert(loss >= 0.)
        return loss
    
    @torch.no_grad()
    def predict(self, x, k, get_class=False):
        '''
        Compute prediction by input x
        :Params:
        x: torch.tensor: input tensor
        k: int: task number
        get_class: bool: if False, returns the predictive probability
        distribution over the classes, othervise returns the predicted class
        '''
        assert(k >= 0)
        assert(k < len(self.tasks_omegas))
        omega = self.tasks_omegas[k]
        hidden_transformer = self.hidden_classes_transformers[k]
        distr = self.pred_func(torch.matmul(self.base(x), omega.T).squeeze())
        if get_class:
            if len(distr.shape) == 1:
                return hidden_transformer(torch.argmax(distr).cpu()).item()
            else:
                return hidden_transformer(torch.argmax(distr, dim=-1).cpu()).numpy()
        else:
            return distr.cpu()
    
    @torch.no_grad()
    def select_inducing(self, task_dataset, N=100, criterion='random'):
        '''
        Given task dataset compoute N inducing points 
        for the current task
        '''
        assert(len(self.tasks_omegas) == 1 + len(self.tasks_replay_buffers))

        if criterion == "random":
            indices = torch.randperm(len(task_dataset), generator=self.torch_gen).numpy()
            select_indices = indices[:N]
            self.tasks_replay_buffers.append(Subset(task_dataset, select_indices))
        else:
            raise Exception(
                "Criterion {} not implemented".format(criterion))


class FRCL(nn.Module):
    def __init__(self, base_model, h_dim, device, sigma_prior=1):
        super(FRCL, self).__init__()
        self.device = device
        self.base = base_model.to(device)
        self.sigma_prior = sigma_prior
        self.L = Parameter(torch.eye(h_dim), requires_grad=True).to(device)
        self.mu = Parameter(torch.normal(0, 0.1, size=(h_dim,)), requires_grad=True).to(device)
        self.w_distr = MultivariateNormal(self.mu, scale_tril=self.L)
        self.w_prior = MultivariateNormal(torch.zeros(h_dim).to(device),
                                          covariance_matrix=sigma_prior*torch.eye(h_dim).to(device))
        self.quadr = GaussHermiteQuadrature1D().to(device)
        self.prev_tasks_distr = [] #previous tasks as torch distributions
        self.prev_tasks_tensors = [] #previous tasks as torch tensors
       
    def forward(self, x, target, N_k):
        """
        Return -ELBO
        N_k = len(dataset), required for unbiased estimate through minibatch
        """
        elbo = 0
        phi = self.base(x)
        def loglik(sample): #currently hardcoded for binary classification
            return -torch.log(1 + torch.exp(-target * sample))
        mu = self.mu
        cov = self.L @ self.L.T
        means = phi @ mu
        variances = torch.diagonal(phi @ cov @ phi.T, 0)
        elbo += (self.quadr(loglik, means, variances)).sum()
        elbo /= x.shape[0] #mean
        
        kls = 0
        kls -= kl_divergence(self.w_distr, self.w_prior)

        for i in range(len(self.prev_tasks_distr)):
            phi_i = self.base(self.prev_tasks_tensors[i])
            cov_i = phi_i @ phi_i.T
            p_u = MultivariateNormal(torch.zeros(phi_i.shape[0]).to(self.device),
                                     scale_tril=phi_i)#cov_i * self.sigma_prior)
            kls -= kl_divergence(self.prev_tasks_distr[i], p_u)
        elbo += kls / N_k

        return -elbo 

    @torch.no_grad()
    def get_predictive(self, x, k):
        """ Computes predictive distribution according to section 2.5
            x - batch of data
            k - index of task
            Return predictive distribution q_\theta(f)
        """
        phi_x = self.base(x)
        phi_z = self.base(self.prev_tasks_tensors[k])
        k_xx = phi_x @ phi_x.T
        k_xz = phi_x @ phi_z.T
        k_zz = phi_z @ phi_z.T
        k_zz_ = torch.inverse(k_zz)
        mu_u = self.prev_tasks_distr[k].mean
        cov_u = self.prev_tasks_distr[k].covariance_matrix

        mu = phi_x @ phi_z.T @ k_zz_ @  mu_u
        sigma = k_xx + k_xz @ k_zz_ @ (cov_u  - k_zz) @ k_zz_ @ k_xz.T
        sigma *= torch.eye(sigma.shape[0]).to(self.device) #we are interested only 
                                                           #in diagonal part for inference
        return MultivariateNormal(loc=mu, covariance_matrix=sigma)

    @torch.no_grad()
    def predict(self, x, k, MC_samples=20):
        """Compute p(y) by MC estimate from q_\theta(f)?
        """
        distr = self.get_predictive(x.to(self.device), k)

        def likelihood(y): #currently hardcoded for binary classification
            return 1./ (1 + torch.exp(-y))

        predicted = 0
        for i in range(MC_samples):
            sample = distr.sample()
            predicted += likelihood(sample)
        predicted /= MC_samples
        return predicted

    @torch.no_grad()
    def select_inducing(self, task_dataloader, N=100, criterion="random"):
        """Given task dataloader compute N inducing points
           Updates self.prev_tasks_distr and self.prev_tasks_tensors
        """
        if (criterion=="random"):
            smp = torch.utils.data.DataLoader(task_dataloader.dataset,
                                              batch_size=N,
                                              shuffle=True)
            X, y = next(iter(smp))
            X = X.to(self.device)
            phi = self.base(X)
            mu_u = phi @ self.mu
            L_u = phi @ self.L
            cov = L_u @ L_u.T
            self.prev_tasks_distr.append(MultivariateNormal(loc=mu_u,
                                                            covariance_matrix=cov))
            self.prev_tasks_tensors.append(X)
            return
            
                
    @torch.no_grad()
    def detect_boundary(self, x, l_old):
        """Given new batch x and kl divergence for previous minibatch l_old
           compute l_new and perform statistical test
           Returns l_new and indicator of significance (0 or 1)
        """
        pass

            
            

if __name__ == "__main__":
    clb = CLBaseline(None, None)

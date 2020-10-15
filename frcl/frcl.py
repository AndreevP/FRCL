import torch
import torch.nn as nn
from torch.nn import ParameterList, Parameter
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from .quadrature import GaussHermiteQuadrature1D
from torch.distributions.kl import kl_divergence
from copy import copy
from scipy.stats import ttest_ind
import abc


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

class CLBaseline(nn.Module, abc.ABC):

    def __len__(self):
        return len(self.tasks_replay_buffers)

    def __init__(
        self, base_model, h_dim, out_dim=1, device='cpu', 
        seed=None):
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

    @abc.abstractmethod
    def forward(self, cl_batch):
        pass

    @abc.abstractmethod
    def create_replay_buffer(self, dataset):
        pass
    
    def create_new_task(self):
        '''
        Create new task-specific tensor \omega
        :Parameters:
        classes: list: list of target classes
        '''
        w = Parameter(torch.randn(
            (self.out_dim, self.h_dim), generator=self.torch_gen)).to(self.device)
        self.tasks_omegas.append(w)
    
    def _compute_task_loss(self, k, X, target):
        omega = self.tasks_omegas[k]
        return self.loss_func(torch.matmul(self.base(X), omega.T).squeeze(), target)
    
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
        distr = self.pred_func(torch.matmul(self.base(x), omega.T).squeeze())
        if get_class:
            if len(distr.shape) == 1:
                return torch.argmax(distr).cpu().item()
            else:
                return torch.argmax(distr, dim=-1).cpu().numpy()
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
            self.create_replay_buffer(Subset(task_dataset, select_indices))
        else:
            raise Exception(
                "Criterion {} not implemented".format(criterion))

class DeterministicCLBaseline(CLBaseline):

    def forward(self, X, target):
        '''
        returns the objective with respect to the (X, target)
        '''
        loss = self._compute_task_loss(-1, X, target)
        for i, repl_buffer in enumerate(self.tasks_replay_buffers):
            curr_X, curr_target = repl_buffer
            curr_loss = self._compute_task_loss(i, curr_X, curr_target)
            # curr_loss *= X.size(0)/float(curr_X.size(0))
            loss += curr_loss
        return loss
    
    def create_replay_buffer(self, dataset):
        dataloader = DataLoader(dataset, batch_size=len(dataset))
        self.tasks_replay_buffers.append(next(iter(dataloader)))


class StochasticCLBaseline(CLBaseline):

    def forward(self, cl_batch):
        '''
        returns the objective with respect to the cl_batch
        '''

        # main_batch_size = cl_batch[-1][0].size(0)
        loss = 0.0
        for i in range(len(self.tasks_omegas)):
            omega = self.tasks_omegas[i]
            x = cl_batch[i][0]
            target = cl_batch[i][1]
            curr_unb_loss = self._compute_task_loss(i, x, target)
            # curr_batch_size = cl_batch[i][0].size(0)
            # loss += main_batch_size/float(curr_batch_size) * curr_unb_loss
            loss += curr_unb_loss
        return loss 
    
    def create_replay_buffer(self, dataset):
        self.tasks_replay_buffers.append(dataset)


class FRCL(nn.Module):
    def __init__(self, base_model, h_dim, device, sigma_prior=1):
        super(FRCL, self).__init__()
        self.device = device
        self.h_dim = h_dim
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
    
    def __len__(self):
        return len(self.prev_tasks_tensors)
       
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
            p_u = MultivariateNormal(torch.zeros(cov_i.shape[0]).to(self.device),
                                     covariance_matrix=cov_i)#cov_i * self.sigma_prior)
            kls -= kl_divergence(self.prev_tasks_distr[i], p_u)
        elbo += kls / N_k

        return -elbo 

    @torch.no_grad()
    def get_predictive(self, x, k, return_distr=True):
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
        return MultivariateNormal(loc=mu, covariance_matrix=sigma) if return_distr else (mu, sigma)

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
            random_points = torch.utils.data.DataLoader(task_dataloader.dataset,
                                              batch_size=N,
                                              shuffle=True)
            Z, _ = next(iter(random_points))
            Z = Z.to(self.device)

        if (criterion=="deterministic"):

            def calculate_induce_quality_statistic(all_points, inducing):
                """calculates trace statistic of inducing cquality"""
                phi_x = self.base(all_points)
                phi_z = self.base(inducing)
                k_xx = phi_x @ phi_x.T
                k_xz = phi_x @ phi_z.T
                k_zz = phi_z @ phi_z.T
                return torch.trace(k_xx - k_xz @ torch.inverse(k_zz) @ k_xz.T)
            
            def find_best_inducing_points(start_inducing_set="random",
                                          task_dataloader=task_dataloader, 
                                          max_iter=1000, return_statistic=False):
                
                """Sequentially adds a new point instead of a random one in
                the initial set of inducing points, if the value of the statstic
                above lessens and does not do anything otherwise.

                - start_inducing_set: list of points to start from
                - max_iter: maximum number of tries to add a point
                """
                
                if start_inducing_set == "random":
                    random_points = torch.utils.data.DataLoader(task_dataloader.dataset,
                                              batch_size=N,
                                              shuffle=False)
                    Z, _ = next(iter(random_points))
                    Z = Z.to(self.device)
                else:
                    assert len(start_inducing_set) == N 
                    assert isinstance(start_inducing_set[0], torch.Tensor)
                    Z = start_inducing_set.to(self.device)
                 
                all_points = [elem[0] for elem in task_dataloader.dataset]
                
                potential_points = torch.utils.data.DataLoader(task_dataloader.dataset,
                                              batch_size=max_iter,
                                              shuffle=True)
                
                for point, _ in potential_points:
                    Z_new = Z.copy()
                    Z_new[np.random.randint(0, N - 1)] = point
                    T = calculate_induce_quality_statistic(all_points, Z)
                    T_new = calculate_induce_quality_statistic(all_points, Z_new)
                    if T_new < T:
                        Z = Z_new
                
                return Z, min(T, T_new) if return_statistic else Z
            
            Z = find_best_inducing_points()
        
        phi = self.base(Z)
        mu_u = phi @ self.mu
        L_u = phi @ self.L
        cov = L_u @ L_u.T
        self.prev_tasks_distr.append(MultivariateNormal(loc=mu_u,
                                                        covariance_matrix=cov))
        self.prev_tasks_tensors.append(Z)
        return
            
                
    @torch.no_grad()
    def detect_boundary(self, x, l_old, return_p_value=False):
        """Given new batch x and kl divergence for previous minibatch l_old
           compute l_new and perform statistical test
           Returns l_new and indicator of significance (0 or 1)
        """
        assert len(x) == len(l_old)
        
        def gaus_sym_kl(p_mu, p_var, q_mu, q_var, dim=1):
            inv_p_var = torch.inverse(p_var)
            inv_q_var = torch.inverse(q_var)
            return (torch.trace(inv_p_var @ q_var + inv_q_var @ p_var) - 2 * dim + \
               (p_mu - q_mu).T @ (inv_p_var + inv_q_var) @ (p_mu - q_mu)) / 4
        
        phi_x = self.base(x)
        p_vars = torch.diagonal(phi_x @ phi_x.T, 0)
        p_mus = torch.zeros(len(l_old))
        q_mus, q_covar_matrix = self.get_predictive(x, -1, return_distr=False)
        q_vars = torch.diagonal(q_covar_matrix, 0)
        l_new = torch.Tensor([gaus_sym_kl(p_mus[i], p_vars[i], q_mus[i], q_vars[i]) for i in range(len(l_old))])
        t, p_value = ttest_indest_ind(l_old, l_new, equal_var=False)    
        return l_new, t, p if return_p_value else l_new, t
    
if __name__ == "__main__":
    clb = CLBaseline(None, None)

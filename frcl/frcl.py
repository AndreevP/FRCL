import torch
import torch.nn as nn
from torch.nn import ParameterList, Parameter
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from frcl.quadrature import GaussHermiteQuadrature1D
from torch.distributions.kl import kl_divergence
from copy import copy
import abc

def moveaxis(tensor: torch.Tensor, source: int, destination: int) -> torch.Tensor:
    dim = tensor.dim()
    perm = list(range(dim))
    if destination < 0:
        destination += dim
    perm.pop(source)
    perm.insert(destination, source)
    return tensor.permute(*perm)


def base_model_eval(function):
    def wrapper(self, *_args, **_kwargs):
        self.base.eval()
        res = function(self, *_args, **_kwargs)
        self.base.train()
        return res
    return wrapper


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
        self, base_model, h_dim, multiclass=False, device='cpu', 
        seed=None):
        '''
        :Parameters:
        base_model: torch.nn.Module: task-agnostic model
        h_dim: int: dimension of base_model output
        multiclass: bool: kind of tasks we will solve
        (dimension of loss_function input)
        '''
        super().__init__()
        self.base = base_model
        self.h_dim = h_dim
        self.device = device
        self.multiclass = multiclass
        #replay buffers for the previous tasks (includes torch.utils.data.Subsets)
        self.tasks_replay_buffers = []
        #task-specific tensors (which applied to base model outputs)
        self.tasks_omegas = ParameterList()
        # out dims for subsequent tasks
        self.out_dims = []

        if not self.multiclass:
            # computes loss
            self.loss_func = nn.BCEWithLogitsLoss()

            # predicts distribution over the classes
            def pred_func(input):
                pred = F.sigmoid(input)
                return torch.stack([1. - pred, pred], dim=-1).squeeze()
        
            self.pred_func = pred_func
        
        if self.multiclass:
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
    
    def create_new_task(self, out_dim):
        '''
        Create new task-specific tensor \omega
        :Parameters:
        classes: list: list of target classes
        '''
        if self.multiclass:
            assert(out_dim > 1)
        else:
            assert(out_dim == 1)
        self.out_dim = out_dim
        self.out_dims.append(out_dim)
        w = Parameter(torch.randn(
            (self.out_dim, self.h_dim), generator=self.torch_gen).to(self.device))
        self.tasks_omegas.append(w)
    
    def _compute_task_loss(self, k, X, target):
        omega = self.tasks_omegas[k]
        return self.loss_func(torch.matmul(self.base(X), omega.T).squeeze(), target)
    
    @base_model_eval
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
        x = x.to(self.device)
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
    
    @base_model_eval
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
            curr_X = curr_X.to(self.device)
            if self.out_dim == 1:
                curr_target = curr_target.float().to(self.device)
            else:
                curr_target = curr_target.long().to(self.device)
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

def kl(m1, S1, m2, S2):
    S2 = S2 + torch.eye(S2.shape[0]).to(S2) * 1e-3
    S1 = S1 + torch.eye(S1.shape[0]).to(S1) * 1e-3
    S2_ = torch.inverse(S2)
    return 0.5 * (torch.trace(S2_ @ S1) + (m2 - m1).T @ S2_ @ (m2 - m1) \
                  - S1.shape[0] + torch.logdet(S2) - torch.logdet(S1))
        
class FRCL(nn.Module):
    def __init__(self, base_model, h_dim, device, sigma_prior=1, multiclass=False):
        super(FRCL, self).__init__()
        
        self.h_dim = h_dim
        self.sigma_prior = sigma_prior
        self.device = device
        self.w_prior = MultivariateNormal(torch.zeros(h_dim).to(device),
                                          covariance_matrix=sigma_prior*torch.eye(h_dim).to(device)) 
        self.prev_tasks_distr = [] #previous tasks as torch distributions
        self.prev_tasks_tensors = [] #previous tasks as torch tensors
        self.quadr = GaussHermiteQuadrature1D().to(device)
        self.base = base_model.to(device)
        self.multiclass = multiclass
        
        if not multiclass:
            # computes loss
            self.loss_func = nn.BCEWithLogitsLoss(reduction='none')

            # predicts distribution over the classes
            def pred_func(input):
                pred = F.sigmoid(input)
                return torch.stack([1. - pred, pred], dim=-1).squeeze()
            self.pred_func = pred_func
           
        if multiclass:
            self.loss_func = nn.CrossEntropyLoss(reduction='none')
            self.pred_func = nn.Softmax()   

        # out dims for subsequent tasks
        self.out_dims = [] 

    def create_new_task(self, out_dim):
        if self.multiclass:
            assert(out_dim > 1)
        else:
            assert(out_dim == 1)
        self.out_dim = out_dim
        self.out_dims.append(out_dim)

        self.L = ParameterList([Parameter(torch.eye(self.h_dim).to(self.device), requires_grad=True) \
                  for _ in range(self.out_dim)]) 
        self.mu = ParameterList([Parameter(torch.normal(0, 1, size=(self.h_dim,)).to(self.device), requires_grad=True) \
                   for _ in range(self.out_dim)])  

    def __len__(self):
        return len(self.prev_tasks_tensors)
       
    def forward(self, x, target, N_k, state=None, method="SVI", N_samples=10):
        """
        Return -ELBO
        N_k = len(dataset), required for unbiased estimate through minibatch
        """
        assert method in ["SVI", "quadrature"]
        
        elbo = 0
        phi = self.base(x)

        def loglik(sample):
            sample = moveaxis(sample.view(sample.shape[0], self.out_dim, x.shape[0]),
                                   1, 2)
            if (self.out_dim == 1):
                sample = sample[:, :, 0]
                tar = target.float()
            else:
                tar = target.long()
            return torch.stack([-self.loss_func(sample[i], tar) for i in range(sample.shape[0])], axis=0)

        mu = self.mu
        cov = [self.L[i] @ self.L[i].T for i in range(len(self.L))]
        means = torch.cat([phi @ mu[i] for i in range(len(mu))], axis=0)
      #  variances = torch.cat([((phi @ cov[i]) * phi).sum(-1) for i in range(len(cov))], axis = 0)
        variances = torch.cat([torch.diagonal(phi @ cov[i] @ phi.T, 0) for i in range(len(cov))], axis=0) 
     #   print(means.shape[0])
     #   print(variances)
        if (method == "quadrature"):
            assert self.out_dim == 1, "Quadrature is biased for out_dim > 1"
            elbo += (self.quadr(loglik, means, variances)).sum()
            elbo /= x.shape[0] #mean
        else:
            samples = torch.stack([means + torch.sqrt(variances + 1e-6) * \
                                  torch.randn(means.shape[0]).to(self.device) \
                                  for i in range(N_samples)], axis=0)
            elbo = loglik(samples).mean() 
        if state is not None:
            state.e = elbo.item()
            state.kls = []
            
        kls = 0
        for i in range(self.out_dim):
           # kls -= kl_divergence(self.w_distr[i], self.w_prior)
            kls -= kl(self.mu[i], self.L[i] @ self.L[i].T,
                      self.w_prior.mean, self.w_prior.covariance_matrix)
            curr_task_kls = -kls.item()

        for i in range(len(self.prev_tasks_distr)):
            out_dim = self.out_dims[i]
            phi_i = self.base(self.prev_tasks_tensors[i])
            cov_i = phi_i @ phi_i.T + torch.eye(phi_i.shape[0]).to(self.device) * 1e-6
           # p_u = MultivariateNormal(torch.zeros(cov_i.shape[0]).to(self.device),
           #                          covariance_matrix=cov_i * self.sigma_prior)
           # kls -= sum([kl_divergence(self.prev_tasks_distr[i][j], p_u) for j in range(self.out_dim)])
            prev_cls = sum([kl(self.prev_tasks_distr[i][j].mean, self.prev_tasks_distr[i][j].covariance_matrix,
                          torch.zeros(cov_i.shape[0]).to(self.device), cov_i * self.sigma_prior) \
                       for j in range(out_dim)])
            if state is not None:
                state.kls.append(prev_cls.item())
            kls -= prev_cls
        elbo += kls / N_k

        if state is not None:
            state.kls.append(curr_task_kls)
            state.kls_div_nk = kls.item() / N_k
            state.elbo = elbo.item()

        return -elbo

    @base_model_eval
    @torch.no_grad()
    def get_predictive(self, x, k):
        """ Computes predictive distribution according to section 2.5
            x - batch of data
            k - index of task
            Return predictive distribution q_\theta(f)
        """
        phi_x = self.base(x)
        phi_z = self.base(self.prev_tasks_tensors[k])
        k_xx = phi_x @ phi_x.T * self.sigma_prior
        k_xz = phi_x @ phi_z.T * self.sigma_prior
        k_zz = phi_z @ phi_z.T * self.sigma_prior 
        k_zz_ = torch.inverse(k_zz + torch.eye(phi_z.shape[0]).to(self.device) * 1e-3)
        out_dim = self.out_dims[k]
        mu_u = [self.prev_tasks_distr[k][i].mean for i in range(out_dim)]
        cov_u = [self.prev_tasks_distr[k][i].covariance_matrix for i in range(out_dim)]

        mu = [phi_x @ phi_z.T @ k_zz_ @  mu_u[i] for i in range(out_dim)]
        sigma = [k_xx + k_xz @ k_zz_ @ (cov_u[i]  - k_zz + \
                                        torch.eye(k_zz.shape[0]).to(self.device) * 1e-4) \
                                        @ k_zz_ @ k_xz.T for i in range(out_dim)]
        sigma = [sigma[i] * torch.eye(sigma[i].shape[0]).to(self.device)+\
                 torch.eye(sigma[i].shape[0]).to(self.device) * 1e-6\
                 for i in range(out_dim)] 
        print([s.min() for s in sigma])
        sigma = [torch.clamp(sigma[i], min=0, max=10000.)+\
                 torch.eye(sigma[i].shape[0]).to(self.device) * 1e-6\
                 for i in range(out_dim)]    
                                                             #we are interested only 
                                                             #in diagonal part for inference ?
        return [MultivariateNormal(loc=mu[i], covariance_matrix=sigma[i]) for i in range(out_dim)]

    @torch.no_grad()
    def predict(self, x, k, MC_samples=20):
        """Compute p(y) by MC estimate from q_\theta(f)?
        """
        distr = self.get_predictive(x.to(self.device), k)
        predicted = 0
        out_dim = self.out_dims[k]
        for i in range(MC_samples):
            sample = [distr[i].sample() for i in range(out_dim)]
            predicted += self.pred_func(torch.stack(sample, axis=1))
        predicted /= MC_samples
        return predicted

    @base_model_eval
    @torch.no_grad()
    def select_inducing(self, task_dataloader, N=100, criterion="random"):
        """
        Given task dataloader compute N inducing points
        Updates self.prev_tasks_distr and self.prev_tasks_tensors
        """
        
        if (criterion=="random"):
            smp = torch.utils.data.DataLoader(task_dataloader.dataset,
                                              batch_size=N,
                                              shuffle=True)
            
            Z, y = next(iter(smp))
            Z = Z.to(self.device)

        if (criterion=="deterministic"):

            def calculate_induce_quality_statistic(inducing_points):
                """
                Calculates trace statistic of inducing cquality 
                (up to multiplication by prior variance)
                """
                statistic = 0
                
                loader_for_batch_computation = torch.utils.data.DataLoader(
                                              task_dataloader.dataset,
                                              batch_size=500,
                                              shuffle=False)
                
                phi_z = self.base(inducing_points.to(device))
                k_zz = phi_z @ phi_z.T
                inv_k_zz = torch.inverse(k_zz +\
                 torch.eye(k_zz.shape[0]).to(self.device) * 1e-3)
                for x_batch, _ in loader_for_batch_computation:
                    phi_x = self.base(x_batch.to(device))
                    k_xz = phi_x @ phi_z.T
                    k_xx = phi_x @ phi_x.T
                    statistic += torch.trace(k_xx - k_xz @ inv_k_zz @ k_xz.T)
                return statistic
            
            def find_best_inducing_points(start_inducing_set="random",
                                          task_dataloader=task_dataloader, 
                                          max_iter=300, 
                                          early_stop_num_iter=80,
                                          verbose=True):
                
                """Sequentially adds a new point instead of a random one in
                the initial set of inducing points, if the value of the statstic
                above lessens, and does not do anything otherwise.
                - start_inducing_set: list of points to start from
                - max_iter: maximum number of tries to add a point
                """
                
                if start_inducing_set == "random":
                    random_points = torch.utils.data.DataLoader(task_dataloader.dataset,
                                              batch_size=N,
                                              shuffle=True)
                    Z, _ = next(iter(random_points))
                    Z = Z.to(self.device)
                else:
                    assert len(start_inducing_set) == N 
                    assert isinstance(start_inducing_set[0], torch.Tensor)
                    Z = start_inducing_set.to(self.device)
                 

                potential_points, _ = next(iter(torch.utils.data.DataLoader(
                                      task_dataloader.dataset,
                                      batch_size=max_iter,
                                      shuffle=True)))
                
                T = calculate_induce_quality_statistic(Z)
                new_point_counter = 0
                early_stop_counter = 0
                for i, point in enumerate(potential_points):
                    Z_new = copy(Z)
                    Z_new[np.random.randint(0, N)] = point
                    T_new = calculate_induce_quality_statistic(Z_new)
                    if T_new < T:
                        T, Z = T_new, Z_new
                        new_point_counter += 1
                        early_stop_counter = 0
                    else:
                         early_stop_counter += 1
                    if verbose and i % 10 == 0:
                       print("Iteration {} out of {} is in progress".
                             format(i, max_iter))
                       print("Current best statistic is ", round(T.item(), 3))
                       print("New points added ", new_point_counter, '\n')
                    if early_stop_counter == early_stop_num_iter:
                        print("Early stop activated!")
                        break
                return Z
            
            Z = find_best_inducing_points()
        
        phi = self.base(Z)
        mu_u = [phi @ self.mu[i] for i in range(self.out_dim)]
        L_u = [phi @ self.L[i] for i in range(self.out_dim)]
        cov = [L_u[i] @ L_u[i].T for i in range(self.out_dim)]
        #regularization
        cov = [cov[i] + torch.eye(cov[i].shape[0]).to(self.device) * 1e-4 for i in range(self.out_dim)]
        self.prev_tasks_distr.append([MultivariateNormal(loc=mu_u[i],
                                    covariance_matrix=cov[i]) for i in range(self.out_dim)])
        self.prev_tasks_tensors.append(Z)

        return
            
    @base_model_eval           
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
        p_covar = self.sigma_prior ** 2 * phi_x @ phi_x.T + 1e-6 * torch.eye(phi_x.shape[0])
        p_mu = torch.zeros(len(l_old))
        q_dists = self.get_predictive(x, -1, return_distr=False)
        q_mus, q_covars = zip(*[(dist.mean, dist.covariance_matrix) for dist in q_dists])
        l_new = torch.Tensor([gaus_sym_kl(p_mu, p_covar, q_mus[i], q_covars[i], 
                                          dim=self.out_dim) 
                                          for i in range(len(l_old))])
        t, p_value = ttest_indest_ind(l_old, l_new, equal_var=False)    
        return l_new, t, p if return_p_value else l_new, t

if __name__ == "__main__":
    clb = CLBaseline(None, None)

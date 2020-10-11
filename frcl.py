from torch import nn
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import Parameter
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D
from torch.distributions.kl import kl_divergence


class FRCL(nn.Module):
    def __init__(self, base_model, h_dim, sigma_prior=1):
        super(FRCL, self).__init__()
        self.base = base_model
        self.sigma_prior = sigma_prior
        self.L = Parameter(torch.eye(h_dim), requires_grad=True)
        self.mu = Parameter(torch.normal(0, 0.1, size=(h_dim,)), requires_grad=True)
        self.w_distr = MultivariateNormal(self.mu, scale_tril=self.L)
        self.w_prior = MultivariateNormal(torch.zeros(h_dim),
                                          covariance_matrix=sigma_prior*torch.eye(h_dim))
        self.quadr = GaussHermiteQuadrature1D()
        self.prev_tasks_distr = [] #previous tasks as torch distributions
        self.prev_tasks_tensors = [] #previous tasks as torch tensors
       
    def forward(self, x, target, N_k):
        """
        Return ELBO
        N_k = len(dataset), required for unbiased estimate through minibatch
        """
        elbo = 0
        phi = self.base(x)
       # crit = torch.nn.CrossEntropyLoss()
        for i in range(phi.shape[0]):
            def loglik(sample): #currently hardcoded for binary classification
                return -torch.log(1 + torch.exp(-target[i] * sample))
            mu = self.w_distr.mean
            cov = self.w_distr.covariance_matrix
            sample_dist = torch.distributions.Normal(torch.dot(phi[i], mu),
                                                     phi[i][None, :] @ cov @ phi[i][:, None])
            elbo += self.quadr(loglik, sample_dist)
        elbo /= x.shape[0] #mean
        
        kls = 0
        kls -= kl_divergence(self.w_distr, self.w_prior)

        for i in range(len(self.prev_tasks_distr)):
            phi_i = self.base(self.prev_tasks_tensors[i])
            cov_i = phi_i @ phi_i.T
            p_u = MultivariateNormal(torch.zeros(h_dim),
                                     covariance_matrix=cov_i * self.sigma_prior)
            kls -= kl_divergence(self.prev_tasks_distr[i], p_u)
        elbo += kls / N_k

        return elbo 

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
        sigma *= torch.eye(sigma.shape[0]) #we are interested only 
                                           #in diagonal part for inference
        return MultivariateNormal(loc=mu, covariance_matrix=sigma)

    @torch.no_grad()
    def predict(self, x, k, MC_samples=20):
        """Compute p(y) by MC estimate from q_\theta(f)?
        """
        distr = self.get_predictive(x, k)

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
            phi = self.base(X)
            mu_u = phi @ self.mu
            L_u = phi @ self.L
            self.prev_tasks_distr.append(MultivariateNormal(loc=mu_u,
                                                            scale_tril=L_u))
            self.prev_tasks_tensors.append(X)
            return
            
                
    @torch.no_grad()
    def detect_boundary(self, x, l_old):
        """Given new batch x and kl divergence for previous minibatch l_old
           compute l_new and perform statistical test
           Returns l_new and indicator of significance (0 or 1)
        """
        pass


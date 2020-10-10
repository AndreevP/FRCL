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
       
    def forward(self, x, target):
        """
        Return ELBO
        """
        elbo = 0
        phi = self.base(x)
       # crit = torch.nn.CrossEntropyLoss()
        for i in range(phi.shape[0]):
            def loglik(sample):
                return -torch.log(1e-8 + torch.exp(target[i] * sample))
            mu = self.w_distr.mean
            cov = self.w_distr.covariance_matrix
            sample_dist = torch.distributions.Normal(torch.dot(phi[i], mu),
                                                     phi[i][None, :] @ cov @ phi[i][:, None])
            elbo += self.quadr(loglik, sample_dist)

        elbo = -kl_divergence(self.w_distr, self.w_prior)

        for i in range(len(self.prev_tasks_distr)):
            phi_i = self.base(self.prev_tasks_tensors[i])
            cov_i = phi_i @ phi_i.T
            p_u = MultivariateNormal(torch.zeros(h_dim),
                                     covariance_matrix=cov_i * self.sigma_prior)
            elbo -= kl_divergence(self.prev_tasks_distr[i], p_u)

        return elbo

    def get_predictive(self, x, k):
        """ Computes predictive distribution according to section 2.5
            x - batch of data
            k - index of task
            Return predictive distribution q_\theta(f)
        """
        pass

    def predict(self, x, k):
        """Compute p(y) by MC estimate from q_\theta(f)?
        """
        pass


    def select_inducing(self, task_dataloader, criterion=None):
        """Given task dataloader compute inducing points
           Updates self.prev_tasks_distr and self.prev_tasks_tensors
        """
        pass
    
    def detect_boundary(self, x, l_old):
        """Given new batch x and kl divergence for previous minibatch l_old
           compute l_new and perform statistical test
           Returns l_new and indicator of significance (0 or 1)
        """
        pass

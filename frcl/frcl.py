import torch
import torch.nn as nn
from torch.nn import ParameterList, Parameter
import torch.nn.functional as F
from torch.utils.data import Subset

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

        if out_dim == 1:
            # computes loss
            self.loss_func = nn.BCEWithLogitsLoss()

            # predicts distribution over the classes
            def pred_func(input):
                pred = F.sigmoid(input)
                return torch.stack([1. - pred, pred]).squeeze()
        
            self.pred_func = pred_func
        
        if out_dim > 1:
            self.loss_func = nn.CrossEntropyLoss()
            self.pred_func = nn.Softmax()
        
        self.torch_gen = create_torch_random_gen(seed)
    
    def create_new_task(self):
        '''
        Create new task-specific tensor \omega
        '''
        w = Parameter(torch.randn(
            (self.out_dim, self.h_dim), generator=self.torch_gen)).to(self.device)
        self.tasks_omegas.append(w)

    def forward(self, cl_batch):
        '''
        Returns the objective with respect to (x, target)
        '''
        if len(self.tasks_replay_buffers) == 0:
            raise Exception("No tasks have been created yet!")

        # some consistency check
        assert(len(self.tasks_omegas) == 1 + len(self.tasks_replay_buffers))
        assert(len(cl_batch) == len(self.tasks_omegas))


        loss = self.loss_func(
            self.tasks_omegas[-1] @ self.base(cl_batch[-1][0]), 
            cl_batch[-1][1])

        for i in range(len(self.tasks_replay_buffers)):
            repl_buff = self.tasks_replay_buffers[i]
            omega = self.tasks_omegas[i]
            x = cl_batch[i][0]
            target = cl_batch[i][1]
            curr_unb_loss = self.loss_func(omega @ self.base(x), target)
            loss += (len(repl_buff.dataset)/float(len(repl_buff))) * curr_unb_loss
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
        distr = self.pred_func(omega @ self.base(x))
        if get_class:
            if len(distr.shape) == 1:
                return torch.argmax(distr).item()
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
            self.tasks_replay_buffers.append(Subset(task_dataset, select_indices))
        else:
            raise Exception(
                "Criterion {} not implemented".format(criterion))

if __name__ == "__main__":
    clb = CLBaseline(None, None)

import torch
import numpy as np
import sys
sys.path.append("../..")
from .estimators import AccuracyTasksEstimator
from ..datasets.split_mnist import SplitMnistDataset
from ..models import SplitMnistModel
from ..frcl import DeterministicCLBaseline, FRCL
from torch.utils.data import DataLoader
import abc
import matplotlib.pyplot as plt
from IPython.display import clear_output
from .state import State
from ..utils import StatCollector
import warnings
from tqdm import tqdm

def make_x_y(hist):
    x = np.concatenate([np.linspace(i, i + 1, len(hist[i]), endpoint=False) for i in range(len(hist))])
    y = np.concatenate([np.asarray(epoch_hist) for epoch_hist in hist])
    return x, y

class Pipeline(abc.ABC):

    def get_out_dim(self):
        return 1 if self.n_classes == 2 else self.n_classes

    def _mro_launch(self, name, reverse):
        self_class = self.__class__
        try:
            mro = self.__class__.__mro__[:-2]
            mro = tuple(reversed(mro)) if reverse else mro
            for _cls in mro:
                if name in _cls.__dict__.keys():
                    self.__class__ = _cls
                    getattr(self, name)()
        except:
            self.__class__ = self_class
            raise
        self.__class__ = self_class

    def _make_value_per_task(self, name, value):
        if not isinstance(value, list):
            setattr(self, name, [value]*len(self.task_provider))
            return
        setattr(self, name, value)

    def __init__(self, task_provider, n_inducing=10, 
        inducing_criterion='random', n_repeat=10, device='cuda', 
        batch_size=32, lr=1e-3, n_epochs=10, n_classes=None):
        '''
        :Parameters:
        task_provider: provider for subsequent tasks, must implement __getitem__, 
        __len__ and model property (which returns base_model)
        n_inducing: int: count of points to induce on each iteration
        inducing_criterion: str: criterion used for points inducing
        n_repeat: int : count of experiments to repeat 
        device : device to train the models
        batch_size : int or list of ints : batch size used for training each task
        lr : float or list of floats : learning rate used for training each task
        n_epochs : int or list of ints : count of epochs used for training each task
        n_classes : int : count of target classes of the tasks (BECOME OBSOLETE NOW)
        '''
        if n_classes is not None:
            warnings.warn("n_classes option not supported now")
        # self.out_dim = 1 if self.n_classes == 2 else self.n_classes
        self.task_provider = task_provider
        self.n_inducing = n_inducing
        self.inducing_criterion = inducing_criterion
        self.n_repeat = n_repeat
        self.device = device
        self._make_value_per_task('batch_size', batch_size)
        self._make_value_per_task('lr', lr)
        self._make_value_per_task('n_epochs', n_epochs)
        self.state = State()
        self.state.attach_level()
    
    def _before_run(self):
        self._mro_launch('before_run', True)
    
    def before_run(self):
        pass

    def _before_cl_cycle(self):
        self._mro_launch('before_cl_cycle', True)

    def before_cl_cycle(self):
        pass

    def _before_task_train(self):
        self._mro_launch('before_task_train', True)

    def before_task_train(self):
        pass

    def _before_epoch_train(self):
        self._mro_launch('before_epoch_train', True)

    def before_epoch_train(self):
        pass

    def _before_batch_train(self):
        self._mro_launch('before_batch_train', True)

    def before_batch_train(self):
        pass

    @abc.abstractmethod
    def create_cl_model(self):
        pass

    @abc.abstractmethod
    def compute_loss(self):
        pass

    def _after_batch_train(self):
        self._mro_launch('after_batch_train', False)

    def after_batch_train(self):
        pass

    def _after_epoch_train(self):
        self._mro_launch('after_epoch_train', False)

    def after_epoch_train(self):
        pass

    def _after_task_train(self):
        self._mro_launch('after_task_train', False)

    def after_task_train(self):
        pass

    def _after_cl_cycle(self):
        self._mro_launch('after_cl_cycle', False)

    def after_cl_cycle(self):
        pass

    def _after_run(self):
        self._mro_launch('after_run', False)

    def after_run(self):
        pass

    def _get_current_return(self):
        _ret = self.run_return()
        if _ret is not None:
            self._run_output.append(_ret)

    def _run_return(self):
        self._run_output = []
        self_class = self.__class__
        try:
            mro = self.__class__.__mro__[:-2]
            for _cls in mro:
                if 'run_return' in _cls.__dict__.keys():
                    self.__class__ = _cls
                    _ret = self.run_return()
                    if _ret is not None:
                        self._run_output.append(_ret)
        except:
            self.__class__ = self_class
            raise
        self.__class__ = self_class
        return self._run_output

    def run_return(self):
        pass
    
    def run(self):
        self._before_run()
        for i_repeat in range(self.n_repeat):
            self.i_repeat = i_repeat
            self.n_classes = self.task_provider[0][-1]
            self.base_model = self.task_provider.model.to(self.device)
            self.cl_model = self.create_cl_model()

            self._before_cl_cycle()
            for i_task in range(len(self.task_provider)):
                self.i_task = i_task
                train_ds, test_ds, n_classes = self.task_provider[i_task]
                self.n_classes = n_classes
                self.train_ds = train_ds
                self.test_ds = test_ds
                self.train_dl = DataLoader(
                    self.train_ds, batch_size=self.batch_size[i_task], shuffle=True)
                self._before_task_train()
                self.optim = torch.optim.Adam(self.cl_model.parameters(), lr=self.lr[self.i_task])
                for i_epoch in range(self.n_epochs[self.i_task]):
                    self.i_epoch = i_epoch
                    self._before_epoch_train()
                    for i_batch, batch in tqdm(
                            enumerate(self.train_dl), 
                            total=len(self.train_dl), 
                            desc="Task: {}; Epoch: {}".format(self.i_task, self.i_epoch)):
                        self.i_batch = i_batch
                        self.optim.zero_grad()
                        self.X = batch[0]
                        self.target = batch[1]
                        self._before_batch_train()
                        self.X = self.X.to(self.device)
                        self.target = self.target.to(self.device)
                        self.loss = self.compute_loss()
                        self.loss.backward()
                        self._after_batch_train()
                        self.optim.step()
                    self._after_epoch_train()
                self._after_task_train()
            self._after_cl_cycle()
        self._after_run()
        return self._run_return()
    
class StatDrawer(Pipeline):

    def clear_output(self):
        if self.draw_ticket:
            clear_output(wait=True)
            self.draw_ticket = False

    def __init__(self, *args, upd_factor=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.upd_factor = upd_factor
    
    def before_epoch_train(self):
        self.draw_ticket = True

class FRCLStatsObsPipeline(StatDrawer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def before_task_train(self):
        s = State()
        s.attach_level(
            e = StatCollector(self.upd_factor), 
            elbo=StatCollector(self.upd_factor), 
            kls=[StatCollector(self.upd_factor) for _ in range(1 + self.i_task)], 
            kls_div_nk=StatCollector(self.upd_factor))
        self.state.frcl_stat = s

    def after_batch_train(self):
        frcl_stat = self.state.frcl_stat
        state = self.state
        frcl_stat.e.add(state.e)
        frcl_stat.elbo.add(state.elbo)
        frcl_stat.kls_div_nk.add(state.kls_div_nk)
        for i in range(len(frcl_stat.kls)):
            frcl_stat.kls[i].add(state.kls[i])
    
    def after_epoch_train(self):
        self.clear_output()
        print("N tasks: {}, Epoch {}".format(self.i_task + 1, self.i_epoch))
        frcl_stat = self.state.frcl_stat
        e_hist = frcl_stat.e.get_history()
        x = np.arange(len(e_hist))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(x, e_hist, label='e history')
        elbo_hist = frcl_stat.elbo.get_history()
        axes[0].plot(x, elbo_hist, label='elbo history')
        kls_div_nk_hist = frcl_stat.kls_div_nk.get_history()
        axes[0].plot(x, kls_div_nk_hist, label = 'kls div nk history')
        axes[0].legend()
        for i in range(len(frcl_stat.kls)):
            hist = frcl_stat.kls[i].get_history()
            axes[1].plot(x, hist, label='kls task {}'.format(i))
        if len(frcl_stat.kls) > 0:
            axes[1].legend()
        plt.show()


class LossEstPipeline(StatDrawer):
    '''
    This class hels to trace and 
    visualize loss during training
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def before_task_train(self):
        self.loss_hist = []
    
    def before_epoch_train(self):
        self.cumm_loss = 0.
        self.loss_hist.append([])

    def after_batch_train(self):
        self.cumm_loss += self.loss.item()
        if (self.i_batch + 1) % self.upd_factor == 0:
            self.cumm_loss /= self.upd_factor
            self.loss_hist[-1].append(self.cumm_loss)
            self.cumm_loss = 0.
    
    def after_epoch_train(self):
        self.clear_output()
        x, y = make_x_y(self.loss_hist)
        plt.plot(x, y)
        plt.title('Loss_{}'.format(self.i_task))
        plt.show()        
    
class AccEstPipeline(Pipeline):
    '''
    This class helps to evaluate accuracy of 
    different tasks during training.
    See AccuracyTasksEstimator to get into details
    '''

    def before_run(self):
        self.acc_estimators = []
    
    def before_cl_cycle(self):
        self.acc_estimator = AccuracyTasksEstimator()
    
    def before_task_train(self):
        self.acc_estimator.register_task(self.test_ds)
    
    def after_task_train(self):
        self.acc_estimator(self.cl_model)
    
    def after_cl_cycle(self):
        self.acc_estimators.append(self.acc_estimator)
    
    def run_return(self):
        return self.acc_estimators

class BaselinePipeline(Pipeline):
    '''
    This class implements pipeline for 
    training baseline model
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_cl_model(self):
        multiclass = True
        if self.n_classes == 2:
            multiclass = False
        cl_model = DeterministicCLBaseline(
            self.base_model, self.base_model.hid, device=self.device, multiclass=multiclass)
        return cl_model

    def before_task_train(self):
        self.cl_model.create_new_task(self.get_out_dim())
    
    def before_batch_train(self):
        if self.n_classes == 2:
            self.target = self.target.float()
            return
        self.target = self.target.long()
    
    def after_task_train(self):
        self.cl_model.select_inducing(
            self.train_ds, N=self.n_inducing, criterion=self.inducing_criterion)

    def compute_loss(self):
        return self.cl_model(self.X, self.target)

class FRCLPipeline(Pipeline):
    '''
    This class implements pipeline for training
    FRCL model
    '''

    def __init__(self, *args, sigma_prior = 1, int_compute_method='SVI', N_samples=20, **kwargs):
        '''
        :Parameters:
        int_compute_method: str: method to comute expectation
        N_samples : int : count of points to estimate expectation
        '''
        super().__init__(*args, **kwargs)
        self.sigma_prior = sigma_prior
        self.int_compute_method = int_compute_method
        self.N_samples = N_samples

    def create_cl_model(self):
        multiclass = True
        if self.n_classes == 2:
            multiclass = False
        cl_model = FRCL(
            self.base_model, self.base_model.hid, 
            self.device, sigma_prior=self.sigma_prior, multiclass=multiclass)
        return cl_model
    
    def before_task_train(self):
        self.cl_model.create_new_task(self.get_out_dim())

    def before_batch_train(self):
        if self.n_classes == 2:
            self.target = self.target.float()
            return
        self.target = self.target.long()
    
    def after_task_train(self):
        self.cl_model.select_inducing(
            self.train_dl, N=self.n_inducing, criterion=self.inducing_criterion)

    def compute_loss(self):
        return self.cl_model(
                self.X, self.target, len(self.train_dl) * self.batch_size[self.i_task], 
                method = self.int_compute_method, N_samples=self.N_samples)

class FRCLPipelineState(FRCLPipeline):

    def compute_loss(self):
        return self.cl_model(
                self.X, self.target, len(self.train_dl) * self.batch_size[self.i_task], 
                method = self.int_compute_method, N_samples=self.N_samples, state=self.state)


class BaselineTrainDemo(BaselinePipeline, AccEstPipeline, LossEstPipeline):
    '''
    Baseline pipeline with evaluating accuracy
    and drawing losses
    '''
    pass

class FRCLTrainDemo(FRCLPipeline, AccEstPipeline, LossEstPipeline):
    '''
    FRCL pipeline with evaluationg accuracy
    and drawing losses
    '''
    pass

class FRCLStatTrace(FRCLPipelineState, AccEstPipeline, FRCLStatsObsPipeline, LossEstPipeline):
    pass
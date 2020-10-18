import torch
import abc
from torch.utils.data import DataLoader
import numpy as np

class TaskEstimator(abc.ABC):

    def __init__(self, n, test_dataset, batch_size=32):
        '''
        Creates task 
        :Parameters:
        n: number of the task
        test_dataset: test_dataset corresponding to the task
        cl_solver: solver of the task
        '''
        self.n = n
        self.test_dataset = test_dataset
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=batch_size, drop_last=False)
        self.estimations = {}
    
    @abc.abstractmethod
    def estimate(self, cl_solver):
        pass

    def __call__(self, cl_solver):
        n_solved = 0
        try:
            n_solved = len(cl_solver)
        except AttributeError:
            raise Exception("cl_solver must implement __len__ method")

        if n_solved <= self.n:
            raise Exception("can not estimate unsolved task; "
                "solved tasks: {}, task to estimate: {}".format(n_solved, self.n))
        
        res = self.estimate(cl_solver)
        self.estimations[n_solved] = res

class AccuracyTaskEstimator(TaskEstimator):

    def __init__(self, n, test_dataset, batch_size=32):

        super().__init__(n, test_dataset, batch_size=batch_size)
    
    def estimate(self, cl_solver):
        acc = 0.
        for x, target in self.test_dataloader:
            prediction = cl_solver.predict(x, self.n)
            if len(prediction.shape) > 1:
                x_pred = torch.argmax(prediction, dim=-1).cpu().numpy()
            else:
                x_pred = (prediction.cpu().numpy() > 1/2).astype(int)
            acc += np.sum(x_pred == target.cpu().numpy())
        acc /= len(self.test_dataloader.dataset)
        return acc

class TasksEstimator(abc.ABC):

    @abc.abstractproperty
    def task_estimator_class(self):
        pass

    def __init__(self):
        self.tasks = []
        
    def register_task(self, test_dataset):
        '''
        Creates new task with given test_dataset
        '''
        n = len(self.tasks)
        self.tasks.append(self.task_estimator_class(n, test_dataset))
    
    def __call__(self, cl_solver):
        n_solved = 0
        try:
            n_solved = len(cl_solver)
        except AttributeError:
            raise Exception("cl_solver must implement __len__ method")
        for i in range(n_solved):
            self.tasks[i](cl_solver)
    
    def get_task_estimations(self, n, format='plt'):
        '''
        Returns estimations of the task n
        at different moments of cl_solver training
        :Parameters:
        n: int: number of the task
        '''
        if format == 'plt':
            if not n in range(len(self.tasks)):
                return [], []
            est = self.tasks[n].estimations
            est = list(est.items())
            est.sort(key=lambda x: x[0])
            est = list(zip(*est))
            return list(est[0]), list(est[1])
        
        raise Exception(
            "format type '{}' is not implemented".format(format))
    
    def get_solver_estimations(self, n, format='plt'):
        '''
        Returns estimations of tasks
        when n tasks were solved
        :Parameters:
        n: int: number of solved tasks
        '''
        if format == 'plt':
            n_tasks = []
            values = []
            for i in range(len(self.tasks)):
                task = self.tasks[i]
                try:
                    val = task.estimations[n]
                    n_tasks.append(i)
                    values.append(val)
                except KeyError:
                    pass
            return n_tasks, values
        
        raise Exception(
            "format type '{}' is not implemented".format(format))

class AccuracyTasksEstimator(TasksEstimator):

    @property
    def task_estimator_class(self):
        return AccuracyTaskEstimator

        
            



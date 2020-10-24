from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
Path("./results").mkdir(parents=True, exist_ok=True)
from frcl.tasks import SplitMnistTask, PermutedMnistTask, OmniglotTask
from frcl.experiments.pipeline import RS_FRCLTrain, RS_FRCLTrainExponMuSchedule, RS_BaselineTrain
import seaborn as sns
sns.set()

settings = {
    'split_mnist': {
        'task': SplitMnistTask(),
        'methods': {
            'baseline' : [
                    {
                        'pipeline': RS_BaselineTrain,
                        'settings': {
                            'n_epochs': 4, 
                            'lr': 1e-5, 
                            'batch_size': 100, 
                            'n_inducing': 2
                        }
                    },
                    {
                        'pipeline': RS_BaselineTrain,
                        'settings': {
                            'n_epochs': 4,
                            'lr': 1e-5,
                            'batch_size': 100,
                            'n_inducing': 40
                        }
                    }
                ],
            'frcl_random': [
                    {
                        'pipeline': RS_FRCLTrain,
                        'settings': {
                            'n_epochs': 4,
                            'lr': 1e-4,
                            'batch_size': 64,
                            'n_inducing': 2,
                            'init_mu_sigma':1.
                        }
                    },
                    {
                        'pipeline': RS_FRCLTrain,
                        'settings': {
                            'n_epochs': 4,
                            'lr': 1e-4,
                            'batch_size': 64,
                            'n_inducing': 40,
                            'init_mu_sigma':1.
                        }
                    }
                ],
            'frcl_trace': [
                    {
                        'pipeline': RS_FRCLTrain,
                        'settings': {
                            'n_epochs': 4,
                            'lr': 1e-4,
                            'batch_size': 64,
                            'init_mu_sigma':1.,
                            'inducing_criterion': "deterministic",
                            'n_inducing': 2
                        },
                    },
                    {
                        'pipeline': RS_FRCLTrain,
                        'settings': {
                            'n_epochs': 4,
                            'lr': 1e-4,
                            'batch_size': 64,
                            'init_mu_sigma':1., 
                            'inducing_criterion': "deterministic",
                            'n_inducing': 40
                        }
                    }
                ]
            }
    },

    'permuted_mnist': {
        'task': PermutedMnistTask(10), 
        'methods': {
            'baseline': [
                    {
                        'pipeline': RS_BaselineTrain,
                        'settings': {
                            'n_epochs': 6,
                            'lr': 1e-4,
                            'batch_size': 100,
                            'n_inducing': 10
                        }
                    },
                    {
                        'pipeline': RS_BaselineTrain,
                        'settings': {
                            'n_epochs': 6,
                            'lr': 1e-4,
                            'batch_size': 100,
                            'n_inducing': 200                    
                        }
                    }
                    {
                        'pipeline': RS_BaselineTrain,
                        'settings': {
                            'n_epochs': 6,
                            'lr': 1e-4,
                            'batch_size': 100,
                            'n_inducing': 80
                        }
                    }
                ],
            'frcl_random': [
                    {
                        'pipeline': RS_FRCLTrainExponMuSchedule,
                        'settings': {
                            'n_epochs': 8,
                            'lr': 1e-4,
                            'init_mu_sigma': 0.04,
                            'batch_size': 32,
                            'n_inducing': 10
                        }
                    },               
                    {
                        'pipeline': RS_FRCLTrain,
                        'settings': {
                            'n_epoch': 8,
                            'batch_size': 64,
                            'lr': 1e-4,
                            'init_mu_sigma':1., 
                            'n_inducing': 200
                        }
                    }
            ]
        }
    },
    'omniglot': {
        'task': OmniglotTask(n_tasks=10),
        'methods': {
            'baseline': [
                    {
                        'pipeline': RS_BaselineTrain,
                        'settings': {
                            'n_epochs': 12,
                            'lr': 1e-3,
                            'batch_size': 64,
                            'n_inducing': 60,
                        },
                    }
                ],
            'frcl_random': [
                    {
                        'pipeline': RS_FRCLTrain,
                        'settings': {
                            'n_epochs': 8,
                            'lr': 1e-3,                            
                            'init_mu_sigma':1., 
                            'n_inducing': 60
                        }
                    }
            ]
        }
    }
}

parser = argparse.ArgumentParser(description='Runs CL experiments')
parser.add_argument('--device', action='store', type=str)
parser.add_argument('--task', action='store', type=str, required=True)
parser.add_argument('--method', action='store', type=str, required=True)
parser.add_argument('--n_inducing', action='store', type=int, required=True)

args = parser.parse_args()

assert args.task in settings.keys(), "'task' must be in {}, got '{}' instead".format(
    str(list(settings.keys())), args.task)
task_params = settings[args.task]
task = task_params['task']
methods_params = task_params['methods']

assert args.method in methods_params.keys(), "'method' must be in {}, got '{}' instead".format(
    str(list(methods_params.keys())), args.method)

method_list = methods_params[args.method]

assert args.n_inducing in [m['settings']['n_inducing'] for m in method_list], "'n_inducing' must be in {}, got '{}' instaed".format(
    str([m['settings']['n_inducing'] for m in method_list]), args.n_inducing)

method = None
for _method in method_list:
    if _method['settings']['n_inducing'] == args.n_inducing:
        method = _method
        break

p_settings = method['settings']

p_settings['device'] = args.device
p_settings['draw_each_epoch'] = False
pipeline = method['pipeline'](task, **p_settings)
res = pipeline.run()

for i in range(len(task)):
    x, y = res[0][0].get_task_estimations(i)
    plt.plot(x, y, "-o", label = "task {}".format(i))
    plt.xlabel('observed tasks')
    plt.ylabel('accuracy')
    plt.legend()

mean = np.mean(res[0][0].get_solver_estimations(len(task))[1])

file_name = "./results/res_task_'{}'_method_'{}'_n_inducing_'{}'_mean_'{}'.png".format(
    args.task, args.method, args.n_inducing, round(mean, 3))
plt.savefig(file_name)



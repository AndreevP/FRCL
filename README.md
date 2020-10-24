# FRCL
Functional Regularisation for Continual Learning with Gaussian Processes

by Pavel Andreev, Peter Mokrov and Alexander Kagan

This is an unofficial PyTorch implementation of the paper https://arxiv.org/abs/1901.11356 . The main goal of this project is to provide an independent reproduction of the results presented in the paper.

**Project Proposal**: [pdf](https://drive.google.com/file/d/1AoGfMXKVplaxxKazk3kX9us1z8ZypY7t/view?usp=sharing)

## Experiments launching

To launch our experiments use `results_script.py`
The example of script run below:

```bash
> python .\results_script.py --device 'your device' --task 'permuted_mnist' --method 'baseline' --n_inducing 2
```
Available options for ```--task``` argument are ```split_mnist```, ```permuted_mnist``` and ```omniglot```. 
Available options for ```--method``` argument are ```baseline```, ```frcl_random``` and ```frcl_trace```. 

Results of our experiments are presented in '.\results'. 
Besides, one can find notebooks with minimal working examples in '.\notebooks'.

## Results

The presentation with the project main results is available [here](https://drive.google.com/file/d/1Bur5QPy8DoAhv8c-fbr7eFIA7Nlj-X17/view?usp=sharing).

We results are also summarized in the table below.

| Datset | Method | N points | Criteria  | Accuracy (ours)| Accuracy (paper)|
| ------------- | ------------- | ------------- | ------------- | ------------- |  ------------- | 
| Split-MNIST  | baseline  | 2 | -  | 0.981 | - |
| Split-MNIST  | baseline   | 40 | -  | 0.985 | 0.958 |
| Split-MNIST  | FRCL   | 2 | Random  | 0.827 | 0.598|
| Split-MNIST  | FRCL   | 2 | Trace  | 0.82 | 0.82 |
| Split-MNIST  | FRCL   | 40 | Random  | 0.986 | 0.971 |
| Split-MNIST  | FRCL   | 40 | Trace  | 0.979 | 0.978 |
| Permuted-MNIST  | baseline  | 10 | -  | 0.695 | 0.486|
| Permuted-MNIST | baseline   | 80 | -  | 0.865 | - |
| Permuted-MNIST | baseline   | 200 | -  | 0.908 | 0.823 |
| Permuted-MNIST  | FRCL  | 10 | Random  | 0.628/0.527* | 0.482|
| Permuted-MNIST  | FRCL  | 80 | Random  | 0.838 | - |
| Permuted-MNIST  | FRCL   | 200 | Random  | 0.942 | 0.943 | 
| Omniglot-10  | baseline  | 60 | -  | 0.381 | - | 
| Omniglot-10  | FRCL  | 60 | Random  | 0.376 | - |

Results of our experiments are presented in `.\results`

\* the results appeared to significantly depend on initialization of parameters
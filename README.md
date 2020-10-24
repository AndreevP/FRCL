# FRCL
Functional Regularisation for Continual Learning with Gaussian Processes

by Pavel Andreev, Peter Mokrov and Alexander Kagan

This is an unofficial PyTorch implementation of the paper https://arxiv.org/abs/1901.11356 . The main goal of this project is to provide an independent reproduction of the results presented in the paper.

**Project Proposal**: [pdf](https://drive.google.com/file/d/1AoGfMXKVplaxxKazk3kX9us1z8ZypY7t/view?usp=sharing)

## Experiments launching

To launch our experiments use 'results_script.py'
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

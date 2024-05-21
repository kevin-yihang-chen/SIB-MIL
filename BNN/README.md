# R2D2-Net: Shrinking Bayesian Neural Networks via R2D2 Prior
Code repository for "R2D2-Net: Shrinking Bayesian Neural Networks via R2D2 Prior"

## R2D2 Layers
The implementations of R2D2 layers together with other BNN layers can be found in the `./configs` directory. 

## Architectures
The neural network architectures (e.g., LeNet) composed by the BNN layers can be found in the `./models` directory. Users can customize own architectures also.

## Get Started
Create a yaml config file in the `./configs` directory (examples can be found in the same directory), and run the following codes to run an experiment
```
python main.py
```
Results will be saved in `./checkpoints'.

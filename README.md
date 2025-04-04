# MAD RL Policy

PyTorch implementation of policy paramertizations as described in 
"MAD: A Magnitude And Direction Policy Parametrization for Stability Constrained Reinforcement Learning."

## Installation

```bash
git clone https://github.com/DecodEPFL/performance-boosting_controllers.git
cd performance-boosting_controllers
python setup.py install
```

## Policy Parametrizations

We provide the implementation of four policy parametrizations:
1. MAD Policy: $\quad$ $~$ $u_t = |\mathtt{LRU}_t(\theta_1)(\widehat{w}\_{t:0}) + \mathtt{LRU}_t(\theta_2)(x\_0) | \tanh(\mathtt{NN(x_t, \psi)})$
2. MA Policy: $\quad$ $~~~$ $u_t = |\mathtt{LRU}_t(\theta_1)(\widehat{w}\_{t:0}) + \mathtt{LRU}_t(\theta_2)(x\_0) |$
3. AD Policy: $\quad$ $~~~~$ $u_t = | \mathtt{LRU}_t(\theta_2)(x\_0) | \tanh(\mathtt{NN(x_t, \psi)})$
4. DDPG Policy: $\quad$ $u_t = \mathtt{NN(x_t, \psi)}$

## Basic Usage

An environment of mobile robots in the <i>xy</i>-plane is proposed to train the MAD RL Policies.
We propose the problem _mountains_, where two agents need to pass through a narrow corridor 
while avoiding collisions.

To train the controllers, run the following script:
```bash
cd training
python3 train.py --controller_type [CONTROLLER_TYPE] --tag [MODEL_TAG] --episodes [NUM_EPISODES]
```
where available values for `CONTROLLER_TYPE` are `MAD`, `MA`, `AD` and `DDPG`. 
The `MODEL_TAG` and `NUM_EPISODES` can be set to any natural numbers for naming the model and setting the number of episodes for training respectively. 

To validate the trained controllers, run the following script:
```bash
cd validation
python3 validate.py --controller_type [CONTROLLER_TYPE] --tag [MODEL_TAG] --validation_type [VALIDATION_TYPE]
```
where `VALIDATION_TYPE` can be either from `Training` or `Generalization` initializations.

## Examples

The following gifs show the closed loop trajectories of the 2 robots with the trained MAD Policy based controller. Validation of the trained policies is done by sampling initial conditions from the same neighborhoods around those seen in training. Generalization capabilities are assessed by interchanging the distributions of the initial conditions of the two agents.

<p align="center">
<img src="animations/validation.gif" alt="robot_trajectories_validation" width="400"/>
<img src="animations/generalization.gif" alt="robot_trajectories_generalization" width="400"/>
</p> 

## License

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by] 

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

## References
[[1]](https://arxiv.org/pdf/2504.02565v1) Luca Furieri, Sucheth Shenoy, Danilo Saccani, Andrea Martin, Giancarlo Ferrari-Trecate.
"MAD: A Magnitude And Direction Policy Parametrization for Stability Constrained Reinforcement Learning,"
arXiv:2504.02565, 2025.

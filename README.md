# jsbsim-rl
RL compatible use of the JSBSim simulator

## Installation
Requires python 3.10 or 3.11
```
pip install -r pip_requirements.txt
pip install -e .
```
or
```
conda env create --file environment_cross_platform.yml
pip install -e .
```

## Useful files
jsbgym/envs/ contains the gymnasium envs and tasks.
jsbgym/ppo_train, jsbgym/td3_train.py are the training scripts.
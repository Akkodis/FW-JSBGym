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
fw_jsbgym/envs/ contains the gymnasium envs and tasks.
fw_jsbgym/ppo_train, fw_jsbgym/td3_train.py are the training scripts.
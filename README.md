# Risk-sensitive-RL

Code base for paper: [Adaptive Risk Tendency: Nano Drone Navigation in Cluttered Environments with Distributional Reinforcement Learning]

## Dependencies

+ python 3.6+
+ gym 0.10.5
+ numpy 1.19.5
+ pytorch 1.9.0

## Installation of Crazyflie Training Environment

```shell
git clone https://github.com/tudelft/risk-sensitive-rl.git
cd risk-sensitive-rl/crazyflie-env
python3 -m pip install -e '.[crazyflie-env]'
```

## Import Crazyflie Environment

```python
import gym, crazyflie_env
env = gym.make('CrazyflieEnv-v0') # env id
```
Note that the interface for this environment is slightly different than OpenAI gym. By calling `env.step(<action>)`, the returned `state` is a object that need to be transferred to numpy arrays. This is implemented by a transfer function series in `art-iqn/utils/util.py`.

For the details of using this environment, refer to `art-iqn/train.py` or `art-iqn/tactical.py`.

## Reproduction

To reproduce our real-world experiments, one needs a *crazyflie* drone system equiped with *multiranger* deck and an global position system like OptiTrack to locate the drone's absolute position. Onboard position estimation system can also be used but there maybe exist large drift.

### IQN policy with fixed CVaR values

```shell
python3 run_cyber_zoo.py --cvar=<your_cvar_value> --exp_id='<your_experiment_id>'
```

### ART-IQN

```shell
python3 run_cyber_zoo_art.py 
```

### Video
Supplementary video for our real-world experiment can be found in `art-iqn/experiments/videos`.
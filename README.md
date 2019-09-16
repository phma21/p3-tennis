# Udacity Deep Reinforcement Learning nano degree

> This repo was forked from https://github.com/ShangtongZhang/DeepRL . Big thanks to the author for providing well 
structured implementations of popular algorithmns, as well as a nice framework for evaluation!

## Project 2: Continuous control

~~This repository contains my solution to the second project of Udacity deep RL Nanodegree. The goal of this project is
to solve the Unity Agent environment Reacher.~~

~~TAfter forking from the original implementation, I needed to adapt the unity agents environment provided by Udacity to work with the original code.
To achieve this, I wrapped the Unity Agents environment into a gym.Wrapper object, such that the Reacher environment behaves like a gym one.~~

~~After some time spent on tuning the parameters for PPO, as well as reading some resources how to tune PPO, 
I was able to solve the Reacher environment.~~


### Tennis env
~~The goal of Reacher environment is to learn how to move and hold a double jointed robot arm into a spheric area.
To make things harder, that area moves around the arm at different speed in every episode.
There are 4 continuous actions possible, corresponding to the torque applicable to the joints.
The observation space contains 26 variables corresponding to physical attributes of the joints and the sphere.
The agent is rewarded with +0.1 if the agent's hand rests inside the goal area for one step, zero reward otherwise.
The environment is considered to be solved if an agent manages to achieve a mean reward of 30 over a series of 100 episodes.~~


### Lessons learned



### Installation
To build the complete environment with Mojoco and all contents of the original framework, you may have a look at the 
original readme (ORIGINAL_README.md) and/or follow the steps performed in the Dockerfile.

For a more simple setup sufficient to run Tennis Env, you may follow those steps:

* `
pip3 install -r requirements_just_reacher.txt
`
* `
pip3 install git+https://github.com/openai/baselines.git@8e56dd#egg=baselines
`
* `
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip
`
`
unzip Tennis_Linux.zip
`
> If pip install gets you "ERROR: unityagents 0.4.0 has requirement protobuf==3.5.2, but you'll have protobuf 3.9.1 which is incompatible.",
no worries, it still works ;-)

### How to run
`
python3 examples.py
`

~~By default, it will load the best performing model in evaluation mode.
To train the agent, all you need to do is comment out loading of the model (line 437) and uncomment line 439.~~


### Results & details
See Report.md
# Udacity Deep Reinforcement Learning nano degree

> This repo was forked from https://github.com/ShangtongZhang/DeepRL . Big thanks to the author for providing well 
structured implementations of popular algorithmns, as well as a nice framework for evaluation!

## Project 2: Continuous control

This repository contains my solution to the second project of Udacity deep RL Nanodegree. The goal of this project is
to solve the Unity Agent environment Reacher.

After forking from the original implementation, I needed to adapt the unity agents environment provided by Udacity to work with the original code.
To achieve this, I wrapped the Unity Agents environment into a gym.Wrapper object, such that the Reacher environment behaves like a gym one.

After some time spent on tuning the parameters for PPO, as well as reading some resources how to tune PPO, 
I was able to solve the Reacher environment.


### Lessons learned
- In comparision to the environments of OpenAI Gym, the Reacher environment appears to be rather hard to solve and the rewards are a lot more sparse. 
Therefore, the amount of collected trajectories for each network update played a big role. I ended up collecting about 8 episodes for each optimization phase.
- A bigger batch size is almost always better
- A bigger rollout is always better, but of course slows down your training. For noisy environments, the rollout should cover multiple episodes
- PPO is a really good algorithm which - in contrast to others - does not suffer performance drops in later stages (given that parameters are tuned properly)



### Installation
To build the complete environment with Mojoco and all contents of the original framework, you may have a look at the 
original readme (ORIGINAL_README.md) and/or follow the steps performed in the Dockerfile.

For a more simple setup sufficient to run Reacher Env, you may follow those steps:

* `
pip3 install -r requirements_just_reacher.txt
`
* `
pip3 install git+https://github.com/openai/baselines.git@8e56dd#egg=baselines
`
* `
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip
`
`
unzip Reacher_Linux.zip
`
> If pip install gets you "ERROR: unityagents 0.4.0 has requirement protobuf==3.5.2, but you'll have protobuf 3.9.1 which is incompatible.",
no worries, it still works ;-)

### How to run
`
python3 examples.py
`

By default, it will load the best performing model in evaluation mode.
To train the agent, all you need to do is comment out loading of the model (line 437) and uncomment line 439.


### Results & details
See Report.md
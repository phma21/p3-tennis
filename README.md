# Udacity Deep Reinforcement Learning nano degree

> This repo was forked from https://github.com/ShangtongZhang/DeepRL . Big thanks to the author for providing well 
structured implementations of popular algorithms, as well as a nice framework for evaluation!

## Project 3: Tennis

This repository contains my solution to the third project of Udacity deep RL Nanodegree. The goal of this project is
to solve the Unity Agents environment Tennis.

After forking from the original implementation, I needed to adapt the unity agents environment provided by Udacity to work with the original code.
To achieve this, I wrapped the Unity Agents environment into a gym.Wrapper object, such that Tennis environment behaves like a gym one.
Different to the second project, I needed to adapt the original PPO code to work with the two agent setup that Tennis Env is using. 
The good part is that I was able to spend some more time understanding the PPO implementation.

### Tennis env
The goal of Tennis environment is to teach two agents to collaboratively play tennis with each other.
To solve the environment, the agents must be able to consistently pass to ball over the net, without hitting it or overshot 
the other player's area at least for a few times in each episode.

There are 2 continuous actions possible: The location, moving left or right, and jumping.
Each agent perceives it's own, local environment with 8 variables, stacked 3 times to capture time context, 
making it a total of 24 observations.
Each agent is rewarded +0.1 if he hits the ball into the other player's area.

The reward for both agents is calculated by max(reward_agent_1, reward_agent_2).
The environment is considered to be solved if both agents achieve a reward of > 0.5 for a series of 100 consecutive episodes.

### Lessons learned
* I used the very same set of parameters that solved the Reacher env (project 2) as well, so it appear's that those environments
are fairly similar to each other.
* Training an agent through collaborative self-play may end up in a local minima, as the agent learns to go easy on it's opponent


### Installation
To build the complete setup with Mojoco and all contents of the original framework, you may have a look at the
original readme (ORIGINAL_README.md) and/or follow the steps performed in the Dockerfile.

For a more simple setup sufficient to run Tennis Env, you may follow those steps:

* `
pip3 install -r requirements_just_tennis.txt
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

By default, it will load the best performing model and enable you to **play against one of the agents yourself!**
Use the arrow keys on your keyboard to move left and right, as well as jumping.

To train the agent, all you need to do is comment out loading of the model (line 441) and uncomment line 444.

To evaluate on 100 episodes, comment in line 447. Be aware that this takes a while.

### Results & details
See Report.md
# Udacity Deep Reinforcement Learning Nano degree

> This repo was forked from todo, big thanks!

## Project 2: Continious control

This repository contains my solution to the second project of Udacity deep RL Nanodegree.

- fork shaangtong
- create a openai gym wrapper for unity agent env (note that for newer versions this exisists for unity agent as well)
- after initial failed attempts, get some help from: <websites> to try different params
https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
https://github.com/llSourcell/Unity_ML_Agents/blob/master/docs/best-practices-ppo.md
- turns out that compared to gym, I need much larger batch sizes and rollouts (I use 8 episodes now -> ~8000 steps)


lessons learned:
- bigger batch size == always better
- bigger rollout == always better, however slower learning, as we need to throw away after each network update step
- PPO is a really good algorithm and shangtongs implementation / eval framework is awesome


details:
network: both actor and critic (advantage estimator): (256, 256)
- copy rest of params over


eval:
- after training with seed x for 1 Mio iterations, i run eval on 100 episodes.
result: episodic_return_test mean: 37.66 std: 0.22

![eval image](good_models/PPO-eval.png "eval image]")
![train image](good_models/PPO-train.png "train image")



installation:
- docker build .., or use dockerfile provided at: todo!

further steps:
- more finetuning, could very easily try out the other algorithmns implemented here, like DDPG or TD3


# todo: 
- tell seed and adapt requirements (dockerfile) to run properly
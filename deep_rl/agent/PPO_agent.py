#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
from pynput.keyboard import Key, Listener

from ..network import *
from ..component import *
from .BaseAgent import *


class PPOAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.opt = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length // 2):
            prediction_both = (self.network(states[0:1]), self.network(states[1:2]))
            next_states_both, rewards_both, terminals_both, info_both = self.task.step(
                np.vstack([
                    to_np(prediction_both[0]['a']),
                    to_np(prediction_both[1]['a'])]))

            for i in range(2):
                rewards, terminals, info = rewards_both[i:i+1], np.array(terminals_both[i:i+1]), info_both[i:i+1]
                prediction = prediction_both[i]

                self.record_online_return(info)
                rewards = config.reward_normalizer(rewards)
                # next_states = config.state_normalizer(next_states)
                storage.add(prediction)
                storage.add({'r': tensor(rewards).unsqueeze(-1),
                             'm': tensor(1 - terminals).unsqueeze(-1),
                             's': tensor(states[i:i+1])})

            next_states_both = config.state_normalizer(next_states_both)
            states = next_states_both
            self.total_steps += 2

        self.states = states
        # only train with single agent, but that doesnt matter, as trajectories are collected for both and they
        # share the network
        prediction = self.network(states[0:1])
        storage.add(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((1, 1)))
        returns = prediction['v'].detach()
        for i in reversed(range(config.rollout_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        states, actions, log_probs_old, returns, advantages = storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv'])
        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()

        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(states.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                prediction = self.network(sampled_states, sampled_actions)
                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['ent'].mean()

                value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()

                self.opt.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), config.gradient_clip)
                self.opt.step()

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)

        predictions = [self.network(state[i:i+1]) for i in range(2)]

        self.config.state_normalizer.unset_read_only()
        return [to_np(prediction['a']) for prediction in predictions]

    def eval_episode(self, player_agent=False):
        if player_agent:
            return self._eval_episode_player()
        else:
            return self._eval_episode_player()

    def _eval_episode(self):
        env = self.task
        states = env.reset()

        while True:
            actions = self.eval_step(states)
            states, rewards, dones, infos = env.step(np.vstack(actions))

            # Take maximum of both agents
            ret = [info['episodic_return'] for info in infos]
            if any(ret):
                break
        return max(ret)

    def _eval_episode_player(self):
        env = self.task
        states = env.reset()

        keyhandler = KeyHandler()
        listener = Listener(on_press=keyhandler.on_press, on_release=keyhandler.on_release)
        listener.start()

        while True:
            actions = self.eval_step(states)

            actions[0] = keyhandler.player_action

            states, rewards, dones, infos = env.step(np.vstack(actions))

            # Take maximum of both agents
            ret = [info['episodic_return'] for info in infos]
            if any(ret):
                break
        listener.stop()
        return max(ret)


class KeyHandler:

    def __init__(self):
        self.player_action = np.array((0., 0.))

    def on_press(self, key):
        # global player_action
        if key == Key.up:
            self.player_action = np.array((0., 10.))
        elif key == Key.down:
            self.player_action = np.array((0, -10.))
        elif key == Key.left:
            self.player_action = np.array((0.5, 0.))
        elif key == Key.right:
            self.player_action = np.array((-0.5, 0.))
        else:
            self.player_action = np.array((0., 0.))

    def on_release(self, key):
        if key in (Key.up, Key.down, Key.left, Key.right):
            self.player_action = np.array((0., 0.))

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage, ReplayBuffer
import torch
import torch.nn as nn
import torch.nn.functional as F

class AMPPPO(PPO):
    def __init__(self,
                 actor_critic,
                 discriminator,
                 amp_data,
                 amp_normalizer=None,
                 amp_replay_buffer_size=100000,
                 min_std=None,
                 **kwargs):
        
        super().__init__(actor_critic, **kwargs)

        self.min_std = min_std
        self.amp_normalizer = amp_normalizer

        # 初始化 AMP 相关组件
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.amp_transition = ReplayBuffer.Transition()
        self.amp_storage = ReplayBuffer(
            discriminator.input_dim // 2, amp_replay_buffer_size, self.device)
        self.amp_data = amp_data

        # 添加判别器的优化器参数
        self.optimizer.add_param_group({
            'params': self.discriminator.trunk.parameters(),
            'weight_decay': 1e-4,
            'name': 'amp_trunk'
        })
        self.optimizer.add_param_group({
            'params': self.discriminator.amp_linear.parameters(),
            'weight_decay': 1e-2,
            'name': 'amp_head'
        })

    def act(self, obs, critic_obs, amp_obs):

        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        
        # 计算动作和值
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()

        # 存储观测
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.amp_transition.amp_observations = amp_obs  # 存储 AMP 的观测数据

        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, amp_obs):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # 如果有超时信息，进行奖励修正
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # 将 AMP 数据存储到 AMP Replay Buffer
        self.amp_storage.insert(self.amp_transition.amp_observations, amp_obs)

        # 存储 PPO 的过渡
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.amp_transition.clear()
        self.actor_critic.reset(dones)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches)
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches)

        for sample, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):
            # PPO 的损失计算
            obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch = sample

            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            entropy_batch = self.actor_critic.entropy
            sigma_batch = self.actor_critic.action_std
            mu_batch = self.actor_critic.action_mean
            
            # KL
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
            surrogate_loss = torch.max(-advantages_batch * ratio,
                                   -advantages_batch * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)).mean()
            
            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # AMP 损失计算
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert

            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                    policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                    expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                    expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)

            policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
            expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))

            expert_loss = F.mse_loss(expert_d, torch.ones_like(expert_d))
            policy_loss = F.mse_loss(policy_d, -1 * torch.ones_like(policy_d))
            amp_loss = 0.5 * (expert_loss + policy_loss)

            grad_pen_loss = self.discriminator.compute_grad_pen(expert_state, expert_next_state, lambda_=10)

            # 总损失
            loss = (
                surrogate_loss +
                self.value_loss_coef * value_loss -
                self.entropy_coef * entropy_batch.mean() +
                amp_loss + grad_pen_loss
            )

            if not self.actor_critic.fixed_std and self.min_std is not None:
                self.actor_critic.std.data = self.actor_critic.std.data.clamp(min=self.min_std)
            
            if self.amp_normalizer is not None:
                self.amp_normalizer.update(policy_state.cpu().numpy())
                self.amp_normalizer.update(expert_state.cpu().numpy())
            
            # 梯度更新
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_loss.item()
            mean_expert_pred += expert_loss.item()
        
        self.storage.clear()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        return mean_value_loss / num_updates, mean_surrogate_loss / num_updates, mean_amp_loss / num_updates, mean_grad_pen_loss / num_updates, mean_policy_pred / num_updates, \
                mean_expert_pred / num_updates

# al = AMPPPO(ActorCritic(60, 64, 12))
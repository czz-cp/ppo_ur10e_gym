"""
PPO (Proximal Policy Optimization) Implementation - Isaac Gym版本

针对Isaac Gym优化的PPO实现，支持大规模并行训练
集成RL-PID混合控制和奖励归一化功能
"""

# IMPORTANT: Isaac Gym must be imported before PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import gym
from typing import Dict, Any, List, Tuple, Optional
import time
import os

from ur10e_env_isaac import UR10ePPOEnvIsaac
from utils import (ValueNormalization, GAE, assert_same_device, check_tensor_devices,
                   get_tensor_device, ensure_device, get_forced_device)


class ActorNetwork(nn.Module):
    """Actor网络 - PPO策略函数"""

    def __init__(self, state_dim: int = 18, action_dim: int = 6, hidden_dim: int = 64):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.max_torques = np.array([264.0, 264.0, 120.0, 43.2, 43.2, 43.2], dtype=np.float32)
        self.action_space_high = self.max_torques
        self.action_space_low = -self.max_torques

        self.register_buffer("max_torques_tensor", torch.tensor(self.max_torques, dtype=torch.float32)) 

        # 策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim * 2)  # 均值和标准差
        )

        # 初始化权重
        self._init_actor_weights()

    def _init_actor_weights(self):
        """标准的 Orthogonal 初始化 + 零偏置，不再给输出层加 1.05 偏置"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)


    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            state: [batch_size, state_dim] 状态张量

        Returns:
            mean: [batch_size, action_dim] 动作均值
            std: [batch_size, action_dim] 动作标准差
        """
        #self.max_torques_tensor = torch.tensor(self.max_torques, device=self.device, dtype=torch.float32)

        policy_output = self.policy_net(state)
        mean, log_std = policy_output.chunk(2, dim=-1)

        log_std = torch.clamp(log_std, -2.0, 2.0)
        #std = F.softplus(log_std)   # 确保标准差为正
        #softplus(x) = log(1 + exp(x))
        std = torch.exp(log_std) 
        return mean, std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样动作

        Args:
            state: [batch_size, state_dim] 状态张量

        Returns:
            action: [batch_size, action_dim] 采样动作
            log_prob: [batch_size] 对数概率
        """
        mean, std = self.forward(state)
        dist = Normal(mean, std)

        #action = dist.sample() 
        raw = dist.rsample()  # 用 rsample 方便以后做 reparameterization
        log_prob = dist.log_prob(raw).sum(dim=-1)

        action = torch.tanh(raw)  # 将动作限制在[-1, 1]范围内
        action = action * self.max_torques_tensor

        #max_tau = 30.0PPOIsaac.collect_rollouts
        #action = torch.clamp(action, -max_tau, max_tau)

        return action, log_prob

class CriticNetwork(nn.Module):
    """Critic网络 - PPO价值函数"""

    def __init__(self, state_dim: int = 18, hidden_dim: int = 64):
        super().__init__()

        # 价值网络
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight.data, gain=np.sqrt(2))
            nn.init.constant_(module.bias.data, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            state: [batch_size, state_dim] 状态张量

        Returns:
            value: [batch_size, 1] 状态价值
        """
        return self.value_net(state)

class PPOIsaac:
    """
    PPO算法实现 - Isaac Gym版本

    专门针对Isaac Gym大规模并行训练优化
    集成价值归一化、GAE、梯度裁剪等技术
    """

    def __init__(self,
                 env: UR10ePPOEnvIsaac,
                 config: Dict[str, Any]):
        """
        初始化PPO训练器

        Args:
            env: Isaac Gym环境
            config: 训练配置
        """
        self.env = env
        self.config = config

        # 🎯 [SERVER FIX] 使用环境设备（参考isaac_gym_manipulator实现模式）
        self.device = env.device
        print(f"🔒 [ENV DEVICE] PPO使用环境设备: {self.device}")

        # 🚨 [SERVER SAFETY] 验证设备一致性
        forced_device = get_forced_device()
        if str(self.device) != str(forced_device):
            print(f"⚠️ [DEVICE MISMATCH] 环境设备{self.device} != 强制设备{forced_device}")
            print(f"   强制使用环境设备: {self.device}")
            # 在服务器环境中，强制环境设备为cuda:0
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        # 环境参数
        self.num_envs = env.get_num_envs()
        self.state_dim = env.get_num_obs()
        self.action_dim = env.get_num_actions()

        # 网络初始化
        self.actor = ActorNetwork(self.state_dim, self.action_dim).to(self.device)
        self.critic = CriticNetwork(self.state_dim).to(self.device)

        # 确保网络参数在正确设备上 (修复设备一致性)
        assert next(self.actor.parameters()).device == self.device, "Actor not on correct device"
        assert next(self.critic.parameters()).device == self.device, "Critic not on correct device"

        # 显式设置网络参数requires_grad=True
        for param in self.actor.parameters():
            param.requires_grad = True
        for param in self.critic.parameters():
            param.requires_grad = True

        # 验证梯度设置
        assert all(p.requires_grad for p in self.actor.parameters()), "Actor参数未设置requires_grad"
        assert all(p.requires_grad for p in self.critic.parameters()), "Critic参数未设置requires_grad"

        # 优化器
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=float(config['ppo']['lr_actor'])
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=float(config['ppo']['lr_critic'])
        )

        # 价值归一化器
        self.value_norm = ValueNormalization(
            beta=0.995,
            epsilon=1e-5,
            clip_range=10.0
        ).to(self.device)

        # GAE计算器
        self.gae = GAE(
            gamma=float(config['ppo']['gamma']),
            lam=float(config['ppo']['lam'])
        )

        # 训练参数 - 确保类型转换
        self.clip_eps = float(config['ppo']['clip_eps'])
        self.entropy_coef = float(config['ppo']['entropy_coef'])
        self.value_coef = float(config['ppo']['value_coef'])
        self.max_grad_norm = float(config['ppo']['max_grad_norm'])

        # 缓冲区参数 - 确保整数类型
        self.rollout_length = int(config['train']['rollout_length'])
        self.batch_size = int(config['train']['batch_size'])
        self.num_updates = int(config['train']['num_updates'])

        # 统计信息
        self.episode_count = 0
        self.total_steps = 0
        self.best_performance = -float('inf')

        print(f"🤖 Isaac Gym PPO训练器初始化完成")
        print(f"   并行环境数: {self.num_envs}")
        print(f"   状态维度: {self.state_dim}")
        print(f"   动作维度: {self.action_dim}")
        print(f"   设备: {self.device}")

        # 梯度计算测试
        if not self._test_gradient_flow():
            print("❌ 梯度计算测试失败")
            raise RuntimeError("梯度计算测试失败，请检查网络实现")

    def _test_gradient_flow(self):
        """测试梯度计算是否正常工作"""
        try:
            # 创建测试数据
            test_states = torch.randn(2, self.state_dim, device=self.device, requires_grad=True)
            test_actions = torch.randn(2, self.action_dim, device=self.device)
            test_old_log_probs = torch.randn(2, device=self.device)
            test_advantages = torch.randn(2, device=self.device)

            # 测试actor梯度
            new_means, new_stds = self.actor(test_states)
            dist = Normal(new_means, new_stds)
            new_log_probs = dist.log_prob(test_actions).sum(dim=-1)
            ratio = torch.exp(new_log_probs - test_old_log_probs)
            actor_loss = -(ratio * test_advantages).mean()

            assert actor_loss.requires_grad, "Actor损失没有梯度"
            actor_loss.backward()

            # 检查actor参数梯度
            actor_has_grad = any(p.grad is not None for p in self.actor.parameters())
            assert actor_has_grad, "Actor参数没有梯度"

            # 测试critic梯度
            test_values = self.critic(test_states).squeeze(-1)
            test_returns = torch.randn(2, device=self.device)
            critic_loss = F.mse_loss(test_values, test_returns)

            assert critic_loss.requires_grad, "Critic损失没有梯度"
            critic_loss.backward()

            # 检查critic参数梯度
            critic_has_grad = any(p.grad is not None for p in self.critic.parameters())
            assert critic_has_grad, "Critic参数没有梯度"

            # 清理梯度
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            print("✅ 梯度计算测试通过")
            return True

        except Exception as e:
            print(f"❌ 梯度计算测试失败: {e}")
            return False

    def collect_rollouts(self) -> Dict[str, torch.Tensor]:
        """
        收集经验回放数据

        Returns:
            rollouts: 收集的数据字典
        """
        # 重置环境
        states = self.env.reset()
        # 确保状态在正确的设备上
        states = ensure_device(states, self.device)

        # 初始化缓冲区
        rollouts = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
            'dones': [],
            'next_states': []
        }

        episode_rewards = np.zeros(self.num_envs)
        episode_lengths = np.zeros(self.num_envs)

        for step in range(self.rollout_length):
            # 记录当前状态
            rollouts['states'].append(states.clone())

            # 采样动作 (数据收集时使用no_grad，但状态需要梯度)
            states_for_sampling = states.detach().requires_grad_(True)
            with torch.no_grad():
                actions, log_probs = self.actor.sample(states_for_sampling)
                values = self.critic(states_for_sampling)

            # 调试信息 (每64步显示一次进度)
            if step % 64 == 0 and hasattr(self.env, 'episode_steps'):
                avg_episode_steps = self.env.episode_steps.mean().item()
                max_episode_steps = self.env.episode_steps.max().item()
                print(f"📈 Step {step:3d}: 平均episode步数: {avg_episode_steps:.1f}, 最大: {max_episode_steps}")

            # 执行动作
            next_states, rewards, dones, infos = self.env.step(actions)

            # 确保所有张量在正确设备上
            next_states = ensure_device(next_states, self.device)
            rewards = ensure_device(rewards, self.device)
            dones = ensure_device(dones, self.device)

            # 设备一致性检查 (修复设备不匹配问题)
            try:
                assert_same_device(states, actions, next_states, rewards, dones, device=self.device)
            except AssertionError as e:
                print(f"❌ 设备不匹配错误: {e}")
                print(f"   states: {states.device}")
                print(f"   actions: {actions.device}")
                print(f"   next_states: {next_states.device}")
                print(f"   rewards: {rewards.device}")
                print(f"   dones: {dones.device}")
                print(f"   期望设备: {self.device}")
                raise

            # 记录数据
            rollouts['actions'].append(actions.clone())
            rollouts['log_probs'].append(log_probs.clone())
            rollouts['values'].append(values.squeeze(-1).clone())
            rollouts['rewards'].append(rewards.clone())
            rollouts['dones'].append(dones.clone())
            rollouts['next_states'].append(next_states.clone())

            # 统计信息 (修复设备不匹配)
            rewards_device = rewards.device
            episode_rewards += rewards.detach().cpu().numpy()
            episode_lengths += 1

            # 设备一致性调试信息
            if rewards_device != self.device:
                print(f"⚠️ 设备不匹配: rewards在{rewards_device}, 期望在{self.device}")

            # 处理完成的回合
            for i in range(self.num_envs):
                if dones[i]:
                    self.episode_count += 1
                    self.total_steps += episode_lengths[i]

                    print(f"🎯 Episode完成! 环境{i}, 奖励: {episode_rewards[i]:.4f}, 步数: {episode_lengths[i]}")

                    # 更新最佳性能
                    if episode_rewards[i] > self.best_performance:
                        self.best_performance = episode_rewards[i]
                        print(f"🏆 新最佳性能! {self.best_performance:.4f}")

                    # 重置统计
                    episode_rewards[i] = 0
                    episode_lengths[i] = 0

            states = next_states

        # 转换为张量
        for key in rollouts:
            if key != 'next_states':
                rollouts[key] = torch.stack(rollouts[key], dim=0)  # [rollout_length, num_envs]

        return rollouts

    def update_policy(self, rollouts: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        更新策略网络

        Args:
            rollouts: 经验回放数据

        Returns:
            metrics: 训练指标
        """
        # 🎯 [SERVER FIX] 确保所有rollout数据在正确设备上
        for key, tensor in rollouts.items():
            if isinstance(tensor, torch.Tensor) and tensor.device != self.device:
                print(f"⚠️ [DEVICE FIX] {key}从{tensor.device}移动到{self.device}")
                rollouts[key] = tensor.to(self.device)

        # 准备数据
        states = rollouts['states'].view(-1, self.state_dim)  # [T*N, state_dim]
        actions = rollouts['actions'].view(-1, self.action_dim)  # [T*N, action_dim]
        old_log_probs = rollouts['log_probs'].view(-1)  # [T*N]

        # 计算价值和优势
        values = rollouts['values'].view(self.rollout_length, self.num_envs)  # [T, N]
        rewards = rollouts['rewards'].view(self.rollout_length, self.num_envs)  # [T, N]
        dones = rollouts['dones'].view(self.rollout_length, self.num_envs)  # [T, N]

        # 🔍 [CRITICAL CHECK] 验证设备一致性（预防第500步错误）
        try:
            assert_same_device(states, actions, old_log_probs, values, rewards, dones, device=self.device)
            print(f"✅ [DEVICE OK] 所有rollout数据在{self.device}上")
        except AssertionError as e:
            print(f"❌ [DEVICE ERROR] {e}")
            # 强制修复
            states = states.to(self.device)
            actions = actions.to(self.device)
            old_log_probs = old_log_probs.to(self.device)
            values = values.to(self.device)
            rewards = rewards.to(self.device)
            dones = dones.to(self.device)

        # 获取下一个状态的价值
        with torch.no_grad():
            last_next_state = rollouts['next_states'][-1].to(self.device)
            next_values = self.critic(last_next_state).squeeze(-1)  # [N]
            # 修复：为GAE函数创建正确形状的next_values [T, N]
            next_values_expanded = next_values.unsqueeze(0).expand(self.rollout_length, -1)  # [T, N]

        # 计算GAE优势和回报
        advantages, returns = self.gae(rewards, dones, values, next_values_expanded)

        # 展平
        advantages = advantages.view(-1).float()  # [T*N]
        returns = returns.view(-1).float()  # [T*N]

        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 更新价值归一化器
        self.value_norm.update(returns)

        # 多次更新
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0

        for _ in range(self.num_updates):
            # 随机采样批次
            indices = torch.randperm(states.shape[0])

            for start in range(0, states.shape[0], self.batch_size):
                end = min(start + self.batch_size, states.shape[0])
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # 计算新的动作概率 (需要梯度进行更新)
                #new_means, new_stds = self.actor(batch_states)
                #ist = Normal(new_means, new_stds)
                #batch_new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)

                # 计算比率
                #ratio = torch.exp(batch_new_log_probs - batch_old_log_probs)

                # 1. 得到高斯参数（raw 空间）
                new_means, new_stds = self.actor(batch_states)   # [B, act_dim]
                dist = Normal(new_means, new_stds)

                # 2. 把扭矩动作还原回 raw 空间
                #   2.1 先除以 max_torques 得到 squashed ∈ [-1,1]
                max_torques = self.actor.max_torques_tensor      # [6]
                squashed = batch_actions / max_torques           # [B,6]，自动 broadcast

                #   2.2 数值安全一点，夹紧在 (-1+eps, 1-eps)
                eps = 1e-6
                squashed = torch.clamp(squashed, -1.0 + eps, 1.0 - eps)

                #   2.3 反 tanh：raw = atanh(squashed)
                raw = 0.5 * (torch.log1p(squashed) - torch.log1p(-squashed))
                # 也可以用 torch.atanh(squashed)（如果你的 torch 版本支持）

                # 3. 在 raw 空间下算 log_prob
                batch_new_log_probs = dist.log_prob(raw).sum(dim=-1)

                # 4. 一切照旧
                ratio = torch.exp(batch_new_log_probs - batch_old_log_probs)


                # Actor损失 (PPO裁剪)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # 熵正则化
                entropy = dist.entropy().sum(dim=-1).mean()

                # Critic损失
                batch_values = self.critic(batch_states).squeeze(-1).float()
                normalized_returns = self.value_norm.normalize(batch_returns).float()
                critic_loss = F.mse_loss(batch_values, normalized_returns.detach())

                # 总损失
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                # 梯度计算验证 (调试模式)
                if hasattr(self, '_debug_mode') and self._debug_mode:
                    # 检查损失是否可以计算梯度
                    if not loss.requires_grad:
                        print(f"❌ 损失没有requires_grad: {loss.requires_grad}")

                    # 检查actor_loss梯度
                    if not actor_loss.requires_grad:
                        print(f"❌ actor_loss没有requires_grad: {actor_loss.requires_grad}")

                    # 检查网络参数梯度
                    for name, param in self.actor.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any():
                                print(f"❌ Actor参数{name}梯度包含NaN")
                        elif not param.requires_grad:
                            print(f"❌ Actor参数{name}不需要梯度")

                # 更新Actor
                self.actor_optimizer.zero_grad()

                try:
                    actor_loss.backward(retain_graph=True)

                    # 检查actor梯度
                    if hasattr(self, '_debug_mode') and self._debug_mode:
                        actor_grad_norm = sum(p.grad.norm().item() for p in self.actor.parameters() if p.grad is not None)
                        print(f"📊 Actor梯度范数: {actor_grad_norm:.6f}")

                except Exception as e:
                    print(f"❌ Actor梯度计算失败: {e}")
                    raise

                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # 更新Critic
                self.critic_optimizer.zero_grad()

                try:
                    critic_loss.backward()

                    # 检查critic梯度
                    if hasattr(self, '_debug_mode') and self._debug_mode:
                        critic_grad_norm = sum(p.grad.norm().item() for p in self.critic.parameters() if p.grad is not None)
                        print(f"📊 Critic梯度范数: {critic_grad_norm:.6f}")

                except Exception as e:
                    print(f"❌ Critic梯度计算失败: {e}")
                    raise

                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # 累计统计
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()

        # 计算平均值
        num_updates = self.num_updates * (states.shape[0] // self.batch_size)

        metrics = {
            'actor_loss': total_actor_loss / num_updates,
            'critic_loss': total_critic_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'mean_advantage': advantages.mean().item(),
            'mean_return': returns.mean().item(),
            'policy_std': new_stds.mean().item()
        }

        return metrics

    def train(self, num_episodes: int = 1000, save_dir: str = "./checkpoints"):
        """
        开始训练

        Args:
            num_episodes: 训练回合数
            save_dir: 模型保存目录
        """
        import signal

        # 设置退出信号处理
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                print(f"\n🛑 接收到退出信号 {signum}，正在优雅退出...")
                shutdown_requested = True
            else:
                print(f"⚠️ 强制退出信号 {signum}，立即退出...")
                import sys
                sys.exit(1)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        print(f"🚀 开始训练，目标回合数: {num_episodes}")
        print(f"   按 Ctrl+C 可安全退出训练")
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 🔹 初始化 loss 日志文件
        loss_log_path = os.path.join(save_dir, "loss_curve.csv")
        with open(loss_log_path, "w") as f:
            f.write("log_step,episode,actor_loss,critic_loss,entropy,mean_return\n")

        # 训练统计
        training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_losses': [],
            'critic_losses': [],
            'success_rates': []
        }

        start_time = time.time()

        for episode in range(num_episodes):
            # 检查退出信号
            if shutdown_requested:
                print(f"🛑 收到退出信号，正在安全停止训练...")
                print(f"   已完成 {episode} 个回合")
                break

            # 收集经验
            rollouts = self.collect_rollouts()

            # 更新策略
            metrics = self.update_policy(rollouts)

            # 记录统计信息
            if episode % 10 == 0:
                avg_reward = self.best_performance
                current_time = time.time() - start_time

                print(f"📊 Episode {episode:5d} | "
                      f"Best Reward: {avg_reward:8.4f} | "
                      f"Actor Loss: {metrics['actor_loss']:8.4f} | "
                      f"Critic Loss: {metrics['critic_loss']:8.4f} | "
                      f"Policy Std: {metrics['policy_std']:6.4f} | "
                      f"Time: {current_time/60:6.2f}min")
                
                 # 🔹 追加一行到 CSV（每 10 个 episode 记一次）
                log_step = len(training_stats['actor_losses'])
                with open(loss_log_path, "a") as f:
                    f.write(
                        f"{log_step},{episode},"
                        f"{metrics['actor_loss']:.6f},{metrics['critic_loss']:.6f},"
                        f"{metrics['entropy']:.6f},{metrics['mean_return']:.6f}\n"
                    )

                # 记录训练统计
                training_stats['actor_losses'].append(metrics['actor_loss'])
                training_stats['critic_losses'].append(metrics['critic_loss'])

            # 保存模型
            if episode % 100 == 0 and episode > 0:
                self.save_model(save_dir, episode)

        # 训练完成
        total_time = time.time() - start_time
        print(f"🎉 训练完成！总用时: {total_time/60:.2f}分钟")
        print(f"   总回合数: {self.episode_count}")
        print(f"   总步数: {self.total_steps}")
        print(f"   最佳性能: {self.best_performance:.4f}")

        # 保存最终模型
        self.save_model(save_dir, 'final')

        return training_stats

    def save_model(self, save_dir: str, episode: int):
        """
        保存模型

        Args:
            save_dir: 保存目录
            episode: 回合数
        """
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            'episode': episode,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'value_norm_state_dict': self.value_norm.state_dict(),
            'best_performance': self.best_performance,
            'episode_count': self.episode_count,
            'total_steps': self.total_steps
        }

        torch.save(checkpoint, os.path.join(save_dir, f'ppo_checkpoint_{episode}.pth'))

        # 也保存为最新版本
        torch.save(checkpoint, os.path.join(save_dir, 'latest.pth'))

    def load_model(self, checkpoint_path: str):
        """
        加载模型

        Args:
            checkpoint_path: 模型路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.value_norm.load_state_dict(checkpoint['value_norm_state_dict'])

        self.best_performance = checkpoint['best_performance']
        self.episode_count = checkpoint['episode_count']
        self.total_steps = checkpoint['total_steps']

        print(f"✅ 模型加载完成，回合数: {checkpoint['episode']}")

def load_config_isaac(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    加载Isaac Gym版本配置

    Args:
        config_path: 配置文件路径

    Returns:
        config: 配置字典
    """
    import yaml
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠️ 配置文件 {config_path} 未找到，使用默认配置")
        config = get_default_config_isaac()

    return config

def get_default_config_isaac() -> Dict[str, Any]:
    """获取默认Isaac Gym配置"""
    return {
        'env': {
            'num_envs': 512,
            'max_steps': 1000,
            'dt': 0.01
        },
        'ppo': {
            'lr_actor': 3e-4,
            'lr_critic': 1e-3,
            'clip_eps': 0.2,
            'gamma': 0.99,
            'lam': 0.95,
            'entropy_coef': 0.01,
            'value_coef': 0.5,
            'max_grad_norm': 0.5
        },
        'train': {
            'rollout_length': 2048,
            'batch_size': 512,
            'num_updates': 10,
            'num_episodes': 1000
        }
    }

if __name__ == "__main__":
    # 加载配置
    config = load_config_isaac()

    # 创建环境
    env = UR10ePPOEnvIsaac(
        config_path="config.yaml",
        num_envs=config['env']['num_envs']
    )

    # 创建PPO训练器
    ppo = PPOIsaac(env, config)

    # 开始训练
    training_stats = ppo.train(
        num_episodes=config['train']['num_episodes'],
        save_dir="./checkpoints_isaac"
    )

    # 关闭环境
    env.close()
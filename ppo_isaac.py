"""
PPO (Proximal Policy Optimization) Implementation - Isaac Gym版本

针对Isaac Gym优化的PPO实现，支持大规模并行训练
"""

# IMPORTANT: Isaac Gym must be imported before PyTorch
import numpy as np
import gym
from typing import Dict, Any, List, Tuple, Optional
import time
import os

from ur10e_env_isaac import UR10ePPOEnvIsaac
from ur10e_trajectory_env_isaac import UR10eTrajectoryEnvIsaac
from utils import (ValueNormalization, GAE, assert_same_device, check_tensor_devices,
                   get_tensor_device, ensure_device, get_forced_device)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


class ActorNetwork(nn.Module):
    """Actor网络 - 论文风格 3×256 tanh MLP，高斯策略 + tanh-squash + 动作集成"""

    def __init__(self, state_dim: int = 22, action_dim: int = 6, hidden_dim: int = 256):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # 归一化动作空间 [-1, 1]^action_dim
        self.register_buffer(
            "action_limits_tensor",
            torch.ones(action_dim, dtype=torch.float32)
        )

        # 特征提取 MLP：3 层 × 256, tanh 激活（对齐论文）
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # 独立的 mean / log_std heads
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        self._init_actor_weights()

    def _init_actor_weights(self):
        """Orthogonal 初始化 + 小输出，适配 tanh"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # tanh 通常用 gain=1.0 就够了
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            state: [batch_size, state_dim]

        Returns:
            mean: [batch_size, action_dim]
            std:  [batch_size, action_dim]
        """

        x = self.feature_net(state)              # [B, 256]


        mean = self.mean_head(x)                 # [B, act_dim]
        log_std = self.log_std_head(x)           # [B, act_dim]

        # 1) 先把非有限值处理掉（不切整图）
        mean = torch.where(torch.isfinite(mean), mean, torch.zeros_like(mean))
        log_std = torch.where(torch.isfinite(log_std), log_std, torch.full_like(log_std, -2.0))

        # 2) 限制 mean 幅度：避免 log_prob 里出现 inf/-inf（超重要）
        #   用“软限制”比 hard clamp 更平滑，梯度更健康
        mean = 5.0 * torch.tanh(mean / 5.0)   # raw-space mean ∈ [-5, 5]

        # 防止 std 崩：限制 log_std 范围
        log_std = torch.clamp(log_std, -4.0, 1.0)
        # 和你现在一致，用 softplus 把它变成正数
        #std = F.softplus(log_std)
        std = torch.exp(log_std)              # 比 softplus 更直观
        std = torch.clamp(std, 1e-3, 2.0)

        return mean, std

    def sample(self, state: torch.Tensor, use_delta_std: bool = True, delta_std: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        🎯 Clip-only 采样（论文版本）

        核心原则：log_prob 必须对应同一个"未截断的高斯变量"
        clip 只是给 env 用的安全执行，不要把 clip 之后的值当作随机变量去算 log_prob

        Returns:
            action:   [-1, 1] 内的归一化动作（给环境执行用）
            log_prob: raw变量的log_prob（用于PPO计算）
            raw:      未截断的原始动作（用于存储和重构）
        """
        mean, std = self.forward(state)

        # 🎯 使用固定δ_std（如果启用）
        if use_delta_std:
            std = torch.full_like(std, delta_std)

        dist = Normal(mean, std)
        raw = dist.rsample()              # 用 rsample 方便反传（PPO会用到）
        action = torch.clamp(raw, -1.0, 1.0)  # 给 env 执行用
        log_prob = dist.log_prob(raw).sum(dim=-1)  # 🎯 重要：对 raw 算 log_prob

        return action, log_prob, raw
    def atanh(self, x: torch.Tensor) -> torch.Tensor:
        """数值安全的atanh实现"""
        x = torch.clamp(x, -0.999, 0.999)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def squashed_log_prob(self, dist: torch.distributions.Normal, actions: torch.Tensor) -> torch.Tensor:
        """
        计算squashed Gaussian的log_prob

        Args:
            dist: torch.distributions.Normal, raw空间的高斯分布
            actions: torch.Tensor, [-1,1]范围的squashed动作

        Returns:
            log_prob: torch.Tensor, 考虑tanh变换的log概率
        """
        # 将squashed动作还原到raw空间
        raw = self.atanh(actions)
        # 计算raw空间的log_prob
        logp = dist.log_prob(raw).sum(dim=-1)
        # 减去tanh变换的Jacobian对数行列式: log|det(∂tanh/∂raw)|
        # ∂tanh/∂raw = 1 - tanh²(raw) = 1 - actions²
        # 因为1 - actions² > 0（|actions| < 1），所以可以直接用log
        jacobian_log = torch.log(1 - actions * actions + 1e-6).sum(dim=-1)
        logp -= jacobian_log
        return logp

    def get_dist(self, states: torch.Tensor, fixed_std: float = None) -> torch.distributions.Normal:
        """
        获取策略分布，支持固定标准差

        Args:
            states: [batch_size, state_dim]
            fixed_std: float, 如果提供则使用固定std，否则使用网络std

        Returns:
            dist: torch.distributions.Normal
        """
        mean, std = self.forward(states)
        if fixed_std is not None:
            std = torch.ones_like(std) * fixed_std
        return torch.distributions.Normal(mean, std)

    def compute_aew_ensemble_size(self, current_episode: int, max_episodes: int,
                                alpha: float = 5.0, beta: float = 8.0, lambda_max: float = None) -> int:
        """
        计算AEW（Weibull Action Ensembles）的采样次数

        根据论文：i ~ clip(Weibull(k, λ), 1, λ)
        其中 k = 1 + α * episode / episode_max, λ = 1 + β * episode / episode_max

        Args:
            current_episode: 当前训练episode
            max_episodes: 最大训练episode数
            alpha: Weibull形状参数增长系数
            beta: Weibull尺度参数增长系数

        Returns:
            ensemble_size: 采样次数 i
        """
        progress = current_episode / max(max_episodes, 1)  # 防止除零

        # 计算Weibull分布参数
        k = 1.0 + alpha * progress  # 形状参数
        lam = 1.0 + beta * progress  # 尺度参数

        # 从Weibull分布采样
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        # Weibull采样：u ~ Uniform(0,1), x = λ * (-log(u))^(1/k)
        u = torch.rand(1, device=device)
        weibull_sample = lam * (-torch.log(u)).pow(1.0 / k)

        # 使用你建议的方法：max(1, round(i))，然后clamp(1, max(lam, lambda_max))
        max_ensemble = lambda_max if lambda_max is not None else lam
        ensemble_size = int(torch.clamp(weibull_sample.round(), 1, max_ensemble).item())

        return ensemble_size
    
    def sample_clip(self, state, delta_std: float):
        mean, _ = self.forward(state)
        std = torch.full_like(mean, delta_std)
        dist = Normal(mean, std)

        raw = dist.sample()                       # raw action
        exec_action = torch.clamp(raw, -1.0, 1.0) # clip only for env
        log_prob = dist.log_prob(raw).sum(-1)     # IMPORTANT: prob of raw
        return exec_action, log_prob, raw


    def sample_with_ensemble_clip(self, state, ensemble_size: int, delta_std: float):
        mean, _ = self.forward(state)
        std = torch.full_like(mean, delta_std)

        B, A = mean.shape
        i = ensemble_size
        raw_samples = Normal(
            mean.unsqueeze(1).expand(B, i, A),
            std.unsqueeze(1).expand(B, i, A)
        ).sample()                                # [B,i,A]

        raw_mean = raw_samples.mean(dim=1)        # [B,A]
        exec_action = torch.clamp(raw_mean, -1.0, 1.0)

        eff_std = std / torch.sqrt(torch.tensor(float(i), device=std.device))
        eff_dist = Normal(mean, eff_std)
        log_prob = eff_dist.log_prob(raw_mean).sum(-1)

        return exec_action, log_prob, raw_mean


    def log_prob_ensemble_rawmean(self, state, raw_mean, ensemble_size: int, delta_std: float):
        mean, _ = self.forward(state)
        std = torch.full_like(mean, delta_std)
        eff_std = std / torch.sqrt(torch.tensor(float(ensemble_size), device=std.device))
        eff_dist = Normal(mean, eff_std)
        return eff_dist.log_prob(raw_mean).sum(-1)



class CriticNetwork(nn.Module):
    """Critic网络 - 论文风格 3×256 tanh MLP"""

    def __init__(self, state_dim: int = 22, hidden_dim: int = 256):
        super().__init__()

        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight.data, gain=1.0)
            nn.init.constant_(module.bias.data, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim]

        Returns:
            value: [batch_size, 1]
        """
        return self.value_net(state)


class PPOIsaac:
    """
    PPO算法实现 - Isaac Gym版本

    专门针对Isaac Gym大规模并行训练优化
    集成价值归一化、GAE、梯度裁剪等技术
    """

    def __init__(self,
                 env: UR10eTrajectoryEnvIsaac,
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
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(),
            lr=float(config['ppo']['lr_actor'])
        )
        self.critic_optimizer = optim.AdamW(
            self.critic.parameters(),
            lr=float(config['ppo']['lr_critic'])
        )

        # 价值归一化器
        self.value_norm = ValueNormalization(
            beta=0.995,
            epsilon=1e-5,
            clip_range=10.0
        ).to(self.device)

        # 🎯 动作集成（Action Ensembles, AE）配置
        ae_config = config.get('ae', {})
        self.ae_enabled = ae_config.get('enabled', False)
        self.ae_alpha = float(ae_config.get('alpha', 5.0))
        self.ae_beta = float(ae_config.get('beta', 8.0))
        self.ae_lambda_max = int(ae_config.get('lambda_max', 10))  # 最大集成大小
        self.ae_delta_std = float(ae_config.get('delta_std', 0.1))
        self.current_ensemble_size = 1  # 默认采样次数

        # 🎯 策略反馈（Policy Feedback, PF）配置
        pf_config = config.get('pf', {})
        self.pf_enabled = pf_config.get('enabled', False)
        self.pf_eta_min = float(pf_config.get('eta_min', 0.6))
        self.pf_eta_max = float(pf_config.get('eta_max', 0.99))

        # GAE计算器（支持策略反馈）- 必须在pf_config定义之后
        self.gae = GAE(
            gamma=float(config['ppo']['gamma']),
            lam=float(config['ppo']['lam']),
            device=self.device,
            use_adaptive_gamma=self.pf_enabled,
            eta_min=self.pf_eta_min,
            eta_max=self.pf_eta_max
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
        self.num_episodes = int(config['train']['num_episodes'])

        # 统计信息
        self.episode_count = 0
        self.total_steps = 0
        self.best_performance = -float('inf')

        print(f"🤖 Isaac Gym PPO训练器初始化完成")
        print(f"   并行环境数: {self.num_envs}")
        print(f"   状态维度: {self.state_dim}")
        print(f"   动作维度: {self.action_dim}")
        print(f"   设备: {self.device}")

        # 🎯 显示AE���PF状态
        print(f"   🎯 动作集成(AE): {'启用' if self.ae_enabled else '禁用'}")
        if self.ae_enabled:
            print(f"      Alpha: {self.ae_alpha}, Beta: {self.ae_beta}, Lambda_max: {self.ae_lambda_max}, Delta_std: {self.ae_delta_std}")
        print(f"   🎯 策略反馈(PF): {'启用' if self.pf_enabled else '禁用'}")
        if self.pf_enabled:
            print(f"      Eta范围: [{self.pf_eta_min}, {self.pf_eta_max}]")

        # 梯度计算测试
        if not self._test_gradient_flow():
            print("❌ 梯度计算测试失败")
            raise RuntimeError("梯度计算测试失败，请检查网络实现")

    def update_ensemble_size(self):
        """根据当前训练进度更新AE采样次数"""
        if self.ae_enabled:
            self.current_ensemble_size = self.actor.compute_aew_ensemble_size(
                current_episode=self.episode_count,
                max_episodes=self.num_episodes,
                alpha=self.ae_alpha,
                beta=self.ae_beta,
                lambda_max=self.ae_lambda_max
            )

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
        reset_result = self.env.reset()
        # Handle both single obs and (obs, info) return formats
        if isinstance(reset_result, tuple):
            states, info = reset_result
            # Store info for potential debugging (suppress unused warning)
            _ = info
        else:
            states = reset_result

        # 确保状态在正确的设备上
        states = ensure_device(states, self.device)

        # 初始化缓冲区
        rollouts = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'raw_means': [],  # 🎯 配套要求：存储pre-clip的raw_means
            'values': [],
            'rewards': [],
            'dones': [],
            'next_states': []
        }

        episode_rewards = np.zeros(self.num_envs)
        episode_lengths = np.zeros(self.num_envs)

        # 🎯 A) 修复：一个rollout只用一个固定的ensemble size
        if self.ae_enabled:
            self.update_ensemble_size()
            rollout_ensemble_size = int(self.current_ensemble_size)
        else:
            rollout_ensemble_size = 1

        for step in range(self.rollout_length):
            # 确保states是2D张量 [num_envs, state_dim]
            if states.ndim == 1:
                states = states.unsqueeze(0)  # [state_dim] -> [1, state_dim]
            # 记录当前状态
            rollouts['states'].append(states.clone())

            # 采样动作 (数据收集时使用no_grad，但状态需要梯度)
            states_for_sampling = states.detach()
            with torch.no_grad():
                # 🎯 使用完全贴论文的clip-only方法
                if self.ae_enabled:
                    # 🎯 使用论文版AE：先平均raw，再clip；log_prob用"均值分布"
                    actions, log_probs, raw_means = self.actor.sample_with_ensemble_clip(
                        states_for_sampling,
                        ensemble_size=rollout_ensemble_size,
                        delta_std=self.ae_delta_std
                    )
                else:
                    # 🎯 标准PPO采样 - 完全clip-only版本
                    actions, log_probs, raw_actions = self.actor.sample_clip(
                        states_for_sampling,
                        delta_std=self.ae_delta_std if self.ae_enabled else self.ae_delta_std
                    )
                    raw_means = raw_actions  # 标准模式下，raw_means就是raw_actions

                values = self.critic(states_for_sampling)

            # 调试信息 (每64步显示一次进度)
            if step % 50 == 0 and hasattr(self.env, 'episode_steps'):
                avg_episode_steps = self.env.episode_steps.mean().item()
                max_episode_steps = self.env.episode_steps.max().item()
                print(f"📈 Step {step:3d}: 平均episode步数: {avg_episode_steps:.1f}, 最大: {max_episode_steps}")

            # 执行动作 (Gymnasium格式返回5个值)
            step_result = self.env.step(actions)
            if len(step_result) == 5:
                # Gymnasium格式: (obs, reward, terminated, truncated, info)
                next_states, rewards, terminated, truncated, infos = step_result
                dones = np.logical_or(terminated, truncated)  # 合并terminated和truncated
            elif len(step_result) == 4:
                # 旧格式: (obs, reward, done, info)
                next_states, rewards, dones, infos = step_result
            else:
                raise ValueError(f"环境step返回了{len(step_result)}个值，期望4或5个")

            # 确保所有张量在正确设备上
            next_states = ensure_device(next_states, self.device)
            rewards = ensure_device(rewards, self.device)
            dones = ensure_device(dones, self.device)

            # 处理numpy数组转换为张量
            if isinstance(dones, np.ndarray):
                dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
            elif not isinstance(dones, torch.Tensor):
                dones = torch.tensor([dones], dtype=torch.bool, device=self.device)

            # 确保dones是正确的形状
            if dones.dim() == 0:
                dones = dones.unsqueeze(0)  # [ ] -> [1]
            elif dones.dim() > 1:
                dones = dones.flatten()  # -> [num_envs]

            # 同样处理rewards
            if isinstance(rewards, (float, int, np.float32, np.float64, np.int32, np.int64)):
                rewards = torch.tensor([rewards], dtype=torch.float32, device=self.device)
            elif isinstance(rewards, np.ndarray):
                rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            elif not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

            # 确保rewards是正确的形状
            if rewards.dim() == 0:
                rewards = rewards.unsqueeze(0)  # [ ] -> [1]
            elif rewards.dim() > 1:
                rewards = rewards.flatten()  # -> [num_envs]

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
            rollouts['raw_means'].append(raw_means.clone())  # 🎯 存储pre-clip的raw_means
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
                if i < dones.shape[0] and dones[i]:
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

        # 🎯 添加ensemble size到rollouts中（用于update阶段一致性）
        # ✅ 确保存储为tensor，方便后续设备管理
        rollouts['ensemble_size'] = torch.tensor(rollout_ensemble_size, device=self.device)

        # 转换为张量
        for key in rollouts:
            if key not in ['next_states', 'ensemble_size']:
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
        raw_means = rollouts['raw_means'].view(-1, self.action_dim)  # 🎯 配套：取出存储的raw_means

        # 🔧 修复：计算价值和优势 - GAE需要原始尺度的values
        values_raw = rollouts['values'].view(self.rollout_length, self.num_envs)  # [T, N] - 原始网络输出
        # 🔧 保证value_norm存在时才denormalize（稳健性改进）
        if self.value_norm is not None:
            values = self.value_norm.denormalize(values_raw)  # 反归一化到原始奖励尺度用于GAE
        else:
            values = values_raw
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
            next_values_raw = self.critic(last_next_state).squeeze(-1)  # [N] - 原始网络输出
            # 🔧 保证value_norm存在时才denormalize（稳健性改进）
            if self.value_norm is not None:
                next_values = self.value_norm.denormalize(next_values_raw)  # 反归一化到原始奖励尺度
            else:
                next_values = next_values_raw
            # 修复：为GAE函数创建正确形状的next_values [T, N]
            next_values_expanded = next_values.unsqueeze(0).expand(self.rollout_length, -1)  # [T, N]

        # 计算GAE优势和回报（支持策略反馈）
        if self.pf_enabled:
            # ✅ 策略反馈关键修复：直接使用rollout时存��的log_probs计算action_probs
            # old_log_probs已经在rollout阶段使用与AE/atanh对齐的完整计算流程
            # old_log_probs: [T*N] -> [T, N]
            old_lp = rollouts['log_probs'].view(self.rollout_length, self.num_envs)

            # ✅ PF核心：直接使用rollout存储的log_probs，避免重新计算导致的不对齐
            # 论文定义: γ = clip(π(s,a), η, 1)，其中π(s,a)是rollout时的策略概率
            # 连续动作密度可能>1，因此先用clamp确保π(s,a) <= 1
            action_probs = torch.exp(torch.clamp(old_lp, max=0.0))

            advantages, returns = self.gae(rewards, dones, values, next_values_expanded, action_probs)
        else:
            # 标准GAE
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
        kl_early_stops = 0  # 记录KL early-stop次数
        stop_update = False  # ✅ KL early-stop标志（外层跳出用）

        for update_epoch in range(self.num_updates):
            # 随机采样批次
            indices = torch.randperm(states.shape[0], device=self.device)

            for start in range(0, states.shape[0], self.batch_size):
                end = min(start + self.batch_size, states.shape[0])
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_raw_means = raw_means[batch_indices]  # 🎯 配套：取出对应的raw_means

                # 🎯 B) 修复：使用rollout中存储的ensemble size确保一致性
                if self.ae_enabled:
                    # 🎯 关键修复：从rollout读取固定的ensemble size，避免"混i"问题
                    # ✅ 一次rollout固定一个i，防止step0用i=3、step1用i=9导致的错位
                    rollout_ensemble_size = int(rollouts['ensemble_size'].item())

                    batch_new_log_probs = self.actor.log_prob_ensemble_rawmean(
                        batch_states, batch_raw_means,
                        ensemble_size=rollout_ensemble_size,
                        delta_std=self.ae_delta_std
                    )
                    # AE模式：获取分布用于熵计算
                    mean, std = self.actor(batch_states)
                    if self.ae_delta_std is not None:
                        std = torch.ones_like(std) * self.ae_delta_std
                    dist = Normal(mean, std)
                else:
                    # 🎯 标准模式：使用raw_actions重构log_prob（clip-only版本）
                    # 从raw_means重构log_prob，确保与rollout阶段的计算完全对齐
                    mean, std = self.actor(batch_states)
                    # AE未启用时，std保持网络输出
                    dist = Normal(mean, std)
                    batch_new_log_probs = dist.log_prob(batch_raw_means).sum(dim=-1)  # 🎯 对raw_means计算

                # ✅ KL early-stop (PPO标准稳定器)
                # 计算近似KL散度：KL(pi_new || pi_old) ≈ E[log pi_old - log pi_new]
                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - batch_new_log_probs).mean().item()

                    # 🛑 KL early-stop: 如果KL超过阈值，提前结束本轮update
                if approx_kl > 0.05:  # 0.01~0.03都行，0.02是比较常用的值
                    kl_early_stops += 1
                    if kl_early_stops == 1:  # 只在第一次触发时打印
                        print(f"⚠️ KL early-stop triggered: KL={approx_kl:.4f} > 0.02, stopping update epoch {update_epoch}")
                    stop_update = True  # ✅ 设置标志，跳出两层循环
                    break  # 跳出内层minibatch循环

                # 计算比率
                #ratio = torch.exp(batch_new_log_probs - batch_old_log_probs)
                batch_old_log_probs = torch.nan_to_num(batch_old_log_probs, nan=0.0, posinf=0.0, neginf=0.0)
                log_ratio = batch_new_log_probs - batch_old_log_probs
                log_ratio = torch.nan_to_num(log_ratio, nan=0.0, posinf=20.0, neginf=-20.0)
                log_ratio = torch.clamp(log_ratio, -10.0, 10.0)
                ratio = torch.exp(log_ratio)

                # Actor损失 (PPO裁剪)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # 熵正则化
                entropy = dist.entropy().sum(dim=-1).mean()
                # 把熵打包进 actor_loss
                actor_total_loss = actor_loss - self.entropy_coef * entropy

                # Critic损失
                batch_values = self.critic(batch_states).squeeze(-1).float()
                normalized_returns = self.value_norm.normalize(batch_returns).float()
                critic_loss = F.mse_loss(batch_values, normalized_returns.detach())

                # 总损失（确保熵项符号正确）
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
                    actor_total_loss.backward()

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

            # ✅ 检查外层early-stop标志
            if stop_update:
                break

        # 计算平均值
        num_updates = self.num_updates * (states.shape[0] // self.batch_size)

        # 获取用于显示的策略std
        if self.ae_enabled:
            # AE模式：使用固定δ_std
            policy_std = self.ae_delta_std
        else:
            # 标准模式：使用最后一个批次的网络std
            #policy_std = new_stds.mean().item()
             policy_std = float(std.mean().item())

        metrics = {
            'actor_loss': total_actor_loss / num_updates,
            'critic_loss': total_critic_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'mean_advantage': advantages.mean().item(),
            'mean_return': returns.mean().item(),
            'policy_std': policy_std
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
            'entropy_coef': 0.001,  # 大幅减小 entropy 系数
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
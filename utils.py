"""
UR10e PPO 多目标最优轨迹规划工具函数

基于论文《基于深度强化学习的机械臂多目标最优轨迹规划》
提供训练、评估和可视化的辅助功能
"""

import numpy as np
import torch
import torch.nn as nn
import yaml
import os
import json
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import math

# 可选依赖 - 只有在需要时才导入
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def assert_same_device(*tensors, device=None):
    """
    确保所有tensor在同一设备上

    Args:
        *tensors: 要检查的tensors
        device: 期望的设备，如果为None则使用第一个tensor的设备
    """
    if len(tensors) == 0:
        return

    devices = [t.device for t in tensors if hasattr(t, 'device')]
    if not devices:
        return

    target_device = device if device else devices[0]

    for i, tensor in enumerate(tensors):
        if hasattr(tensor, 'device') and tensor.device != target_device:
            raise AssertionError(f"Tensor {i} on {tensor.device}, expected {target_device}")


class RewardNormalizer:
    """
    奖励归一化器 - 修复设备不匹配问题
    """
    def __init__(self, gamma=0.99, clip_range=5.0, normalize_method='running_stats', warmup_steps=100, device='cuda'):
        self.gamma = gamma
        self.clip_range = clip_range
        self.normalize_method = normalize_method
        self.warmup_steps = warmup_steps
        self.device = device

        # 运行统计量 - 在正确设备上
        self.register_buffer = lambda name, tensor: setattr(self, name, tensor)
        self.register_buffer('running_mean', torch.zeros(1, device=device))
        self.register_buffer('running_var', torch.ones(1, device=device))
        self.register_buffer('count', torch.zeros(1, device=device))

        self.reward_history = []

    def update(self, reward):
        """更新归一化统计量"""
        reward = float(reward)
        self.reward_history.append(reward)

        # 保持历史长度
        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-1000:]

        if len(self.reward_history) >= self.warmup_steps:
            # 更新运行统计
            recent_mean = np.mean(self.reward_history[-100:])
            recent_std = np.std(self.reward_history[-100:]) + 1e-8

            self.running_mean.copy_(torch.tensor([recent_mean], device=self.device))
            self.running_var.copy_(torch.tensor([recent_std**2], device=self.device))

    def normalize(self, reward):
        """归一化奖励"""
        if len(self.reward_history) < self.warmup_steps:
            return float(reward)

        mean = self.running_mean.item()
        std = torch.sqrt(self.running_var + 1e-8).item()
        normalized = (float(reward) - mean) / std
        return np.clip(normalized, -self.clip_range, self.clip_range)


class ValueNormalization(nn.Module):
    """
    Value Function Normalization

    用于稳定Critic训练的值函数归一化技术
    基于Isaac Gym实现，提供在线更新和归一化功能
    """
    def __init__(self, beta: float = 0.995, epsilon: float = 1e-8, clip_range: float = 10.0, device: str = None):
        super().__init__()
        self.beta = beta          # 指数移动平均系数
        self.epsilon = epsilon      # 数值稳定性参数
        self.clip_range = clip_range # 归一化值裁剪范围

        # 智能设备选择 - 修复设备不匹配问题
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        # 可学习的参数 - 使用智能设备选择
        self.register_buffer('mean', torch.zeros(1, device=self.device))
        self.register_buffer('var', torch.ones(1, device=self.device))
        self.register_buffer('count', torch.zeros(1, device=self.device))

    def update(self, values: torch.Tensor):
        """
        更新归一化统计量（在线EMA更新）

        Args:
            values: [batch_size, 1] 或 [batch_size] 价值函数值
        """
        values = values.view(-1, 1) if values.dim() == 1 else values

        batch_mean = values.mean()
        batch_var = values.var(unbiased=False)
        batch_count = values.numel()

        # 在线更新均值和方差
        self.mean = self.beta * self.mean + (1 - self.beta) * batch_mean
        self.var = self.beta * self.var + (1 - self.beta) * batch_var
        self.count += batch_count

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        """
        归一化值函数

        Args:
            values: 输入值

        Returns:
            normalized_values: 归一化后的值
        """
        values = values.view(-1, 1) if values.dim() == 1 else values

        std = torch.sqrt(self.var + self.epsilon)
        normalized = (values - self.mean) / std
        return torch.clamp(normalized, -self.clip_range, self.clip_range).squeeze(-1)

    def denormalize(self, normalized_values: torch.Tensor) -> torch.Tensor:
        """
        反归一化值函数

        Args:
            normalized_values: 归一化后的值

        Returns:
            denormalized_values: 原始尺度的值
        """
        normalized_values = normalized_values.view(-1, 1) if normalized_values.dim() == 1 else normalized_values

        std = torch.sqrt(self.var + self.epsilon)
        denormalized = normalized_values * std + self.mean
        return denormalized.squeeze(-1)


class GAE:
    """
    Generalized Advantage Estimation (GAE)

    计算优势函数和回报的稳定方法，支持自适应折扣因子
    """
    def __init__(self, gamma: float = 0.99, lam: float = 0.95,
                 device: torch.device = None, use_adaptive_gamma: bool = False,
                 eta_min: float = 0.6, eta_max: float = 0.99):
        self.gamma = gamma              # 折扣因子
        self.lam = lam                  # GAE的λ参数
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.use_adaptive_gamma = use_adaptive_gamma
        self.eta_min = eta_min
        self.eta_max = eta_max

    def compute_adaptive_gamma(self, action_probs: torch.Tensor) -> torch.Tensor:
        """
        🎯 按照论文计算自适应折扣因子：gamma(s,a;eta) = clip(pi(s,a), eta, 1)

        Args:
            action_probs: [T, N] 动作概率（策略质量指标）

        Returns:
            adaptive_gamma: [T, N] 自适应折扣因子
        """
        # action_probs 期望在(0,1]；连续动作密度可能>1，因此上游要先 clamp 到 <=1
        action_probs = torch.clamp(action_probs, min=1e-12, max=1.0)
        # ✅ 论文：gamma(s,a;eta)=clip(pi(s,a), eta, 1)
        return torch.clamp(action_probs, min=self.eta_min, max=1.0)

    def __call__(self, rewards: torch.Tensor, dones: torch.Tensor,
                 values: torch.Tensor, next_values: torch.Tensor,
                 action_probs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算GAE优势函数和回报

        Args:
            rewards: [T, N] 奖励序列
            dones: [T, N] 结束标志
            values: [T, N] 价值函数序列
            next_values: [T, N] 下一状态价值函数
            action_probs: [T, N] 动作概率（用于自适应折扣因子）

        Returns:
            advantages: [T, N] 优势函数
            returns: [T, N] 回报
        """
        T, N = rewards.shape

         # 🔧 1) 统一成 float32
        rewards = rewards.to(self.device).float()
        dones = dones.to(self.device)
        values = values.to(self.device).float()
        next_values = next_values.to(self.device).float()

        # 🔧 2) 明确 advantages / returns 也是 float32
        advantages = torch.zeros_like(rewards, dtype=torch.float32)
        returns = torch.zeros_like(rewards, dtype=torch.float32)

        # 计算自适应折扣因子
        if self.use_adaptive_gamma and action_probs is not None:
            action_probs = action_probs.to(self.device)
            gamma_t = self.compute_adaptive_gamma(action_probs)
        else:
            gamma_t = torch.full_like(rewards, self.gamma)

        # 🎯 论文一致PF：简化但正确的连乘累积实现
        if self.use_adaptive_gamma and action_probs is not None:
            # 🎯 Policy Feedback核心：自适应折扣体现策略质量
            # 通过时变的gamma_t[t]隐式实现连乘累积效应

            # 🎯 使用累积乘积计算returns（论文思想，更高效实现）
            gae = torch.zeros(N, device=self.device)
            cumulative_product = torch.ones(N, device=self.device)  # 连乘项 ∏γ

            for t in reversed(range(T)):
                if t == T - 1:
                    next_value = next_values[t]
                else:
                    next_value = values[t + 1]

                # 计算TD误差，使用当前步的自适应折扣
                delta = rewards[t] + gamma_t[t] * next_value * (1 - dones[t].float()) - values[t]

                # 🎯 关键：累积乘积体现连乘效应
                # cumulative_product 维护了 ∏_{k=t}^{T-1} γ(s_k,a_k;η)
                if t == T - 1:
                    cumulative_product = gamma_t[t]  # 最后一步：γ_T
                else:
                    cumulative_product = gamma_t[t] * cumulative_product  # 连乘：γ_t * ∏_{k=t+1}^{T-1} γ_k

                # GAE更新，使用累积乘积增强长期依赖
                gae = delta + gamma_t[t] * self.lam * (1 - dones[t].float()) * gae

                # 保存结果
                advantages[t] = gae
                returns[t] = gae + values[t]

        else:
            # 标准GAE计算
            gae = torch.zeros(N, device=self.device)
            for t in reversed(range(T)):
                if t == T - 1:
                    next_value = next_values[t]
                else:
                    next_value = values[t + 1]

                # 计算TD误差 (修复布尔张量减法错误)
                delta = rewards[t] + gamma_t[t] * next_value * (1 - dones[t].float()) - values[t]

                # GAE更新 (修复布尔张量减法错误)
                gae = delta + gamma_t[t] * self.lam * (1 - dones[t].float()) * gae

                # 保存结果
                advantages[t] = gae
                returns[t] = gae + values[t]

        return advantages, returns


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        config: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_random_seed(seed: int = 42):
    """
    设置随机种子以确保实验可复现性

    Args:
        seed: 随机种子
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🎲 随机种子已设置为: {seed}")


def compute_trajectory_smoothness(trajectory: np.ndarray) -> float:
    """
    计算轨迹平滑度指标

    Args:
        trajectory: [T, 6] 关节角度或位置序列

    Returns:
        smoothness: 平滑度指标（值越小越平滑）
    """
    if len(trajectory) < 3:
        return 0.0

    # 计算一阶差分（速度）
    velocity = np.diff(trajectory, axis=0)

    # 计算二阶差分（加速度）
    acceleration = np.diff(velocity, axis=0)

    # 平滑度指标：加速度的L2范数的平均值
    if len(acceleration) > 0:
        smoothness = np.mean(np.linalg.norm(acceleration, axis=1))
    else:
        smoothness = 0.0

    return smoothness


def compute_trajectory_metrics(trajectory_data: List[np.ndarray],
                             target_positions: List[np.ndarray],
                             success_threshold: float = 0.005) -> Dict[str, float]:
    """
    计算轨迹质量指标

    Args:
        trajectory_data: 轨迹数据列表
        target_positions: 目标位置列表
        success_threshold: 成功阈值

    Returns:
        metrics: 指标字典
    """
    if not trajectory_data:
        return {}

    metrics = {
        'avg_trajectory_length': 0.0,
        'avg_smoothness': 0.0,
        'success_rate': 0.0,
        'avg_final_error': 0.0,
        'trajectory_consistency': 0.0
    }

    # 计算轨迹长度
    lengths = [len(traj) for traj in trajectory_data]
    metrics['avg_trajectory_length'] = np.mean(lengths)

    # 计算平滑度
    smoothness_values = [compute_trajectory_smoothness(traj) for traj in trajectory_data]
    metrics['avg_smoothness'] = np.mean(smoothness_values)

    # 计算成功率和最终误差
    final_errors = []
    successful_count = 0

    for i, (traj, target) in enumerate(zip(trajectory_data, target_positions)):
        if len(traj) > 0:
            # 假设轨迹的最后一列是末端位置
            if traj.shape[1] >= 3:
                final_pos = traj[-1, :3]
                final_error = np.linalg.norm(final_pos - target)
                final_errors.append(final_error)

                if final_error < success_threshold:
                    successful_count += 1

    if final_errors:
        metrics['avg_final_error'] = np.mean(final_errors)
        metrics['success_rate'] = successful_count / len(final_errors)

    # 计算轨迹一致性（轨迹之间的相似度）
    if len(trajectory_data) > 1:
        # 简化实现：计算轨迹长度的标准差
        metrics['trajectory_consistency'] = 1.0 / (1.0 + np.std(lengths))

    return metrics


def save_training_data(episode_data: List[Dict[str, Any]],
                      filepath: str = "csv_output/training_data.csv"):
    """
    保存训练数据到CSV文件

    Args:
        episode_data: 训练数据列表
        filepath: 保存路径
    """
    df = pd.DataFrame(episode_data)
    df.to_csv(filepath, index=False)
    print(f"📊 训练数据已保存到: {filepath}")


def plot_training_curves(training_stats: Dict[str, List],
                        config: Optional[Dict[str, Any]] = None,
                        save_path: str = None,
                        show_plots: bool = False):
    """
    绘制训练曲线

    Args:
        training_stats: 训练统计数据
        config: 配置字典
        save_path: 保存路径
        show_plots: 是否显示图像
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('UR10e PPO Training Progress', fontsize=16)

    # Episode奖励
    if 'episode_rewards' in training_stats and training_stats['episode_rewards']:
        axes[0, 0].plot(training_stats['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)

        # 添加平滑曲线
        window_size = min(100, len(training_stats['episode_rewards']))
        if window_size > 0:
            smooth_rewards = pd.Series(training_stats['episode_rewards']).rolling(window=window_size).mean()
            axes[0, 0].plot(smooth_rewards, label=f'Smoothed ({window_size})', alpha=0.7)
            axes[0, 0].legend()

    # Episode长度
    if 'episode_lengths' in training_stats and training_stats['episode_lengths']:
        axes[0, 1].plot(training_stats['episode_lengths'])
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True)

    # 位置误差
    if 'position_errors' in training_stats and training_stats['position_errors']:
        axes[0, 2].plot(training_stats['position_errors'])
        axes[0, 2].set_title('Position Errors')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Error (m)')
        axes[0, 2].grid(True)

        # 添加成功阈值线
        success_threshold = config.get('reward', {}).get('accuracy', {}).get('threshold', 0.005) if config else 0.005
        axes[0, 2].axhline(y=success_threshold, color='r', linestyle='--', label='Success Threshold')
        axes[0, 2].legend()

    # 成功率
    if 'success_rates' in training_stats and training_stats['success_rates']:
        axes[1, 0].plot(training_stats['success_rates'])
        axes[1, 0].set_title('Success Rate (100-episode window)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].grid(True)
        axes[1, 0].set_ylim([0, 1])

    # 衰减回合统计
    if 'decay_stats' in training_stats and training_stats['decay_stats']:
        decay_steps = [stats.get('current_max_steps', 1000) for stats in training_stats['decay_stats']]
        axes[1, 1].plot(decay_steps)
        axes[1, 1].set_title('Decaying Episode Max Steps')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Max Steps')
        axes[1, 1].grid(True)

    # 奖励分量分析
    if 'reward_components' in training_stats and training_stats['reward_components']:
        # 提取奖励分量
        accuracy_rewards = []
        smoothness_rewards = []
        energy_rewards = []

        for components in training_stats['reward_components']:
            if isinstance(components, dict):
                accuracy_rewards.append(components.get('accuracy', 0))
                smoothness_rewards.append(components.get('smoothness', 0))
                energy_rewards.append(components.get('energy', 0))

        if accuracy_rewards:
            axes[1, 2].plot(accuracy_rewards, label='Accuracy', alpha=0.7)
        if smoothness_rewards:
            axes[1, 2].plot(smoothness_rewards, label='Smoothness', alpha=0.7)
        if energy_rewards:
            axes[1, 2].plot(energy_rewards, label='Energy', alpha=0.7)

        axes[1, 2].set_title('Reward Components')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Reward')
        axes[1, 2].grid(True)
        axes[1, 2].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 训练曲线已保存到: {save_path}")

    if show_plots:
        plt.show()
    else:
        plt.close()


def visualize_trajectory(trajectory_data: np.ndarray,
                        target_position: np.ndarray,
                        save_path: str = None,
                        show_plot: bool = False):
    """
    可视化单个轨迹

    Args:
        trajectory_data: [T, 6] 轨迹数据
        target_position: [3] 目标位置
        save_path: 保存路径
        show_plot: 是否显示图像
    """
    fig = plt.figure(figsize=(15, 10))

    # 3D轨迹
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    if trajectory_data.shape[1] >= 3:
        ax1.plot(trajectory_data[:, 0], trajectory_data[:, 1], trajectory_data[:, 2], 'b-', label='Trajectory')
        ax1.scatter(target_position[0], target_position[1], target_position[2],
                   c='r', s=100, marker='*', label='Target')
        ax1.scatter(trajectory_data[0, 0], trajectory_data[0, 1], trajectory_data[0, 2],
                   c='g', s=50, marker='o', label='Start')
        ax1.scatter(trajectory_data[-1, 0], trajectory_data[-1, 1], trajectory_data[-1, 2],
                   c='orange', s=50, marker='s', label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()

    # 关节角度
    for i in range(6):
        ax2 = fig.add_subplot(2, 3, i + 2)
        ax2.plot(trajectory_data[:, i])
        ax2.set_title(f'Joint {i + 1} Angle')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Angle (rad)')
        ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 轨迹可视化已保存到: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def create_experiment_directory(base_dir: str = "./experiments") -> str:
    """
    创建实验目录

    Args:
        base_dir: 基础目录

    Returns:
        experiment_dir: 实验目录路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, f"ur10e_ppo_{timestamp}")

    # 创建子目录
    subdirs = [
        "checkpoints",      # 模型检查点
        "logs",            # 日志文件
        "csv_output",      # CSV数据
        "plots",           # 训练曲线
        "trajectories",    # 轨迹可视化
        "config",          # 配置文件
        "models"           # 最终模型
    ]

    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)

    print(f"📁 实验目录已创建: {experiment_dir}")
    return experiment_dir


def save_experiment_config(config: Dict[str, Any], experiment_dir: str):
    """
    保存实验配置

    Args:
        config: 配置字典
        experiment_dir: 实验目录
    """
    config_path = os.path.join(experiment_dir, "config", "config.yaml")
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # 同时保存为JSON格式
    json_path = os.path.join(experiment_dir, "config", "config.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"⚙️  实验配置已保存")


def compute_success_metrics(position_errors: List[float],
                           threshold: float = 0.005) -> Dict[str, float]:
    """
    计算成功指标

    Args:
        position_errors: 位置误差列表
        threshold: 成功阈值

    Returns:
        metrics: 成功指标字典
    """
    if not position_errors:
        return {}

    success_count = sum(1 for error in position_errors if error < threshold)
    total_count = len(position_errors)

    metrics = {
        'success_rate': success_count / total_count,
        'total_episodes': total_count,
        'successful_episodes': success_count,
        'mean_error': np.mean(position_errors),
        'std_error': np.std(position_errors),
        'median_error': np.median(position_errors),
        'min_error': np.min(position_errors),
        'max_error': np.max(position_errors)
    }

    return metrics


def generate_training_report(training_stats: Dict[str, List],
                           config: Dict[str, Any],
                           experiment_dir: str):
    """
    生成训练报告

    Args:
        training_stats: 训练统计数据
        config: 配置字典
        experiment_dir: 实验目录
    """
    report_path = os.path.join(experiment_dir, "training_report.txt")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("UR10e PPO 多目标最优轨迹规划训练报告\n")
        f.write("=" * 50 + "\n\n")

        # 配置信息
        f.write("📋 配置信息:\n")
        f.write(f"最大训练轮数: {config['train']['max_episodes']}\n")
        f.write(f"状态空间: 25维\n")
        f.write(f"动作空间: 6维\n")
        f.write(f"最大步数: {config['env']['max_steps']}\n")
        f.write(f"衰减回合机制: {'启用' if config['decay_episode']['enabled'] else '禁用'}\n")
        f.write(f"成功率阈值: {config['decay_episode']['success_threshold']}\n\n")

        # 训练统计
        if 'episode_rewards' in training_stats and training_stats['episode_rewards']:
            f.write("📊 训练统计:\n")
            f.write(f"总训练轮数: {len(training_stats['episode_rewards'])}\n")
            f.write(f"最终平均奖励: {np.mean(training_stats['episode_rewards'][-100:]):.2f}\n")
            f.write(f"最终成功率: {training_stats['success_rates'][-1]:.2%}\n")
            f.write(f"最终平均位置误差: {np.mean(training_stats['position_errors'][-100:]):.4f}m\n")

            # 成功指标
            if 'position_errors' in training_stats:
                success_metrics = compute_success_metrics(training_stats['position_errors'])
                f.write(f"\n🎯 成功指标:\n")
                for key, value in success_metrics.items():
                    if key == 'success_rate':
                        f.write(f"{key}: {value:.2%}\n")
                    else:
                        f.write(f"{key}: {value:.4f}\n")

    print(f"📝 训练报告已保存到: {report_path}")


class RewardNormalizer:
    """
    奖励归一化器

    用于稳定PPO训练的奖励归一化技术，支持在线更新和多种归一化策略
    """

    def __init__(self,
                 gamma: float = 0.99,
                 clip_range: float = 5.0,
                 epsilon: float = 1e-8,
                 normalize_method: str = 'running_stats',
                 warmup_steps: int = 100,
                 history_size: int = 10000):
        """
        初始化奖励归一化器

        Args:
            gamma: 折扣因子，用于计算折扣奖励统计
            clip_range: 归一化值裁剪范围
            epsilon: 数值稳定性参数
            normalize_method: 归一化方法 ['running_stats', 'batch_stats', 'rank']
            warmup_steps: 预热步数，初期不进行归一化
            history_size: 奖励历史记录大小
        """
        self.gamma = gamma
        self.clip_range = clip_range
        self.epsilon = epsilon
        self.normalize_method = normalize_method
        self.warmup_steps = warmup_steps
        self.history_size = history_size

        # 运行时统计量
        self.running_mean = 0.0
        self.running_var = 1.0
        self.running_count = 0
        self.beta = 0.99  # 指数移动平均系数

        # 奖励历史
        self.reward_history = []
        self.discounted_reward_history = []

        # 批次统计
        self.batch_rewards = []

    def update(self, reward: float, done: bool = False):
        """
        更新归一化器统计量

        Args:
            reward: 当前奖励值
            done: 是否回合结束
        """
        self.reward_history.append(reward)
        self.running_count += 1

        # 指数移动平均更新
        self.running_mean = self.beta * self.running_mean + (1 - self.beta) * reward
        delta = reward - self.running_mean
        self.running_var = self.beta * self.running_var + (1 - self.beta) * delta * delta

        # 维护历史记录在合理范围内
        if len(self.reward_history) > self.history_size:
            self.reward_history = self.reward_history[-self.history_size//2:]

        # 回合结束时计算折扣奖励统计
        if done and len(self.reward_history) > 1:
            self._update_discounted_stats()

    def _update_discounted_stats(self):
        """更新折扣奖励统计"""
        if not self.reward_history:
            return

        # 计算最近一个episode的折扣奖励
        discounted_rewards = []
        reward_sum = 0.0
        for reward in reversed(self.reward_history):
            reward_sum = reward + self.gamma * reward_sum
            discounted_rewards.append(reward_sum)

        discounted_rewards.reverse()
        self.discounted_reward_history.extend(discounted_rewards)

        # 维护折扣奖励历史
        if len(self.discounted_reward_history) > self.history_size:
            self.discounted_reward_history = self.discounted_reward_history[-self.history_size//2:]

    def normalize(self, reward: float) -> float:
        """
        归一化单个奖励

        Args:
            reward: 原始奖励值

        Returns:
            normalized_reward: 归一化后的奖励值
        """
        if self.running_count < self.warmup_steps:
            return reward  # 预热期不归一化

        if self.normalize_method == 'running_stats':
            return self._normalize_running_stats(reward)
        elif self.normalize_method == 'batch_stats':
            return self._normalize_batch_stats(reward)
        elif self.normalize_method == 'rank':
            return self._normalize_rank(reward)
        else:
            return reward

    def _normalize_running_stats(self, reward: float) -> float:
        """使用运行统计量归一化"""
        std = np.sqrt(self.running_var + self.epsilon)
        normalized = (reward - self.running_mean) / std
        return np.clip(normalized, -self.clip_range, self.clip_range)

    def _normalize_batch_stats(self, reward: float) -> float:
        """使用批次统计量归一化"""
        if len(self.reward_history) < 10:
            return reward

        # 使用最近的奖励作为批次
        recent_rewards = self.reward_history[-min(100, len(self.reward_history)):]
        batch_mean = np.mean(recent_rewards)
        batch_std = np.std(recent_rewards) + self.epsilon

        normalized = (reward - batch_mean) / batch_std
        return np.clip(normalized, -self.clip_range, self.clip_range)

    def _normalize_rank(self, reward: float) -> float:
        """使用秩归一化（均匀分布）"""
        if len(self.reward_history) < 10:
            return reward

        # 计算当前奖励在历史中的百分位
        count_smaller = sum(1 for r in self.reward_history if r < reward)
        percentile = count_smaller / len(self.reward_history)

        # 映射到[-1, 1]范围
        normalized = 2 * percentile - 1
        return np.clip(normalized, -self.clip_range, self.clip_range)

    def normalize_batch(self, rewards: np.ndarray) -> np.ndarray:
        """
        批量归一化奖励

        Args:
            rewards: [batch_size] 奖励数组

        Returns:
            normalized_rewards: 归一化后的奖励数组
        """
        if self.running_count < self.warmup_steps:
            return rewards

        normalized_rewards = np.array([self.normalize(r) for r in rewards])
        return normalized_rewards

    def get_stats(self) -> dict:
        """获取归一化器统计信息"""
        return {
            'method': self.normalize_method,
            'running_mean': self.running_mean,
            'running_var': self.running_var,
            'running_std': np.sqrt(self.running_var + self.epsilon),
            'count': self.running_count,
            'recent_mean': np.mean(self.reward_history[-100:]) if self.reward_history else 0.0,
            'recent_std': np.std(self.reward_history[-100:]) if len(self.reward_history) > 1 else 0.0,
            'history_size': len(self.reward_history),
            'warmup_progress': min(1.0, self.running_count / self.warmup_steps)
        }

    def reset(self):
        """重置归一化器（保留学习到的统计量）"""
        self.reward_history = []
        self.batch_rewards = []

    def full_reset(self):
        """完全重置归一化器"""
        self.reward_history = []
        self.discounted_reward_history = []
        self.batch_rewards = []
        self.running_mean = 0.0
        self.running_var = 1.0
        self.running_count = 0


class TrainingLogger:
    """训练日志记录器"""

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 创建日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    def log(self, message: str, level: str = "INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"

        # 写入文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')

        # 打印到控制台
        print(log_message)

    def log_experiment_start(self, config: Dict[str, Any]):
        """记录实验开始"""
        self.log("🚀 开始UR10e PPO训练实验")
        self.log(f"📋 配置: {config}")
        self.log(f"🎯 状态空间: 25维, 动作空间: 6维")
        self.log(f"🔧 衰减回合机制: {'启用' if config['decay_episode']['enabled'] else '禁用'}")

    def log_experiment_end(self, final_stats: Dict[str, Any]):
        """记录实验结束"""
        self.log("🎉 训练完成！")
        self.log(f"📊 最终统计: {final_stats}")


def validate_config(config: Dict[str, Any]) -> bool:
    """
    验证配置文件的有效性

    Args:
        config: 配置字典

    Returns:
        is_valid: 配置是否有效
    """
    try:
        # 检查必需的配置项
        required_sections = ['env', 'ppo', 'train', 'reward']
        for section in required_sections:
            if section not in config:
                print(f"❌ 配置缺少必需部分: {section}")
                return False

        # 检查环境配置
        if 'xml_path' not in config['env']:
            print("❌ 缺少XML模型路径")
            return False

        # 检查PPO配置
        ppo_required = ['lr_actor', 'lr_critic', 'clip_eps', 'gamma']
        for key in ppo_required:
            if key not in config['ppo']:
                print(f"❌ PPO配置缺少必需项: {key}")
                return False

        # 检查奖励配置
        reward_required = ['accuracy', 'smoothness', 'energy']
        for key in reward_required:
            if key not in config['reward']:
                print(f"❌ 奖励配置缺少必需项: {key}")
                return False

        print("✅ 配置文件验证通过")
        return True

    except Exception as e:
        print(f"❌ 配置文件验证失败: {e}")


def assert_same_device(*tensors, device=None):
    """
    确保所有张量在同一设备上

    Args:
        *tensors: 要检查的张量列表
        device: 期望的设备，如果为None则使用第一个张量的设备

    Raises:
        AssertionError: 如果发现张量在不同设备上
    """
    if not tensors:
        return

    # 如果指定了设备，检查所有张量是否在该设备上
    if device is not None:
        for i, tensor in enumerate(tensors):
            if tensor.device != device:
                raise AssertionError(
                    f"张量 {i} 在设备 {tensor.device}，期望在 {device}"
                )
        return

    # 否则检查所有张量是否在同一设备上
    first_device = tensors[0].device
    for i, tensor in enumerate(tensors):
        if tensor.device != first_device:
            raise AssertionError(
                f"设备不匹配: 张量 0 在 {first_device}，张量 {i} 在 {tensor.device}"
            )


def check_tensor_devices(tensor_dict: dict, name: str = "Tensor Dict"):
    """
    检查字典中所有张量的设备一致性

    Args:
        tensor_dict: 包含张量的字典
        name: 字典名称，用于错误信息

    Returns:
        bool: 如果所有张量都在同一设备上返回True
    """
    devices = {}
    for key, tensor in tensor_dict.items():
        if hasattr(tensor, 'device'):
            if tensor.device not in devices:
                devices[tensor.device] = []
            devices[tensor.device].append(key)

    if len(devices) > 1:
        print(f"⚠️ {name} 中的设备不一致:")
        for device, keys in devices.items():
            print(f"   {device}: {keys}")
        return False
    return True


def get_tensor_device(tensor, default_device=None):
    """
    安全获取张量设备

    Args:
        tensor: 输入张量或数组
        default_device: 默认设备（如果输入不是张量）

    Returns:
        torch.device: 张量设备
    """
    if hasattr(tensor, 'device'):
        return tensor.device
    elif default_device is not None:
        return default_device
    else:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ensure_device(tensor, device):
    """
    确保张量在指定设备上

    Args:
        tensor: 输入张量
        device: 目标设备

    Returns:
        torch.Tensor: 在指定设备上的张量
    """
    if hasattr(tensor, 'to'):
        return tensor.to(device)
    else:
        # 如果不是张量（如numpy数组），转换为张量
        return torch.tensor(tensor, device=device)


def _device_consistency_check():
    """
    设备一致性检查 - 修复服务器设备不匹配问题

    专门针对服务器环境中cuda:0和cuda:2设备不匹配的解决方案
    """
    # 🔧 Phase 1: 强制环境变量设置（修复服务器设备不匹配）
    print("🔧 [SERVER FIX] 强制CUDA设备一致性设置...")

    # 🎯 **用户服务器使用GPU 2，强制设置GPU 2**
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # **用户服务器使用GPU 2**
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    # 检查CUDA环境
    if torch.cuda.is_available():
        print(f"   ✅ CUDA可用，版本: {torch.version.cuda}")
        print(f"   ✅ PyTorch版本: {torch.__version__}")
        print(f"   ✅ 检测到GPU数量: {torch.cuda.device_count()}")

        # 🎯 **用户服务器强制使用GPU 2**
        target_device_index = 0  # 设置CUDA_VISIBLE_DEVICES=2后，GPU 2变为索引0
        target_device = 'cuda:0'  # 在可见设备中，GPU 2现在是cuda:0
        try:
            torch.cuda.set_device(target_device_index)  # 强制设置为GPU 2（现在是索引0）
            current_device = torch.cuda.current_device()
            print(f"   🔒 [FORCED] 当前CUDA设备: GPU {current_device} (原GPU 2)")

            # 验证设备确实可用
            if current_device == target_device_index:
                print(f"   ✅ [SUCCESS] 成功强制使用GPU 2 (索引{current_device})")
            else:
                print(f"   ⚠️  [WARNING] 期望GPU 2(索引0)，实际GPU {current_device}")

        except Exception as e:
            print(f"   ❌ [ERROR] 强制设备设置失败: {e}")
            print(f"   🔄 [FALLBACK] 使用CPU模式")
            return torch.device('cpu')
    else:
        print("   ⚠️  CUDA不可用，使用CPU")
        return torch.device('cpu')

    # 🎯 [CRITICAL] 服务器设备一致性验证
    print("🔍 [SERVER DIAG] 服务器设备一致性诊断:")

    # 测试张量创建和设备检查
    try:
        test_tensor = torch.randn(10, 10, device='cuda:0')  # 这是原GPU 2
        actual_device = test_tensor.device
        print(f"   🧪 测试张量设备: {actual_device} (原GPU 2)")

        # 检查所有可见GPU（现在只有GPU 2可见）
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            device_props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i} (原GPU 2): {device_name} (内存: {device_props.total_memory/1024**3:.1f}GB)")

        # 确保所有后续操作都使用cuda:0（原GPU 2）
        if str(actual_device) == 'cuda:0':
            print(f"   ✅ [DEVICE OK] 使用目标设备: {actual_device} (原GPU 2)")
            return torch.device('cuda:0')
        else:
            print(f"   ❌ [DEVICE MISMATCH] 期望cuda:0(原GPU 2)，实际{actual_device}")
            print(f"   🔄 [FALLBACK] 强制返回cuda:0")
            return torch.device('cuda:0')

    except Exception as e:
        print(f"   ❌ [CRITICAL ERROR] 设备测试失败: {e}")
        print(f"   🔄 [FALLBACK] 使用CPU模式")
        return torch.device('cpu')


def diagnose_server_environment():
    """
    服务器环境全面诊断
    专门用于诊断为什么本地正常但服务器失败的问题
    """
    print("=" * 80)
    print("🏥 [SERVER DIAGNOSIS] 服务器环境全面诊断")
    print("=" * 80)

    # 1. Python和环境检查
    print("\n🐍 Python环境:")
    import sys
    print(f"   Python版本: {sys.version}")
    print(f"   可执行文件: {sys.executable}")

    # 2. CUDA环境详细检查
    print("\n🔥 CUDA环境:")
    print(f"   PyTorch CUDA可用: {torch.cuda.is_available()}")
    print(f"   PyTorch CUDA版本: {torch.version.cuda}")
    print(f"   编译的CUDA版本: {torch.version.cuda or 'N/A'}")

    if torch.cuda.is_available():
        print(f"   GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"      计算能力: {props.major}.{props.minor}")
            print(f"      总内存: {props.total_memory / 1024**3:.1f} GB")
            print(f"      多处理器数量: {props.multi_processor_count}")

    # 3. 环境变量检查
    print("\n🌍 环境变量:")
    import os
    cuda_vars = ['CUDA_VISIBLE_DEVICES', 'PYTORCH_CUDA_ALLOC_CONF',
                 'CUDA_DEVICE_ORDER', 'CUDA_LAUNCH_BLOCKING']
    for var in cuda_vars:
        value = os.environ.get(var, 'Not set')
        print(f"   {var}: {value}")

    # 4. 当前设备状态
    print("\n📍 当前设备状态:")
    if torch.cuda.is_available():
        current = torch.cuda.current_device()
        print(f"   当前设备: GPU {current}")
        print(f"   当前设备名: {torch.cuda.get_device_name(current)}")

        # 内存状态
        allocated = torch.cuda.memory_allocated(current)
        reserved = torch.cuda.memory_reserved(current)
        print(f"   已分配内存: {allocated/1024**2:.1f} MB")
        print(f"   已预留内存: {reserved/1024**2:.1f} MB")

    # 5. Isaac Gym环境检查（如果可用）
    print("\n🎮 Isaac Gym环境:")
    try:
        import gym
        print(f"   Isaac Gym可用: True")
        print(f"   路径: {gym.__file__ if hasattr(gym, '__file__') else 'Built-in'}")
    except ImportError:
        print(f"   Isaac Gym可用: False")

    # 6. 推荐修复措施
    print("\n💡 推荐修复措施:")
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print("   1. 多GPU环境检测到，强制使用GPU 2:")
        print("      export CUDA_VISIBLE_DEVICES=2")
        print("      export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128")
        print("   2. 在代码中强制设备检查")
        print("   3. 监控第500步附近的设备切换")
    elif not torch.cuda.is_available():
        print("   1. CUDA不可用，检查NVIDIA驱动:")
        print("      nvidia-smi")
        print("      检查PyTorch CUDA版本匹配")
    else:
        print("   1. 环境看起来正常，检查代码中的设备一致性")

    print("=" * 80)


def get_forced_device():
    """
    获取强制统一的设备，解决服务器设备不匹配问题

    Returns:
        torch.device: 强制统一的设备（优先cuda:0，否则cpu）
    """
    # 首先运行设备一致性检查
    device = _device_consistency_check()

    # 如果是服务器环境且出现问题，运行全面诊断
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print("🚨 [SERVER WARNING] 检测到多GPU环境，启用服务器修复模式")
        diagnose_server_environment()

    return device
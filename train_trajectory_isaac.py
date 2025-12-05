"""
轨迹跟踪训练脚本 - Isaac Gym版本
使用Task-Space RRT* + PPO进行UR10e机械臂轨迹跟踪训练
"""

import os
import sys
import argparse
import yaml
import numpy as np
from typing import Dict, Any

from ppo_isaac import PPOIsaac
from ur10e_trajectory_env_isaac import UR10eTrajectoryEnvIsaac
import torch

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠️ 配置文件 {config_path} 未找到，使用默认配置")
        config = get_default_config()
    return config


def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    return {
        'env': {
            'num_envs': 1,
            'max_steps': 500,
            'dt': 0.01
        },
        'device': 'cuda:0',
        'ppo': {
            'lr_actor': 5e-4,
            'lr_critic': 5e-4,
            'clip_eps': 0.15,
            'gamma': 0.995,
            'lam': 0.95,
            'entropy_coef': 0.01,
            'value_coef': 0.5,
            'max_grad_norm': 0.5
        },
        'train': {
            'rollout_length': 512,
            'batch_size': 64,
            'num_updates': 10,
            'num_episodes': 10000,
            'save_interval': 100,
            'log_interval': 10
        },
        'trajectory_tracking': {
            'waypoint_threshold': 0.15,
            'waypoint_bonus': 5.0,
            'smooth_coef': 0.1,
            'use_deviation_penalty': False,
            'deviation_coef': 2.0,
            'distance_weight': 2.0,
            'progress_weight': 3.0
        },
        'task_space': {
            'workspace_bounds': {
                'x': [-0.8, 0.8],
                'y': [-0.8, 0.8],
                'z': [0.1, 1.0]
            }
        },
        'ts_rrt_star': {
            'replanning_threshold': 0.1,
            'max_waypoints': 50
        }
    }


def main():
    parser = argparse.ArgumentParser(description='UR10e Trajectory Tracking Training - Isaac Gym')
    parser.add_argument('--config', type=str, default='config_isaac.yaml',
                       help='配置文件路径')
    parser.add_argument('--num_envs', type=int, default=1,
                       help='并行环境数量')
    parser.add_argument('--device_id', type=int, default=0,
                       help='GPU设备ID')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='训练回合数')
    parser.add_argument('--mode', type=str, default='trajectory_tracking',
                       choices=['trajectory_tracking', 'point_to_point'],
                       help='训练模式')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_trajectory',
                       help='模型保存目录')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的checkpoint路径')

    args = parser.parse_args()

    print("🚀 UR10e Trajectory Tracking Training - Isaac Gym")
    print(f"   配置文件: {args.config}")
    print(f"   训练模式: {args.mode}")
    print(f"   并行环境: {args.num_envs}")
    print(f"   GPU设备ID: {args.device_id}")
    print(f"   训练回合: {args.episodes}")

    # 加载配置
    config = load_config(args.config)

    # 覆盖配置参数
    #config['env']['num_envs'] = args.num_envs
    #config['train']['num_episodes'] = args.episodes

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    print("🎮 初始化Isaac Gym环境...")

    # 创建轨迹跟踪环境
    env = UR10eTrajectoryEnvIsaac(
        config_path=args.config,
        num_envs=config['env']['num_envs'],
        mode=args.mode
    )

    print("🤖 初始化PPO训练器...")

    # 创建PPO训练器
    ppo = PPOIsaac(env, config)

    # 如果指定了恢复训练，加载模型
    if args.resume:
        if os.path.exists(args.resume):
            print(f"📂 恢复训练: {args.resume}")
            ppo.load_model(args.resume)
        else:
            print(f"⚠️ Checkpoint文件不存在: {args.resume}")

    print("🏃 开始训练...")

    # 开始训练
    training_stats = ppo.train(
        num_episodes=args.episodes,
        save_dir=args.save_dir
    )

    print("🎉 训练完成！")

    # 关闭环境
    env.close()

    print("✅ 程序正常退出")


if __name__ == "__main__":
    main()
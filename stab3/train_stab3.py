#!/usr/bin/env python3

"""
UR10e Stable-Baselines3 PPO训练脚本

基于stable-baselines3库实现的UR10e机械臂强化学习训练，
替代原有的自定义Isaac Gym PPO实现。

主要特性:
- 使用stable-baselines3的PPO实现
- 6D增量力矩控制 (Δτ₁, Δτ₂, ..., Δτ₆)
- Isaac Gym物理仿真
- 与原始train_isaac_fixed.py保持相同的命令行接口
"""

# IMPORTANT: Isaac Gym must be imported before PyTorch
import os
import sys
import time
import signal
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

# Isaac Gym imports MUST be before PyTorch
try:
    from isaacgym import gymapi
    from isaacgym import gymtorch
    from isaacgym import gymutil
    from isaacgym.torch_utils import *
    print("✅ Isaac Gym imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Isaac Gym: {e}")
    print("Please ensure Isaac Gym is properly installed")
    sys.exit(1)

# Now import PyTorch after Isaac Gym
import torch

# Stable-Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Import our utilities after Isaac Gym
from utils_stab3 import (
    load_config_stab3, check_environment, test_basic_isaac_gym,
    test_stable_baselines3_components, get_forced_device,
    setup_training_directories, parse_arguments,
    print_system_info, validate_config, TrainingProgressCallback, exiter
)
from ur10e_env_stab3 import UR10eEnvStab3, make_ur10e_env_stab3

class TrainingMonitorCallback(BaseCallback):
    """自定义训练监控回调"""

    def __init__(self, eval_freq: int = 10000, save_freq: int = 50000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.best_mean_reward = -np.inf
        self.start_time = time.time()

    def _on_rollout_start(self) -> None:
        """收集rollout前调用"""
        pass

    def _on_rollout_end(self) -> None:
        """收集rollout后调用"""
        pass

    def _on_step(self) -> bool:
        """每步后调用"""
        if self.n_calls % self.eval_freq == 0:
            # 评估当前策略
            rewards = []
            distances = []
            successes = 0

            # 创建评估环境
            eval_env = make_ur10e_env_stab3(
                config_path=self.training_env.envs[0].config_path,
                render=False
            )

            for _ in range(10):  # 10个评估回合
                obs, _ = eval_env.reset()
                done = False
                episode_reward = 0
                episode_distances = []

                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    episode_distances.append(info.get('distance', 1.0))

                rewards.append(episode_reward)
                distances.append(np.mean(episode_distances))
                if episode_distances[-1] < 0.05:  # 成功阈值
                    successes += 1

            eval_env.close()

            mean_reward = np.mean(rewards)
            mean_distance = np.mean(distances)
            success_rate = successes / 10

            elapsed_time = time.time() - self.start_time

            print(f"\n📈 第{self.n_calls}步评估:")
            print(f"   🎯 平均奖励: {mean_reward:.4f}")
            print(f"   📏 平均距离: {mean_distance:.4f}m")
            print(f"   ✅ 成功率: {success_rate*100:.1f}%")
            print(f"   ⏱️ 已用时间: {elapsed_time/60:.1f}分钟")

            # 记录到tensorboard
            self.logger.record('eval/mean_reward', mean_reward)
            self.logger.record('eval/mean_distance', mean_distance)
            self.logger.record('eval/success_rate', success_rate)

        # 检查优雅退出
        if exiter.shutdown:
            print("\n🛑 收到退出信号，正在停止训练...")
            return False

        return True


def make_single_env(config_path: str, device_id: int = 0, render: bool = False):
    """创建单个环境的工厂函数"""
    def _init():
        env = make_ur10e_env_stab3(
            config_path=config_path,
            num_envs=1,
            device_id=device_id,
            render=render
        )
        return Monitor(env)
    return _init


def create_vectorized_env(config_path: str, num_envs: int, device_id: int = 0, render: bool = False):
    """创建向量化环境"""
    if num_envs == 1:
        # 单环境使用DummyVecEnv
        env = DummyVecEnv([make_single_env(config_path, device_id, render)])
    else:
        # 多环境使用SubprocVecEnv
        env = SubprocVecEnv([
            make_single_env(config_path, device_id, render)
            for _ in range(num_envs)
        ])
    return env


def main():
    """主训练函数"""
    # 解析命令行参数
    args = parse_arguments()

    print("🚀 UR10e Stable-Baselines3 PPO训练开始")
    print("=" * 50)
    print(f"配置文件: {args.config}")
    # 提前读取config以显示正确的参数（与原始脚本一致）
    try:
        import yaml
        with open(args.config, 'r', encoding='utf-8') as f:
            config_preview = yaml.safe_load(f)
        print(f"并行环境数: {config_preview['env']['num_envs']} (来自config)")
        print(f"GPU设备: {config_preview['env']['device_id']} (来自config)")
        print(f"训练回合数: {config_preview.get('train', {}).get('num_episodes', 'N/A')} (来自config)")
    except Exception as e:
        print(f"⚠️ 读取配置文件失败: {e}")
        print(f"并行环境数: config文件设置")
        print(f"GPU设备: config文件设置")
        print(f"训练回合数: config文件设置")
    print(f"保存目录: {args.save_dir}")
    print(f"测试模式: {'是' if args.test else '否'}")
    print("=" * 50)

    # 打印系统信息
    print_system_info()

    # 加载配置
    print("\n📋 加载配置...")
    config = load_config_stab3(args.config)

    # 验证配置
    if not validate_config(config):
        print("❌ 配置验证失败")
        sys.exit(1)

    # 覆盖配置中的参数
    if args.render:
        config['visualization']['enable'] = True
    # 注意：args.timesteps 在原始脚本中不存在，使用配置中的训练参数

    # 保存配置副本
    save_dir = setup_training_directories(args.save_dir)
    config_path = save_dir / "config.yaml"
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"💾 配置已保存到: {config_path}")

    # 环境检查
    print("\n🔍 环境检查...")
    if not check_environment():
        print("❌ 环境检查失败")
        sys.exit(1)

    # 基本功能测试
    if not test_basic_isaac_gym():
        print("❌ Isaac Gym基本功能测试失败")
        sys.exit(1)

    # Stable-Baselines3组件测试
    if not test_stable_baselines3_components():
        print("❌ Stable-Baselines3组件测试失败")
        sys.exit(1)

    if args.test:
        print("\n🎯 仅测试模式，完成所有测试")
        print("✅ 所有测试通过，环境准备就绪！")
        return True

    # 获取强制设备
    device = get_forced_device()

    print(f"\n🎯 开始训练...")
    print(f"   设备: {device}")
    print(f"   环境: {config['env']['num_envs']}x并行")
    print(f"   训练步数: {config['ppo']['total_timesteps']}")
    print(f"   保存目录: {save_dir}")

    try:
        # 创建环境
        print(f"\n🏗️ 创建训练环境...")
        num_envs = config.get('env', {}).get('num_envs', 1)
        device_id = config.get('env', {}).get('device_id', 0)

        # 创建向量化环境
        train_env = create_vectorized_env(
            config_path=args.config,
            num_envs=num_envs,
            device_id=device_id,
            render=args.render
        )

        print("✅ 环境创建成功")

        # 检查环境兼容性
        print("🔍 检查环境兼容性...")
        check_env(train_env.envs[0], warn=True)
        print("✅ 环境与stable-baselines3兼容")

        # 获取PPO参数
        ppo_config = config.get('ppo', {})
        policy_kwargs = ppo_config.get('policy_kwargs', {})

        # 创建PPO模型
        print(f"\n🤖 创建PPO模型...")
        model = PPO(
            policy=ppo_config.get('policy', 'MlpPolicy'),
            env=train_env,
            learning_rate=ppo_config.get('learning_rate', 3e-4),
            n_steps=ppo_config.get('n_steps', 2048),
            batch_size=ppo_config.get('batch_size', 64),
            n_epochs=ppo_config.get('n_epochs', 10),
            gamma=ppo_config.get('gamma', 0.99),
            gae_lambda=ppo_config.get('gae_lambda', 0.95),
            clip_range=ppo_config.get('clip_range', 0.2),
            clip_range_vf=ppo_config.get('clip_range_vf', None),
            normalize_advantage=ppo_config.get('normalize_advantage', True),
            ent_coef=ppo_config.get('ent_coef', 0.01),
            vf_coef=ppo_config.get('vf_coef', 0.5),
            max_grad_norm=ppo_config.get('max_grad_norm', 0.5),
            use_sde=False,  # 不使用状态依赖探索
            sde_sample_freq=-1,
            target_kl=ppo_config.get('target_kl', None),
            tensorboard_log=str(save_dir / "logs"),
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            device=device
        )

        print("✅ PPO模型创建成功")

        # 恢复训练（如果指定）
        if args.resume:
            if os.path.exists(args.resume):
                print(f"🔄 恢复训练: {args.resume}")
                model = PPO.load(args.resume, env=train_env, device=device)
                print("✅ 模型加载成功")
            else:
                print(f"⚠️ 检查点文件不存在: {args.resume}")
                print("   从头开始训练")

        # 设置回调
        print(f"\n📊 设置训练回调...")

        # 评估回调
        eval_env = create_vectorized_env(args.config, 1, device_id, False)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(save_dir / "models"),
            log_path=str(save_dir / "evaluations"),
            eval_freq=ppo_config.get('eval_freq', 10000),
            n_eval_episodes=10,  # 固定为10，与原始脚本保持一致
            deterministic=True,
            render=False,
            verbose=1
        )

        # 检查点回调
        checkpoint_callback = CheckpointCallback(
            save_freq=ppo_config.get('save_freq', 50000),
            save_path=str(save_dir / "models"),
            name_prefix='ur10e_ppo'
        )

        # 自定义监控回调
        monitor_callback = TrainingMonitorCallback(
            eval_freq=ppo_config.get('eval_freq', 10000),
            save_freq=ppo_config.get('save_freq', 50000),
            verbose=1
        )

        callbacks = [eval_callback, checkpoint_callback, monitor_callback]

        print("✅ 回调设置完成")

        # 开始训练
        print(f"\n🎯 开始PPO训练...")
        total_timesteps = ppo_config.get('total_timesteps', 1000000)
        start_time = time.time()

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=10,
            progress_bar=True
        )

        end_time = time.time()
        training_time = end_time - start_time

        print(f"\n✅ 训练完成！")
        print(f"   训练时间: {training_time/60:.1f}分钟")
        print(f"   训练步数: {total_timesteps}")

        # 保存最终模型
        final_model_path = save_dir / "models" / "ur10e_ppo_final.zip"
        model.save(str(final_model_path))
        print(f"💾 最终模型已保存: {final_model_path}")

        # 关闭环境
        train_env.close()
        eval_env.close()

        return True

    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
        return False
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return False

    print("\n👋 程序结束")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3

"""
Isaac Gym PPO训练启动脚本
UR10e RL-PID混合控制 - 大规模并行训练
"""

# IMPORTANT: Isaac Gym must be imported before PyTorch
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from isaacgym.torch_utils import *

import os
import sys
import argparse
import time
import signal
from pathlib import Path

import torch
import numpy as np

# 延迟导入训练器以避免导入顺序问题
def train_ppo_isaac(config_path="config_isaac.yaml",
                      num_envs=64,
                      device_id=0,
                      episodes=1000,
                      save_dir="./checkpoints_isaac",
                      resume=None,
                      render=False,
                      debug=False):
    """PPO训练函数"""
    # 延迟导入以避免导入冲突
    from ppo_isaac import PPOIsaac, load_config_isaac
    from ur10e_env_isaac import UR10ePPOEnvIsaac

    # 这里可以开始训练逻辑
    print("✅ 训练模块导入成功，可以开始训练")
    return True


class GracefulExiter:
    """优雅退出处理器"""
    def __init__(self):
        self.shutdown = False

    def __call__(self, signum, frame):
        print(f"\n🛑 接收到退出信号 {signum}，正在优雅退出...")
        self.shutdown = True

# 设置全局退出处理器
exiter = GracefulExiter()
signal.signal(signal.SIGINT, exiter)
signal.signal(signal.SIGTERM, exiter)


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description="Isaac Gym UR10e PPO训练")
    parser.add_argument("--config", "-c", type=str, default="config_isaac.yaml",
                       help="配置文件路径")
    parser.add_argument("--num-envs", "-n", type=int, default=512,
                       help="并行环境数量")
    parser.add_argument("--device", "-d", type=int, default=0,
                       help="GPU设备ID")
    parser.add_argument("--episodes", "-e", type=int, default=1000,
                       help="训练回合数")
    parser.add_argument("--save-dir", "-s", type=str, default="./checkpoints_isaac",
                       help="模型保存目录")
    parser.add_argument("--resume", "-r", type=str, default=None,
                       help="恢复训练的检查点路径")
    parser.add_argument("--render", action="store_true",
                       help="启用渲染（降低训练速度）")
    parser.add_argument("--debug", action="store_true",
                       help="启用调试模式")

    args = parser.parse_args()

    print("🚀 Isaac Gym UR10e PPO训练开始")
    print("=" * 50)
    print(f"配置文件: {args.config}")
    print(f"并行环境数: {args.num_envs}")
    print(f"GPU设备: {args.device}")
    print(f"训练回合数: {args.episodes}")
    print(f"保存目录: {args.save_dir}")
    print("=" * 50)

    # 检查Isaac Gym环境
    try:
        from isaacgym import gymapi
        print("✅ Isaac Gym导入成功")
    except ImportError as e:
        print("❌ Isaac Gym导入失败，请确保Isaac Gym已正确安装")
        print(f"错误信息: {e}")
        sys.exit(1)

    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，Isaac Gym需要GPU支持")
        sys.exit(1)

    print(f"✅ CUDA可用，设备数: {torch.cuda.device_count()}")
    print(f"✅ 当前设备: {torch.cuda.get_device_name(args.device)}")

    # 加载配置
    print(f"\n📋 加载配置文件: {args.config}")
    try:
        config = load_config_isaac(args.config)
    except FileNotFoundError:
        print(f"⚠️ 配置文件未找到，使用默认配置")
        config = load_config_isaac()  # 使用默认配置

    # 覆盖命令行参数
    config['env']['num_envs'] = args.num_envs
    config['env']['device_id'] = args.device
    config['train']['num_episodes'] = args.episodes
    if args.render:
        config['simulator']['enable_rendering'] = True

    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置副本
    import yaml
    config_path = save_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"💾 配置已保存到: {config_path}")

    try:
        # 创建环境
        print(f"\n🏗️ 创建Isaac Gym环境...")
        env = UR10ePPOEnvIsaac(
            config_path="config_isaac.yaml",
            num_envs=config['env']['num_envs'],
            device_id=args.device
        )

        # 创建PPO训练器
        print(f"🤖 创建PPO训练器...")
        ppo = PPOIsaac(env, config)

        # 恢复训练（如果指定）
        if args.resume:
            print(f"🔄 恢复训练: {args.resume}")
            ppo.load_model(args.resume)

        # 训练循环
        print(f"\n🎯 开始训练...")
        start_time = time.time()

        for episode in range(config['train']['num_episodes']):
            if exiter.shutdown:
                print("🛑 收到退出信号，保存模型并退出...")
                break

            # 收集经验
            rollouts = ppo.collect_rollouts()

            # 更新策略
            metrics = ppo.update_policy(rollouts)

            # 日志记录
            if episode % config['train']['log_interval'] == 0:
                elapsed_time = time.time() - start_time
                best_performance = ppo.best_performance

                print(f"📊 Episode {episode:5d} | "
                      f"Best: {best_performance:8.4f} | "
                      f"Actor: {metrics['actor_loss']:8.4f} | "
                      f"Critic: {metrics['critic_loss']:8.4f} | "
                      f"Entropy: {metrics['entropy']:6.4f} | "
                      f"Time: {elapsed_time/60:6.2f}min | "
                      f"Episodes: {ppo.episode_count}")

            # 保存模型
            if episode % config['train']['save_interval'] == 0 and episode > 0:
                checkpoint_path = save_dir / f"ppo_checkpoint_{episode}.pth"
                ppo.save_model(str(save_dir), episode)
                print(f"💾 模型已保存: {checkpoint_path}")

        # 训练完成
        total_time = time.time() - start_time
        print(f"\n🎉 训练完成！")
        print(f"📊 训练统计:")
        print(f"   总用时: {total_time/60:.2f} 分钟")
        print(f"   总回合数: {ppo.episode_count}")
        print(f"   总步数: {ppo.total_steps}")
        print(f"   最佳性能: {ppo.best_performance:.4f}")

        # 保存最终模型
        final_path = save_dir / "ppo_final.pth"
        ppo.save_model(str(save_dir), "final")
        print(f"🏆 最终模型已保存: {final_path}")

        # 关闭环境
        env.close()

    except Exception as e:
        print(f"❌ 训练过程中发生错误: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    print("👋 程序结束")


def check_environment():
    """检查环境和依赖"""
    print("🔍 检查训练环境...")

    # 检查Python版本
    print(f"Python版本: {sys.version}")

    # 检查PyTorch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")

    # 检查Isaac Gym
    try:
        from isaacgym import gymapi
        print("✅ Isaac Gym可用")
    except ImportError:
        print("❌ Isaac Gym不可用")
        return False

    # 检查GPU内存
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")

    return True


if __name__ == "__main__":
    # 检查环境
    if not check_environment():
        print("❌ 环境检查失败")
        sys.exit(1)

    # 启动训练
    main()
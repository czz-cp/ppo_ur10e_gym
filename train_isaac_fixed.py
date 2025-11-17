#!/usr/bin/env python3

"""
Isaac Gym PPO训练启动脚本 - 修复版
UR10e RL-PID混合控制 - 大规模并行训练
"""

# IMPORTANT: Isaac Gym must be imported before PyTorch
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from isaacgym.torch_utils import *

import torch
import numpy as np
import os
import sys
import argparse
import time
import signal
from pathlib import Path

# 优雅退出处理器
class GracefulExiter:
    def __init__(self):
        self.shutdown = False

    def __call__(self, signum, frame):
        print(f"\n🛑 接收到退出信号 {signum}，正在优雅退出...")
        self.shutdown = True

# 设置全局退出处理器
exiter = GracefulExiter()
signal.signal(signal.SIGINT, exiter)
signal.signal(signal.SIGTERM, exiter)


def load_config_isaac(config_path: str = "config_isaac.yaml"):
    """加载Isaac Gym版本配置"""
    import yaml
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠️ 配置文件 {config_path} 未找到，使用默认配置")
        config = get_default_config_isaac()
    return config


def get_default_config_isaac():
    """获取默认Isaac Gym配置"""
    return {
        'env': {
            'num_envs': 64,
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


def check_environment():
    """检查Isaac Gym环境"""
    print("🔍 检查Isaac Gym环境...")

    # 检查CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，Isaac Gym需要GPU支持")
        return False

    print(f"✅ CUDA可用")
    print(f"   GPU数量: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")

    # 检查Isaac Gym
    try:
        gym = gymapi.acquire_gym()
        print("✅ Isaac Gym基础连接成功")
        # 注意：新版本Isaac Gym不需要release_gym
    except Exception as e:
        print(f"❌ Isaac Gym不可用: {e}")
        return False

    return True


def test_basic_isaac_gym():
    """测试基本Isaac Gym功能"""
    print(f"\n🧪 测试基本Isaac Gym功能...")

    try:
        # 创建仿真器
        sim_params = gymapi.SimParams()
        sim_params.dt = 0.01
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.physx.solver_type = 1  # 使用更稳定的求解器
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.use_gpu = True
        # 移除不支持的gpu_pipeline属性

        gym = gymapi.acquire_gym()
        sim_instance = gym.create_sim(compute_device=0, graphics_device=0, params=sim_params)

        if sim_instance is None:
            print("❌ 仿真器创建失败")
            return False

        print("✅ 基本仿真器创建成功")

        # 清理
        gym.destroy_sim(sim_instance)

        return True

    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        return False


def test_training_components():
    """测试训练组件"""
    print(f"\n🤖 测试训练组件...")

    try:
        # 延迟导入PPO组件
        from ppo_isaac import PPOIsaac

        # 创建简单的PPO配置
        config = get_default_config_isaac()

        print("✅ PPO模块导入成功")
        print(f"   Actor学习率: {config['ppo']['lr_actor']}")
        print(f"   Critic学习率: {config['ppo']['lr_critic']}")
        print(f"   批量大小: {config['train']['batch_size']}")

        return True

    except Exception as e:
        print(f"❌ 训练组件测试失败: {e}")
        return False


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description="Isaac Gym UR10e PPO训练")
    parser.add_argument("--config", "-c", type=str, default="config_isaac.yaml",
                       help="配置文件路径")
    parser.add_argument("--num-envs", "-n", type=int, default=None,
                       help="并行环境数量 (已禁用，请使用config文件)")
    parser.add_argument("--device", "-d", type=int, default=0,
                       help="GPU设备ID (已禁用，请使用config文件)")
    parser.add_argument("--episodes", "-e", type=int, default=None,
                       help="训练回合数 (已禁用，请使用config文件)")
    parser.add_argument("--save-dir", "-s", type=str, default="./checkpoints_isaac",
                       help="模型保存目录")
    parser.add_argument("--resume", "-r", type=str, default=None,
                       help="恢复训练的检查点路径")
    parser.add_argument("--render", action="store_true",
                       help="启用渲染（降低训练速度）")
    parser.add_argument("--debug", action="store_true",
                       help="启用调试模式")
    parser.add_argument("--test", action="store_true",
                       help="仅测试环境，不进行训练")

    args = parser.parse_args()

    print("🚀 Isaac Gym UR10e PPO训练开始")
    print("=" * 50)
    print(f"配置文件: {args.config}")
    # 提前读取config以显示正确的参数
    try:
        import yaml
        with open(args.config, 'r', encoding='utf-8') as f:
            config_preview = yaml.safe_load(f)
        print(f"并行环境数: {config_preview['env']['num_envs']} (来自config)")
        print(f"GPU设备: {config_preview['env']['device_id']} (来自config)")
        print(f"训练回合数: {config_preview['train']['num_episodes']} (来自config)")
    except Exception as e:
        print(f"⚠️ 读取配置文件失败: {e}")
        print(f"并行环境数: config文件设置")
        print(f"GPU设备: config文件设置")
        print(f"训练回合数: config文件设置")
    print(f"保存目录: {args.save_dir}")
    print(f"测试模式: {'是' if args.test else '否'}")
    print("=" * 50)

    # 检查环境
    if not check_environment():
        print("❌ 环境检查失败")
        sys.exit(1)

    # 基本功能测试
    if not test_basic_isaac_gym():
        print("❌ 基本功能测试失败")
        sys.exit(1)

    # 训练组件测试
    if not test_training_components():
        print("❌ 训练组件测试失败")
        sys.exit(1)

    if args.test:
        print("🎯 仅测试模式，完成所有测试")
        print("✅ 所有测试通过，Isaac Gym环境准备就绪！")
        return True

    # 延迟导入训练器以避免导入冲突
    try:
        from ppo_isaac import PPOIsaac, load_config_isaac
        from ur10e_env_isaac import UR10ePPOEnvIsaac
        from utils import get_forced_device
    except Exception as e:
        print(f"❌ 训练器导入失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    try:
        # 加载配置
        config = load_config_isaac(args.config)

        # 禁用命令行参数覆盖，完全使用config文件设置
        # 注释掉所有覆盖逻辑，确保config文件优先级最高
        # if args.num_envs is not None:
        #     config['env']['num_envs'] = args.num_envs
        # if args.device is not None:
        #     config['env']['device_id'] = args.device
        # if args.episodes is not None:
        #     config['train']['num_episodes'] = args.episodes
        # if args.render:
        #     config['simulator']['enable_rendering'] = True

        print("✅ 完全使用config文件设置，忽略命令行参数覆盖")

        # 创建保存目录
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置副本
        import yaml
        config_path = save_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"💾 配置已保存到: {config_path}")

        print(f"\n🎯 开始训练...")
        print(f"   配置: {args.config}")
        print(f"   环境: {config['env']['num_envs']}x并行")
        print(f"   设备: GPU {config['env']['device_id']}")
        print(f"   回合: {config['train']['num_episodes']}")
        print(f"   渲染: {'启用' if args.render else '禁用'}")

        # 🎯 [SERVER FIX] 获取强制设备并创建环境
        forced_device = get_forced_device()
        # **用户服务器使用GPU 2，但设置CUDA_VISIBLE_DEVICES=2后，GPU 2变为cuda:0**
        device_id = 2  # 直接使用GPU 2

        print(f"🏗️ 创建Isaac Gym环境...")
        print(f"   🔒 [FORCED] 使用设备: {forced_device} (原GPU 2, device_id: {device_id})")
        env = UR10ePPOEnvIsaac(
            config_path=args.config,
            num_envs=config['env']['num_envs'],
            device_id=device_id
        )

        print("✅ 环境创建成功")
        print(f"   环境数量: {env.get_num_envs()}")
        print(f"   状态维度: {env.get_num_obs()}")
        print(f"   动作维度: {env.get_num_actions()}")

        # 创建PPO训练器
        print(f"🤖 创建PPO训练器...")
        ppo = PPOIsaac(env, config)

        # 恢复训练（如果指定）
        if args.resume:
            print(f"🔄 恢复训练: {args.resume}")
            ppo.load_model(args.resume)

        # 开始真正的PPO训练
        print(f"\n🎯 开始真正的PPO训练...")
        print(f"   训练回合数: {config['train']['num_episodes']}")
        print(f"   保存目录: {save_dir}")

        # 调用PPO训练器进行训练
        ppo.train(
            num_episodes=int(config['train']['num_episodes']),
            save_dir=str(save_dir)
        )

        # 关闭环境
        env.close()

        return True

    except Exception as e:
        print(f"❌ 训练过程中发生错误: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return False

    print("👋 程序结束")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
"""
Utility Functions for Stable-Baselines3 UR10e Training

Adapted from the original train_isaac_fixed.py utilities to support
stable-baselines3 implementation.
"""

# IMPORTANT: Isaac Gym must be imported before PyTorch
import os
import sys
import signal
import time
import argparse
import numpy as np
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../ppo_ur10e_stab3'))

# Isaac Gym imports MUST be before PyTorch
try:
    from isaacgym import gymapi
    from isaacgym import gymutil
    from isaacgym import gymtorch
    from isaacgym.torch_utils import *
    print("✅ Isaac Gym imported successfully in utils_stab3")
except ImportError as e:
    print(f"⚠️ Isaac Gym import failed in utils_stab3: {e}")
    print("   Some functions may not be available")

# Now import PyTorch after Isaac Gym
import torch


# Graceful exit handler - 优雅退出处理器
class GracefulExiter:
    def __init__(self):
        self.shutdown = False

    def __call__(self, signum, frame):
        print(f"\n🛑 接收到退出信号 {signum}，正在优雅退出...")
        self.shutdown = True


# Set global exit handler
exiter = GracefulExiter()
signal.signal(signal.SIGINT, exiter)
signal.signal(signal.SIGTERM, exiter)


def load_config_stab3(config_path: str = "config_stab3.yaml") -> Dict[str, Any]:
    """
    Load stable-baselines3 configuration

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    # Adjust config path to be relative to this directory
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ 配置文件加载成功: {config_path}")
        return config
    except FileNotFoundError:
        print(f"⚠️ 配置文件 {config_path} 未找到，使用默认配置")
        return get_default_config_stab3()
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        print("   使用默认配置")
        return get_default_config_stab3()


def get_default_config_stab3() -> Dict[str, Any]:
    """
    Get default stable-baselines3 configuration

    Returns:
        Default configuration dictionary
    """
    return {
        'env': {
            'max_steps': 1000,
            'dt': 0.01,
            'device_id': 0,
            'num_envs': 1
        },
        'control': {
            'max_increment_torque': 40.0,
            'torque_safety_factor': 0.8
        },
        'ppo': {
            'policy': "MlpPolicy",
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5
        },
        'reward': {
            'distance_weight': 2.0,
            'success_reward': 10.0,
            'success_threshold': 0.05
        },
        'target': {
            'range': {
                'x': [-0.6, 0.6],
                'y': [-0.6, 0.6],
                'z': [0.1, 0.8]
            }
        }
    }


def check_environment() -> bool:
    """
    Check Isaac Gym environment

    Returns:
        True if environment is ready, False otherwise
    """
    print("🔍 检查Isaac Gym环境...")

    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，Isaac Gym需要GPU支持")
        return False

    print(f"✅ CUDA可用")
    print(f"   GPU数量: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")

    # Check Isaac Gym
    try:
        gym = gymapi.acquire_gym()
        print("✅ Isaac Gym基础连接成功")
        # Note: Newer Isaac Gym versions don't need release_gym
    except Exception as e:
        print(f"❌ Isaac Gym不可用: {e}")
        return False

    # Check stable-baselines3
    try:
        import stable_baselines3
        print(f"✅ Stable-Baselines3 可用 (版本: {stable_baselines3.__version__})")
    except ImportError:
        print("❌ Stable-Baselines3 未安装")
        return False

    return True


def test_basic_isaac_gym() -> bool:
    """
    Test basic Isaac Gym functionality

    Returns:
        True if test passes, False otherwise
    """
    print(f"\n🧪 测试基本Isaac Gym功能...")

    try:
        # Create simulator
        sim_params = gymapi.SimParams()
        sim_params.dt = 0.01
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.physx.solver_type = 1  # Use more stable solver
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.use_gpu = True
        # Remove unsupported gpu_pipeline attribute

        gym = gymapi.acquire_gym()
        sim_instance = gym.create_sim(compute_device=0, graphics_device=0, params=sim_params)

        if sim_instance is None:
            print("❌ 仿真器创建失败")
            return False

        print("✅ 基本仿真器创建成功")

        # Cleanup
        gym.destroy_sim(sim_instance)

        return True

    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        return False


def test_stable_baselines3_components() -> bool:
    """
    Test stable-baselines3 components

    Returns:
        True if components are working, False otherwise
    """
    print(f"\n🤖 测试stable-baselines3组件...")

    try:
        # Test stable-baselines3 import
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.callbacks import BaseCallback

        print("✅ Stable-Baselines3模块导入成功")

        # Test our environment
        from ur10e_env_stab3 import UR10eEnvStab3

        # Create simple environment for testing
        print("   测试环境创建...")
        test_env = UR10eEnvStab3(config_path="config_stab3.yaml", num_envs=1)

        # Check environment compatibility
        check_env(test_env, warn=True)
        print("✅ 环境与stable-baselines3兼容")

        # Test environment reset and step
        obs, info = test_env.reset()
        action = test_env.action_space.sample()
        obs, reward, terminated, truncated, info = test_env.step(action)

        print(f"✅ 环境测试成功")
        print(f"   观察空间: {test_env.observation_space}")
        print(f"   动作空间: {test_env.action_space}")

        # Cleanup
        test_env.close()

        return True

    except Exception as e:
        print(f"❌ Stable-Baselines3组件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_forced_device() -> str:
    """
    Get forced device configuration

    Returns:
        Device string (e.g., "cuda:0")
    """
    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("⚠️ CUDA不可用，使用CPU")
        return "cpu"

    # Get GPU count and select device
    gpu_count = torch.cuda.device_count()

    # Try to use specified device from environment
    device_id = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])

    if device_id >= gpu_count:
        print(f"⚠️ 请求的GPU {device_id} 不存在，使用GPU 0")
        device_id = 0

    device = f"cuda:{device_id}"
    print(f"🎯 强制使用设备: {device} (GPU {device_id}/{gpu_count})")

    return device


def setup_training_directories(save_dir: str) -> Path:
    """
    Setup training directories

    Args:
        save_dir: Base save directory

    Returns:
        Path to save directory
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (save_path / "models").mkdir(exist_ok=True)
    (save_path / "logs").mkdir(exist_ok=True)
    (save_path / "evaluations").mkdir(exist_ok=True)

    print(f"💾 训练目录已创建: {save_path}")
    return save_path


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments
    与原始 train_isaac_fixed.py 保持一致
    """
    parser = argparse.ArgumentParser(description="UR10e Stable-Baselines3 PPO训练")

    # 保持与原始脚本完全一致的参数
    parser.add_argument("--config", "-c", type=str, default="config_stab3.yaml",
                       help="配置文件路径")
    parser.add_argument("--num-envs", "-n", type=int, default=None,
                       help="并行环境数量 (已禁用，请使用config文件)")
    parser.add_argument("--device", "-d", type=int, default=0,
                       help="GPU设备ID (已禁用，请使用config文件)")
    parser.add_argument("--episodes", "-e", type=int, default=None,
                       help="训练回合数 (已禁用，请使用config文件)")
    parser.add_argument("--save-dir", "-s", type=str, default="./checkpoints_stab3",
                       help="模型保存目录")
    parser.add_argument("--resume", "-r", type=str, default=None,
                       help="恢复训练的检查点路径")
    parser.add_argument("--render", action="store_true",
                       help="启用渲染（降低训练速度）")
    parser.add_argument("--debug", action="store_true",
                       help="启用调试模式")
    parser.add_argument("--test", action="store_true",
                       help="仅测试环境，不进行训练")

    return parser.parse_args()


def print_system_info():
    """Print system information"""
    print("\n🖥️ 系统信息:")
    print(f"   Python: {sys.version}")
    print(f"   PyTorch: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"   CUDA: {torch.version.cuda}")
        print(f"   GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print(f"   CUDA: 不可用")

    try:
        import stable_baselines3
        print(f"   Stable-Baselines3: {stable_baselines3.__version__}")
    except ImportError:
        print(f"   Stable-Baselines3: 未安装")

    try:
        from isaacgym import gymapi
        print(f"   Isaac Gym: 可用")
    except ImportError:
        print(f"   Isaac Gym: 不可用")


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration

    Args:
        config: Configuration dictionary

    Returns:
        True if config is valid, False otherwise
    """
    required_sections = ['env', 'ppo', 'reward', 'target']

    for section in required_sections:
        if section not in config:
            print(f"❌ 配置缺少必需部分: {section}")
            return False

    # Validate environment section
    env_config = config.get('env', {})
    if env_config.get('max_steps', 0) <= 0:
        print("❌ env.max_steps 必须大于0")
        return False

    # Validate PPO section
    ppo_config = config.get('ppo', {})
    if ppo_config.get('learning_rate', 0) <= 0:
        print("❌ ppo.learning_rate 必须大于0")
        return False

    print("✅ 配置验证通过")
    return True


# Training callback for progress monitoring
class TrainingProgressCallback:
    """
    Simple training progress callback
    """

    def __init__(self, eval_freq: int = 10000, verbose: int = 1):
        self.eval_freq = eval_freq
        self.verbose = verbose
        self.n_calls = 0
        self.start_time = time.time()

    def __call__(self, locals_, globals_):
        self.n_calls += 1

        if self.verbose > 0 and self.n_calls % 100 == 0:
            elapsed_time = time.time() - self.start_time
            print(f"\n📈 训练进度更新 (调用 {self.n_calls}):")
            print(f"   已用时间: {elapsed_time/60:.1f}分钟")

            # Try to get current reward if available
            if 'ep_info_buffer' in locals_ and len(locals_['ep_info_buffer']) > 0:
                recent_rewards = [ep_info['r'] for ep_info in locals_['ep_info_buffer'][-10:]]
                mean_reward = np.mean(recent_rewards)
                print(f"   最近平均奖励: {mean_reward:.4f}")

        # Check for graceful exit
        if exiter.shutdown:
            print("\n🛑 收到退出信号，正在停止训练...")
            return False  # Stop training

        return True  # Continue training


if __name__ == "__main__":
    # Test utility functions
    print("🧪 测试工具函数...")

    # Test config loading
    config = load_config_stab3()
    print(f"✅ 配置加载测试通过")

    # Test system info
    print_system_info()

    # Test environment check
    if check_environment():
        print("✅ 环境检查通过")
    else:
        print("❌ 环境检查失败")

    # Test stable-baselines3 components
    if test_stable_baselines3_components():
        print("✅ Stable-Baselines3组件测试通过")
    else:
        print("❌ Stable-Baselines3组件测试失败")

    print("\n✅ 所有工具函数测试完成")
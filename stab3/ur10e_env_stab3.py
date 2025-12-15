"""
UR10e Environment with Stable-Baselines3 Integration

Adapted from ur10e_incremental_env.py for use in ppo_ur10e_gym/stab3 directory.
This file serves as an adapter to integrate the stable-baselines3 implementation
with the existing ppo_ur10e_gym structure.
"""

# IMPORTANT: Isaac Gym must be imported before PyTorch
import sys
import os
import math
from typing import Dict, Any, Optional, Tuple
from collections import deque
import yaml

# Add parent directory to path to import from ppo_ur10e_stab3
sys.path.append(os.path.join(os.path.dirname(__file__), '../../ppo_ur10e_stab3'))

# Isaac Gym imports MUST be before PyTorch
try:
    from isaacgym import gymapi
    from isaacgym import gymtorch
    from isaacgym import gymutil
    from isaacgym.torch_utils import *
    print("✅ Isaac Gym imported successfully in ur10e_env_stab3")
except ImportError as e:
    print(f"❌ Failed to import Isaac Gym in ur10e_env_stab3: {e}")
    sys.exit(1)

# Now import PyTorch after Isaac Gym
import torch

# Import gymnasium after PyTorch
import gymnasium as gym
from gymnasium import spaces

# Import the original implementation as base
try:
    from ur10e_incremental_env import UR10eIncrementalEnv as BaseUR10eEnv
    print("✅ Base UR10eIncrementalEnv imported successfully")
except ImportError:
    print("⚠️ Base implementation not found, using fallback")
    BaseUR10eEnv = None


class UR10eEnvStab3(gym.Env):
    """
    UR10e Environment Adapter for Stable-Baselines3

    This class adapts the ur10e_incremental_env implementation for use
    in the ppo_ur10e_gym directory structure, maintaining compatibility
    with stable-baselines3 while preserving all functionality.

    Key Features:
    - Action Space: 6D continuous (Δτ₁, Δτ₂, ..., Δτ₆)
    - State Space: 19D observation space
    - Direct torque control with safety limits
    - Isaac Gym physics simulation
    - Stable-Baselines3 compatible interface
    """

    def __init__(self, config_path: str = "config_stab3.yaml", num_envs: int = 1, device_id: int = 0):
        """
        Initialize the environment

        Args:
            config_path: Path to configuration file
            num_envs: Number of parallel environments
            device_id: GPU device ID
        """
        super().__init__()

        # Store parameters
        self.config_path = config_path
        self.num_envs = num_envs
        self.device_id = device_id

        # Adjust config path to be relative to this directory
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.path.dirname(__file__), config_path)

        # Check if base implementation is available
        if BaseUR10eEnv is not None:
            # Use the base implementation
            print("🔧 Using base UR10eIncrementalEnv implementation")
            self._env = BaseUR10eEnv(config_path, num_envs)

            # Expose the base environment's attributes
            self._setup_attributes_from_base()
        else:
            # Fallback implementation
            print("⚠️ Using fallback implementation")
            self._init_fallback_implementation(config_path, num_envs, device_id)

        print(f"✅ UR10eEnvStab3 initialized successfully")
        print(f"   Config: {config_path}")
        print(f"   Environments: {num_envs}")
        print(f"   Device: {device_id}")

    def _setup_attributes_from_base(self):
        """Setup attributes from the base environment"""
        # Copy essential attributes
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self.max_steps = self._env.max_steps
        self.current_step = getattr(self._env, 'current_step', 0)
        self.config = getattr(self._env, 'config', {})

        # 确保观察空间是float32
        if hasattr(self.observation_space, 'dtype'):
            self.observation_space.dtype = np.float32
        else:
            # 如果观察空间没有dtype属性，创建一个新的
            self.observation_space = spaces.Box(
                low=self.observation_space.low,
                high=self.observation_space.high,
                shape=self.observation_space.shape,
                dtype=np.float32
            )

        # 确保动作空间也是float32
        if hasattr(self.action_space, 'dtype'):
            self.action_space.dtype = np.float32
        else:
            self.action_space = spaces.Box(
                low=self.action_space.low,
                high=self.action_space.high,
                shape=self.action_space.shape,
                dtype=np.float32
            )

    def _init_fallback_implementation(self, config_path: str, num_envs: int, device_id: int):
        """Fallback implementation when base is not available"""
        # Load configuration
        self.config = self._load_config(config_path)
        self.max_steps = self.config.get('env', {}).get('max_steps', 1000)
        self.current_step = 0

        # Device configuration
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

        # Define action and observation spaces (simplified version)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),  # 6D torque control
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(19,),  # 19D observation space
            dtype=np.float32
        )

        # Initialize state
        self._state = np.zeros(19, dtype=np.float32)

        print(f"⚠️ Fallback mode - Limited functionality")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"⚠️ Config file {config_path} not found, using defaults")
            config = self._get_default_config()
        return config

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'env': {
                'max_steps': 1000,
                'dt': 0.01,
                'device_id': 0
            },
            'control': {
                'max_increment_torque': 40.0,
                'torque_safety_factor': 0.8
            },
            'reward': {
                'distance_weight': 2.0,
                'success_reward': 10.0,
                'success_threshold': 0.05
            }
        }

    # Fallback methods (only used if base implementation is not available)
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        if hasattr(self, '_env') and self._env is not None:
            obs, info = self._env.reset(seed, options)
            # 确保观察值是float32类型
            if hasattr(obs, 'astype'):
                obs = obs.astype(np.float32)
            return obs, info

        # Fallback implementation
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self._state = np.random.uniform(-0.5, 0.5, 19).astype(np.float32)

        info = {
            'episode': {'r': 0.0, 'l': 0},
            'distance': 1.0,
            'success': False
        }

        return self._state, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step the environment"""
        if hasattr(self, '_env') and self._env is not None:
            return self._env.step(action)

        # Fallback implementation
        self.current_step += 1

        # Simple state update (placeholder)
        self._state = np.clip(
            self._state + action * 0.01 + np.random.normal(0, 0.001, 19),
            self.observation_space.low,
            self.observation_space.high
        ).astype(np.float32)

        # Calculate dummy reward
        distance = np.linalg.norm(self._state[:3])
        reward = -distance

        # Check termination
        terminated = distance < 0.1 or self.current_step >= self.max_steps
        truncated = False

        info = {
            'episode': {'r': reward, 'l': self.current_step},
            'distance': distance,
            'success': distance < 0.1
        }

        return self._state, reward, terminated, truncated, info

    def render(self):
        """Render the environment"""
        if hasattr(self, '_env') and self._env is not None:
            return self._env.render()
        pass  # No rendering in fallback mode

    def close(self):
        """Close the environment"""
        if hasattr(self, '_env') and self._env is not None:
            return self._env.close()
        print("✅ UR10eEnvStab3 closed")

    def seed(self, seed=None):
        """Set random seed"""
        if hasattr(self, '_env') and self._env is not None:
            return self._env.seed(seed)

        if seed is not None:
            np.random.seed(seed)
        return [seed]

    def get_success_rate(self) -> float:
        """Get recent success rate"""
        if hasattr(self, '_env') and hasattr(self._env, 'get_success_rate'):
            return self._env.get_success_rate()
        return 0.0

    def print_performance_stats(self):
        """Print performance statistics"""
        if hasattr(self, '_env') and hasattr(self._env, 'print_performance_stats'):
            self._env.print_performance_stats()
        else:
            print(f"📊 Fallback Mode - Step: {self.current_step}/{self.max_steps}")


# Factory function for easier creation
def make_ur10e_env_stab3(config_path: str = "config_stab3.yaml",
                        num_envs: int = 1,
                        device_id: int = 0,
                        render: bool = False) -> UR10eEnvStab3:
    """
    Factory function to create UR10e environment for stable-baselines3

    Args:
        config_path: Path to configuration file
        num_envs: Number of parallel environments
        device_id: GPU device ID
        render: Whether to enable rendering

    Returns:
        UR10eEnvStab3 instance
    """
    env = UR10eEnvStab3(config_path, num_envs, device_id)

    # Configure rendering if requested
    if render and hasattr(env, '_env'):
        env._env.enable_rendering = True

    return env


if __name__ == "__main__":
    # Test the environment
    env = UR10eEnvStab3()

    print("\n🧪 Testing UR10eEnvStab3...")

    # Test reset
    obs, info = env.reset()
    print(f"✅ Reset successful. Obs shape: {obs.shape}")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")

    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✅ Step successful. Reward: {reward:.4f}")

    # Test multiple steps
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 5 == 0:
            print(f"   Step {step}: Reward={reward:.4f}")

        if terminated:
            print(f"   🎯 Episode completed at step {step}!")
            break

    env.close()
    print("✅ Test completed successfully")
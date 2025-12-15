"""
PPO UR10e Stable-Baselines3 Implementation

This package provides a stable-baselines3 implementation for UR10e robot control,
replacing the custom Isaac Gym PPO implementation with the standard RL framework.
"""

__version__ = "1.0.0"
__author__ = "UR10e RL Team"

from .ur10e_env_stab3 import UR10eEnvStab3, make_ur10e_env_stab3
from .utils_stab3 import (
    check_environment, test_basic_isaac_gym, load_config_stab3,
    get_forced_device, setup_training_directories
)

__all__ = [
    "UR10eEnvStab3",
    "make_ur10e_env_stab3",
    "check_environment",
    "test_basic_isaac_gym",
    "load_config_stab3",
    "get_forced_device",
    "setup_training_directories"
]
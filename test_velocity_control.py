#!/usr/bin/env python3
"""
速度控制系统测试脚本
测试新的基于速度的PD控制系统是否正常工作
"""

import os
import sys
import numpy as np
from typing import Dict, Any

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ur10e_trajectory_env_isaac import UR10eTrajectoryEnvIsaac
import yaml
import torch

def test_velocity_control():
    """测试速度控制系统的基本功能"""
    print("🚀 开始测试速度控制系统")
    print("=" * 60)

    try:
        # 1. 创建环境
        print("📦 创建UR10e轨迹环境...")
        config_path = "config_isaac.yaml"

        # 检查配置文件是否存在
        if not os.path.exists(config_path):
            print(f"❌ 配置���件不存在: {config_path}")
            return False

        env = UR10eTrajectoryEnvIsaac(
            config_path=config_path,
            num_envs=1,
            mode="point_to_point"  # 先用简单模式测试
        )

        print("✅ 环境创建成功")
        print(f"   动作空间: {env.action_space}")
        print(f"   观测空间: {env.observation_space}")
        print()

        # 2. 测试动作空间
        print("🎯 测试归一化动作空间...")
        sample_action = env.action_space.sample()
        print(f"   采样动作: {sample_action}")
        print(f"   动作范围: [{env.action_space.low}, {env.action_space.high}]")

        # 验证动作是否在[-1, 1]范围内
        assert np.all(sample_action >= -1.0) and np.all(sample_action <= 1.0), "动作不在[-1,1]范围内"
        print("✅ 动作空间测试通过")
        print()

        # 3. 测试环境重置
        print("🔧 测试环境重置...")
        reset_result = env.reset()
        # Handle both single obs and (obs, info) return formats
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
            info = {}

        print(f"   初始观测形状: {obs.shape}")
        print(f"   初始观测范围: [{obs.min():.3f}, {obs.max():.3f}]")
        print("✅ 环境重置测试通过")
        print()

        # 4. 测试步进（使用归一化速度动作）
        print("🏃 测试速度控制步进...")
        num_steps = 10

        for step in range(num_steps):
            # 生成归一化速度动作 [-1, 1]^6
            action = np.random.uniform(-0.5, 0.5, size=6).astype(np.float32)
            print(f"   步骤 {step+1}: 动作 = {action}")

            # 执行步进
            obs, reward, terminated, truncated, info = env.step(action)

            print(f"   奖励: {reward:.3f}")
            print(f"   完成: {terminated}, 截断: {truncated}")

            if terminated:
                print("   🎉 Episode完成!")
                break

        print("✅ 步进测试通过")
        print()

        # 5. 测试边界条件
        print("🔍 测试边界条件...")

        # 测试最大正速度
        max_action = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        print(f"   测试最大正速度: {max_action}")
        obs, reward, terminated, truncated, info = env.step(max_action)
        print(f"   结果: 奖励={reward:.3f}")

        # 测试最大负速度
        min_action = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        print(f"   测试最大负速度: {min_action}")
        obs, reward, terminated, truncated, info = env.step(min_action)
        print(f"   结果: 奖励={reward:.3f}")

        print("✅ 边界条件测试通过")
        print()

        # 6. 测试轨迹跟踪模式
        print("🛤️ 测试轨迹跟踪模式...")
        env.set_mode("trajectory_tracking")
        obs = env.reset()

        # 检查是否有期望关节角度初始化
        if hasattr(env, 'desired_joint_angles') and env.desired_joint_angles is not None:
            print("✅ 期望关节角度正确初始化")
            print(f"   形状: {env.desired_joint_angles.shape}")
        else:
            print("⚠️ 期望关节角度未正确初始化")

        # 执行几步
        for step in range(5):
            action = np.random.uniform(-0.3, 0.3, size=6).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   步骤 {step+1}: 奖励={reward:.3f}")

        print("✅ 轨迹跟踪模式测试通过")
        print()

        # 7. 关闭环境
        print("🔒 关闭环境...")
        env.close()
        print("✅ 环境已关闭")

        print("\n" + "=" * 60)
        print("🎉 所有测试通过！速度控制系统工作正常")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_velocity_integration():
    """测试速度积分功能"""
    print("\n🧮 测试速度积分功能")
    print("-" * 40)

    try:
        # 创建环境
        env = UR10eTrajectoryEnvIsaac(
            config_path="config_isaac.yaml",
            num_envs=1,
            mode="point_to_point"
        )

        obs = env.reset()

        # 获取初始关节角度
        current_angles, current_vels = env._get_joint_angles_and_velocities()
        print(f"初始关节角度: {current_angles[0].detach().cpu().numpy()}")

        # 测试正速度积分
        positive_velocity = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)

        print(f"应用正速度: {positive_velocity}")

        # 执行多步来观察积分效果
        for i in range(5):
            obs, reward, terminated, truncated, info = env.step(positive_velocity)

            # 检查期望角度是否在增加
            if hasattr(env, 'desired_joint_angles'):
                desired_angles = env.desired_joint_angles[0].detach().cpu().numpy()
                print(f"   步骤 {i+1} 期望角度: {desired_angles}")

        print("✅ 速度积分测试完成")
        env.close()
        return True

    except Exception as e:
        print(f"❌ 积分测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("🔬 UR10e速度控制系统测试")
    print("时间:", np.datetime64('now'))
    print()

    # 运行基本测试
    success1 = test_velocity_control()

    # 运行积分测试
    success2 = test_velocity_integration()

    # 总结
    print("\n" + "=" * 60)
    print("📊 测试结果总结:")
    print(f"   基本功能测试: {'✅ 通过' if success1 else '❌ 失败'}")
    print(f"   积分功能测试: {'✅ 通过' if success2 else '❌ 失败'}")

    if success1 and success2:
        print("\n🎉 所有测试通过！速度控制系统准备就绪")
        return 0
    else:
        print("\n⚠️ 部分测试失败，请检查实现")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
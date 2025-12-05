#!/usr/bin/env python3
"""
简单的速度控制系统测试
"""

import os
import sys
import numpy as np

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试导入是否正常"""
    print("🔍 测试模块导入...")
    try:
        from ur10e_trajectory_env_isaac import UR10eTrajectoryEnvIsaac
        print("✅ UR10eTrajectoryEnvIsaac 导入成功")

        from ppo_isaac import PPOIsaac, ActorNetwork, CriticNetwork
        print("✅ PPO模块导入成功")

        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_action_space():
    """测试动作空间定义"""
    print("\n🎯 测试动作空间...")
    try:
        from ur10e_trajectory_env_isaac import UR10eTrajectoryEnvIsaac

        # 创建环境（避免图形界面）
        print("   创建环境...")
        env = UR10eTrajectoryEnvIsaac(
            config_path="config_isaac.yaml",
            num_envs=1,
            mode="point_to_point"
        )

        # 检查动作空间
        print(f"   动作空间: {env.action_space}")
        print(f"   动作维度: {env.action_dim}")
        print(f"   动作范围: [{env.action_space.low}, {env.action_space.high}]")

        # 验证是否为归一化速度
        expected_low = np.array([-1.0] * 6)
        expected_high = np.array([1.0] * 6)

        if np.allclose(env.action_space.low, expected_low) and np.allclose(env.action_space.high, expected_high):
            print("✅ 动作空间正确设置为归一化速度[-1,1]^6")
        else:
            print("❌ 动作空间设置错误")
            return False

        # 测试采样
        action = env.action_space.sample()
        print(f"   采样动作: {action}")

        if np.all(action >= -1.0) and np.all(action <= 1.0):
            print("✅ 动作采样正确")
        else:
            print("❌ 动作采样超出范围")
            return False

        # 关闭环境
        env.close()
        return True

    except Exception as e:
        print(f"❌ 动作空间测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_network_output():
    """测试网络输出范围"""
    print("\n🤖 测试PPO网络...")
    try:
        from ppo_isaac import ActorNetwork
        import torch

        # 创建Actor网络
        actor = ActorNetwork(state_dim=19, action_dim=6, hidden_dim=64)
        print("✅ Actor网络创建成功")

        # 创建测试输入
        batch_size = 4
        test_state = torch.randn(batch_size, 19)
        print(f"   测试输入形状: {test_state.shape}")

        # 获取动作输出
        action, log_prob = actor.sample(test_state)
        print(f"   动作输出形状: {action.shape}")
        print(f"   动作范围: [{action.min().item():.3f}, {action.max().item():.3f}]")

        # 验证动作范围
        if torch.all(action >= -1.0) and torch.all(action <= 1.0):
            print("✅ Actor网络输出正确的归一化速度")
        else:
            print("❌ Actor网络输出超出范围")
            return False

        return True

    except Exception as e:
        print(f"❌ 网络测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🧪 速度控制系统简单测试")
    print("=" * 50)

    # 运行测试
    success1 = test_imports()
    success2 = test_action_space()
    success3 = test_network_output()

    # 总结
    print("\n" + "=" * 50)
    print("📊 测试结果:")
    print(f"   模块导入: {'✅' if success1 else '❌'}")
    print(f"   动作空间: {'✅' if success2 else '❌'}")
    print(f"   网络输出: {'✅' if success3 else '❌'}")

    if success1 and success2 and success3:
        print("\n🎉 基本测试通过！速度控制系统配置正确")
        return 0
    else:
        print("\n⚠️ 部分测试失败，请检查配置")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
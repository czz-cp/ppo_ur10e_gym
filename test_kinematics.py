#!/usr/bin/env python3
"""
测试UR10e运动学计算
"""

import os
import sys
import numpy as np
from ur10e_kinematics_fixed import UR10eKinematicsFixed
from ur10e_env_isaac import UR10ePPOEnvIsaac
import torch

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ur10e_kinematics():
    """测试UR10e运动学计算"""
    print("🧪 测试UR10e运动学计算")
    print("=" * 50)

    try:
        

        # 创建运动学对象
        kinematics = UR10eKinematicsFixed()
        print("✅ UR10eKinematics 创建成功")

        # 测试一些关节角度
        test_angles = [
            [0, 0, 0, 0, 0, 0],          # 零位
            [0, np.pi/2, -np.pi/2, 0, 0, 0],  # 典型位置
            [0, 0.8, 0.5, 0, 0, 0],      # 我们使用的大致范围
        ]

        for i, angles in enumerate(test_angles):
            print(f"\n🔧 测试 {i+1}: 关节角度 {angles}")

            # 正运动学计算
            T = kinematics.forward_kinematics(np.array(angles))
            tcp_pos = T[:3, 3]

            print(f"   TCP位置: [{tcp_pos[0]:.3f}, {tcp_pos[1]:.3f}, {tcp_pos[2]:.3f}]")

            # 检查是否在工作空间内
            workspace_bounds = {
                'x': [-1.2, 1.2],
                'y': [-1.2, 1.2],
                'z': [-1.2, 1.5]  # 临时扩大Z轴范围
            }

            in_workspace = (
                workspace_bounds['x'][0] <= tcp_pos[0] <= workspace_bounds['x'][1] and
                workspace_bounds['y'][0] <= tcp_pos[1] <= workspace_bounds['y'][1] and
                workspace_bounds['z'][0] <= tcp_pos[2] <= workspace_bounds['z'][1]
            )

            print(f"   在工作空间内: {'✅' if in_workspace else '❌'}")

        # 测试环境中的运动学计算
        print(f"\n🏗️ 测试环境中的运动学...")
        try:
            env = UR10ePPOEnvIsaac(config_path="config_isaac.yaml", num_envs=1)

            # 获取当前关节角度
            current_angles, current_vels = env._get_joint_angles_and_velocities()
            angles_np = current_angles[0].detach().cpu().numpy()
            print(f"   当前关节角度: {angles_np}")

            # 使用环境方法计算TCP
            tcp_positions = env._compute_end_effector_positions_batch(current_angles)
            tcp_env = tcp_positions[0].detach().cpu().numpy()
            print(f"   环境TCP位置: [{tcp_env[0]:.3f}, {tcp_env[1]:.3f}, {tcp_env[2]:.3f}]")

            # 使用独立运动学验证
            T_kinematics = kinematics.forward_kinematics(angles_np)
            tcp_kinematics = T_kinematics[:3, 3]
            print(f"   运动学TCP位置: [{tcp_kinematics[0]:.3f}, {tcp_kinematics[1]:.3f}, {tcp_kinematics[2]:.3f}]")

            # 比较差异
            diff = np.linalg.norm(tcp_env - tcp_kinematics)
            print(f"   位置差异: {diff:.6f} m")

            env.close()

        except Exception as e:
            print(f"   ❌ 环境测试失败: {e}")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ur10e_kinematics()
    if success:
        print("\n🎉 运动学测试完成")
    else:
        print("\n⚠️ 运动学测试失败")
    sys.exit(0 if success else 1)
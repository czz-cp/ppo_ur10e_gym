#!/usr/bin/env python3
"""
测试工作空间修复
"""

import os
import sys
import numpy as np

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_workspace_fix():
    """测试工作空间修复是否有效"""
    print("🧪 测试工作空间修复")
    print("=" * 50)

    try:
        from ur10e_trajectory_env_isaac import UR10eTrajectoryEnvIsaac

        # 读取配置文件获取 num_envs
        with open("config_isaac.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 创建环境
        print("📦 创建环境...")
        env = UR10eTrajectoryEnvIsaac(
            config_path="config_isaac.yaml",
            num_envs=config['env']['num_envs'],  # 使用配置文件中的值
            mode="trajectory_tracking"
        )

        # 测试多次重置
        for test in range(5):
            print(f"\n🔄 测试 {test + 1}/5:")

            # 重置环境
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result

            # 获取当前TCP位置
            current_angles, current_vels = env._get_joint_angles_and_velocities()
            tcp_pos = env._compute_end_effector_positions_batch(current_angles)[0]

            tcp_pos_np = tcp_pos.detach().cpu().numpy()
            print(f"   TCP位置: [{tcp_pos_np[0]:.3f}, {tcp_pos_np[1]:.3f}, {tcp_pos_np[2]:.3f}]")

            # 检查是否在工作空间内
            workspace_bounds = {
                'x': [-1.2, 1.2],
                'y': [-1.2, 1.2],
                'z': [0.0, 1.5]
            }

            in_workspace = (
                workspace_bounds['x'][0] <= tcp_pos_np[0] <= workspace_bounds['x'][1] and
                workspace_bounds['y'][0] <= tcp_pos_np[1] <= workspace_bounds['y'][1] and
                workspace_bounds['z'][0] <= tcp_pos_np[2] <= workspace_bounds['z'][1]
            )

            if in_workspace:
                print("   ✅ TCP位置在工作空间内")
            else:
                print("   ❌ TCP位置超出工作空间")
                print(f"      X范围: [{workspace_bounds['x'][0]}, {workspace_bounds['x'][1]}], 实际: {tcp_pos_np[0]:.3f}")
                print(f"      Y范围: [{workspace_bounds['y'][0]}, {workspace_bounds['y'][1]}], 实际: {tcp_pos_np[1]:.3f}")
                print(f"      Z范围: [{workspace_bounds['z'][0]}, {workspace_bounds['z'][1]}], 实际: {tcp_pos_np[2]:.3f}")

        # 尝试轨迹规划
        print(f"\n🛤️ 测试轨迹规划...")
        start_tcp = tcp_pos_np
        goal_tcp = np.array([0.5, 0.0, 0.8], dtype=np.float32)

        print(f"   起始TCP: {start_tcp}")
        print(f"   目标TCP: {goal_tcp}")

        success = env.plan_trajectory(start_tcp, goal_tcp)

        if success:
            print("   ✅ 轨迹规划成功")
        else:
            print("   ❌ 轨迹规划失败")

        env.close()
        return success

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_workspace_fix()
    if success:
        print("\n🎉 工作空间修复成功！")
    else:
        print("\n⚠️ 工作空间还有问题")
    sys.exit(0 if success else 1)
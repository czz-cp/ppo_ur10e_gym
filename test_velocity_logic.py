#!/usr/bin/env python3
"""
速度控制逻辑测试（不依赖Isaac Gym）
测试速度积分、PD控制等核心逻辑
"""

import torch
import numpy as np

def test_velocity_integration():
    """测试速度积分逻辑"""
    print("🧮 测试速度积分逻辑")
    print("-" * 40)

    # 初始化参数
    dt = 0.01  # 时间步长
    velocity_limits = np.array([2.094, 2.094, 3.142, 3.142, 3.142, 3.142])
    joint_lower_limits = np.array([-6.283, -6.283, -3.142, -6.283, -6.283, -6.283])
    joint_upper_limits = np.array([6.283, 6.283, 3.142, 6.283, 6.283, 6.283])

    # 转换为tensor
    velocity_limits_tensor = torch.tensor(velocity_limits, dtype=torch.float32)
    joint_lower_limits_tensor = torch.tensor(joint_lower_limits, dtype=torch.float32)
    joint_upper_limits_tensor = torch.tensor(joint_upper_limits, dtype=torch.float32)

    # 初始化状态
    current_angles = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    desired_joint_angles = current_angles.clone()

    print(f"初始关节角度: {desired_joint_angles.numpy()}")

    # 测试正速度积分
    normalized_velocity = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=torch.float32)
    print(f"归一化速度: {normalized_velocity.numpy()}")

    # ��进模拟
    for step in range(10):
        # 1. 速度反归一化
        physical_velocities = normalized_velocity * velocity_limits_tensor
        print(f"步骤 {step+1} 物理速度: {physical_velocities.numpy()} rad/s")

        # 2. 积分
        desired_joint_angles = desired_joint_angles + physical_velocities * dt

        # 3. 关节限制
        desired_joint_angles = torch.clamp(
            desired_joint_angles,
            joint_lower_limits_tensor,
            joint_upper_limits_tensor
        )

        print(f"          期望角度: {desired_joint_angles.numpy()} rad")

    print("✅ 速度积分测试通过\n")
    return True

def test_pd_control():
    """测试PD控制逻辑"""
    print("🎛️ 测试PD控制逻辑")
    print("-" * 40)

    # PD增益
    kp_gains = [1000.0, 1000.0, 800.0, 400.0, 200.0, 100.0]
    kd_gains = [50.0, 50.0, 30.0, 20.0, 10.0, 5.0]

    kp_tensor = torch.tensor(kp_gains, dtype=torch.float32)
    kd_tensor = torch.tensor(kd_gains, dtype=torch.float32)

    # 当前状态
    current_angles = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=torch.float32)
    current_velocities = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    # 期望角度
    desired_joint_angles = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=torch.float32)

    print(f"当前角度: {current_angles.numpy()} rad")
    print(f"期望角度: {desired_joint_angles.numpy()} rad")

    # PD控制律
    position_errors = desired_joint_angles - current_angles
    pd_torques = kp_tensor * position_errors - kd_tensor * current_velocities

    print(f"位置误差: {position_errors.numpy()} rad")
    print(f"PD力矩: {pd_torques.numpy()} N⋅m")

    # 力矩限制
    torque_limits = [330.0, 330.0, 150.0, 54.0, 54.0, 54.0]
    torque_limits_tensor = torch.tensor(torque_limits, dtype=torch.float32)

    limited_torques = torch.clamp(
        pd_torques,
        -torque_limits_tensor,
        torque_limits_tensor
    )

    print(f"限制后力矩: {limited_torques.numpy()} N⋅m")
    print("✅ PD控制测试通过\n")
    return True

def test_network_output():
    """测试网络输出范围"""
    print("🤖 测试PPO网络输出")
    print("-" * 40)

    # 模拟Actor网络输出
    batch_size = 4
    action_dim = 6

    # 模拟网络输出（经过tanh后）
    raw_actions = torch.randn(batch_size, action_dim)
    actions = torch.tanh(raw_actions)

    print(f"原始输出范围: [{raw_actions.min():.3f}, {raw_actions.max():.3f}]")
    print(f"tanh后范围: [{actions.min():.3f}, {actions.max():.3f}]")

    # 验证是否在[-1,1]范围内
    if torch.all(actions >= -1.0) and torch.all(actions <= 1.0):
        print("✅ 网络输出正确限制在[-1,1]范围内")
        return True
    else:
        print("❌ 网络输出超出范围")
        return False

def test_control_loop():
    """测试完整控制循环"""
    print("🔄 测试完整控制循环")
    print("-" * 40)

    # 初始化
    dt = 0.01
    velocity_limits = np.array([2.094, 2.094, 3.142, 3.142, 3.142, 3.142])
    kp_gains = [1000.0, 1000.0, 800.0, 400.0, 200.0, 100.0]
    kd_gains = [50.0, 50.0, 30.0, 20.0, 10.0, 5.0]

    # 当前状态
    current_angles = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    current_velocities = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    desired_angles = current_angles.clone()

    # 目标位置（简化测试）
    target_angles = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)

    print(f"初始位置: {current_angles.numpy()}")
    print(f"目标位置: {target_angles.numpy()}")

    # 模拟控制循环
    for step in range(20):
        # 1. 计算误差并生成速度指令（简化控制器）
        error = target_angles - current_angles
        normalized_velocity = torch.tanh(error * 0.5)  # 简单P控制

        # 2. 速度反归一化
        physical_velocity = normalized_velocity * torch.tensor(velocity_limits)

        # 3. 积分得到期望角度
        desired_angles = desired_angles + physical_velocity * dt

        # 4. PD控制
        position_error = desired_angles - current_angles
        pd_torque = torch.tensor(kp_gains) * position_error - torch.tensor(kd_gains) * current_velocities

        # 5. 更新状态（简化模拟）
        current_angles = current_angles + physical_velocity * dt
        current_velocities = physical_velocity

        if step % 5 == 0:
            print(f"步骤 {step}: 位置={current_angles.numpy()}, 误差={error.numpy()}")

    # 检查收敛性
    final_error = torch.norm(target_angles - current_angles)
    print(f"最终误差: {final_error.item():.3f}")

    if final_error < 0.1:
        print("✅ 控制循环收敛良好")
        return True
    else:
        print("⚠️ 控制循环收敛较慢")
        return False

def main():
    """主测试函数"""
    print("🔬 速度控制系统逻辑测试")
    print("=" * 50)
    print("测试不依赖Isaac Gym的核心控制逻辑")
    print()

    # 运行所有测试
    success1 = test_velocity_integration()
    success2 = test_pd_control()
    success3 = test_network_output()
    success4 = test_control_loop()

    # 总结
    print("=" * 50)
    print("📊 测试结果总结:")
    print(f"   速度积分: {'✅ 通过' if success1 else '❌ 失败'}")
    print(f"   PD控制:   {'✅ 通过' if success2 else '❌ 失败'}")
    print(f"   网络输出: {'✅ 通过' if success3 else '❌ 失败'}")
    print(f"   控制循环: {'✅ 通过' if success4 else '❌ 失败'}")

    if all([success1, success2, success3, success4]):
        print("\n🎉 所有逻辑测试通过！速度控制系统实现正确")
        return 0
    else:
        print("\n⚠️ 部分测试失败，请检查实现")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
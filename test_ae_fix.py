#!/usr/bin/env python3
"""
测试AE/log_prob修复的脚本

验证：
1. sample_with_ensemble返回正确的log_prob
2. squashed_log_prob计算正确
3. AE模式下actions和log_probs一致
"""

import numpy as np
from ppo_isaac import ActorNetwork

import torch

def test_squashed_log_prob():
    """测试squashed_log_prob计算"""
    print("🧪 测试squashed_log_prob计算...")

    # 创建Actor��络
    actor = ActorNetwork(state_dim=22, action_dim=6)

    # 创建测试数据
    batch_size = 4
    states = torch.randn(batch_size, 22)
    actions = torch.randn(batch_size, 6)
    actions = torch.tanh(actions)  # 确保在[-1,1]范围内

    # 获取分布
    dist = actor.get_dist(states, fixed_std=0.1)

    # 计算log_prob
    log_prob = actor.squashed_log_prob(dist, actions)

    print(f"   状态形状: {states.shape}")
    print(f"   动作形状: {actions.shape}")
    print(f"   log_prob形状: {log_prob.shape}")
    print(f"   log_prob值: {log_prob}")

    assert log_prob.shape == (batch_size,), f"log_prob形状错误: {log_prob.shape}"
    assert not torch.isnan(log_prob).any(), "log_prob包含NaN"
    assert not torch.isinf(log_prob).any(), "log_prob包含Inf"

    print("✅ squashed_log_prob测试通过")

def test_sample_with_ensemble():
    """测试sample_with_ensemble返回log_prob"""
    print("🧪 测试sample_with_ensemble返回log_prob...")

    # 创建Actor网络
    actor = ActorNetwork(state_dim=22, action_dim=6)

    # 创建测试数据
    batch_size = 4
    states = torch.randn(batch_size, 22)
    ensemble_size = 5
    delta_std = 0.1

    # 测试AE采样
    actions, log_probs = actor.sample_with_ensemble(
        states,
        ensemble_size=ensemble_size,
        use_delta_std=True,
        delta_std=delta_std
    )

    print(f"   状态形状: {states.shape}")
    print(f"   集成大小: {ensemble_size}")
    print(f"   动作形状: {actions.shape}")
    print(f"   log_prob形状: {log_probs.shape}")
    print(f"   动作值范围: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"   log_prob值: {log_probs}")

    # 验证形状
    assert actions.shape == (batch_size, 6), f"动作形状错误: {actions.shape}"
    assert log_probs.shape == (batch_size,), f"log_prob形状错误: {log_probs.shape}"

    # 验证动作在[-1,1]范围内
    assert (actions >= -1.0).all() and (actions <= 1.0).all(), "动作超出[-1,1]范围"

    # 验证log_prob合理性
    assert not torch.isnan(log_probs).any(), "log_prob包含NaN"
    assert not torch.isinf(log_probs).any(), "log_prob包含Inf"
    assert (log_probs < 0).all(), "log_prob应该为负值"

    print("✅ sample_with_ensemble测试通过")

def test_ae_consistency():
    """测试AE模式下action和log_prob的一致性"""
    print("🧪 测试AE模式action/log_prob一致性...")

    # 创建Actor网络
    actor = ActorNetwork(state_dim=22, action_dim=6)

    # 创建测试数据
    batch_size = 4
    states = torch.randn(batch_size, 22)
    ensemble_size = 3
    delta_std = 0.1

    # AE采样
    actions, log_probs = actor.sample_with_ensemble(
        states,
        ensemble_size=ensemble_size,
        use_delta_std=True,
        delta_std=delta_std
    )

    # 用相同分布重新计算log_prob
    dist = actor.get_dist(states, fixed_std=delta_std)
    recomputed_log_probs = actor.squashed_log_prob(dist, actions)

    print(f"   原始log_prob: {log_probs}")
    print(f"   重新计算log_prob: {recomputed_log_probs}")
    print(f"   差异: {torch.abs(log_probs - recomputed_log_probs)}")

    # 验证一致性（允许小的数值误差）
    assert torch.allclose(log_probs, recomputed_log_probs, atol=1e-5), \
        f"AE log_prob不一致! 最大差异: {torch.max(torch.abs(log_probs - recomputed_log_probs))}"

    print("✅ AE一致性测试通过")

def test_get_dist():
    """测试get_dist方法"""
    print("🧪 测试get_dist方法...")

    # 创建Actor网络
    actor = ActorNetwork(state_dim=22, action_dim=6)

    # 创建测试数据
    batch_size = 4
    states = torch.randn(batch_size, 22)
    fixed_std = 0.1

    # 测试固定std
    dist_fixed = actor.get_dist(states, fixed_std=fixed_std)
    assert torch.allclose(dist_fixed.stddev, torch.full_like(dist_fixed.stddev, fixed_std)), \
        "固定std设置失败"

    # 测试网络std
    dist_network = actor.get_dist(states, fixed_std=None)
    assert not torch.allclose(dist_network.stddev, dist_fixed.stddev), \
        "网络std应该与固定std不同"

    print(f"   固定std: {dist_fixed.stddev[0]}")
    print(f"   网络std: {dist_network.stddev[0]}")

    print("✅ get_dist测试通过")

def main():
    """主测试函数"""
    print("🚀 开始AE/log_prob修复测试")
    print("=" * 50)

    try:
        test_squashed_log_prob()
        print()

        test_sample_with_ensemble()
        print()

        test_ae_consistency()
        print()

        test_get_dist()
        print()

        print("🎉 所有测试通过！AE/log_prob修复成功！")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
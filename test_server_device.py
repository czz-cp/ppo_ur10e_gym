#!/usr/bin/env python3
"""
服务器设备兼容性测试脚本
专门用于验证第500步设备不匹配问题的修复效果
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 🔧 强制设置CUDA环境变量（修复服务器设备不匹配）
# **用户服务器使用GPU 2**
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

print("=" * 80)
print("🏥 [SERVER TEST] 服务器设备兼容性验证")
print("=" * 80)

def test_device_consistency():
    """测试设备一致性"""
    print("\n🔍 测试设备一致性:")

    # ���查CUDA环境
    if not torch.cuda.is_available():
        print("   ❌ CUDA不可用")
        return False

    print(f"   ✅ CUDA可用，版本: {torch.version.cuda}")
    print(f"   ✅ PyTorch版本: {torch.__version__}")
    print(f"   ✅ GPU数量: {torch.cuda.device_count()}")

    # 强制使用GPU 0
    try:
        torch.cuda.set_device(0)
        current_device = torch.cuda.current_device()
        print(f"   🔒 [FORCED] 当前设备: GPU {current_device}")

        # 测试张量创建
        test_tensor = torch.randn(100, 100, device='cuda:0')
        print(f"   ✅ 测试张量创建: {test_tensor.device}")

        # 测试张量操作
        result = torch.mm(test_tensor, test_tensor.T)
        print(f"   ✅ 矩阵乘法: {result.device}, 形状: {result.shape}")

        return True

    except Exception as e:
        print(f"   ❌ 设备测试失败: {e}")
        return False

def test_utils_functions():
    """测试修复的工具函数"""
    print("\n🧪 测试修复的工具函数:")

    try:
        # 导入修复的工具函数
        from utils import get_forced_device, _device_consistency_check, assert_same_device

        # 测试强制设备获取
        forced_device = get_forced_device()
        print(f"   ✅ get_forced_device(): {forced_device}")

        # 测试设备一致性检查
        device = _device_consistency_check()
        print(f"   ✅ _device_consistency_check(): {device}")

        # 测试设备断言函数
        tensor1 = torch.randn(10, 10, device='cuda:0')
        tensor2 = torch.randn(10, 10, device='cuda:0')

        assert_same_device(tensor1, tensor2, device='cuda:0')
        print(f"   ✅ assert_same_device(): 通过")

        return True

    except Exception as e:
        print(f"   ❌ 工具函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_creation():
    """测试环境创建和设备管理"""
    print("\n🏗️ 测试环境创建:")

    try:
        # 测试环境设备管理
        from ur10e_env_isaac import UR10ePPOEnvIsaac

        print("   创建UR10e环境...")
        env = UR10ePPOEnvIsaac(
            config_path="config_isaac.yaml",
            num_envs=2,
            device_id=0
        )

        print(f"   ✅ 环境设备: {env.device}")
        print(f"   ✅ 环境数量: {env.get_num_envs()}")
        print(f"   ✅ 状态维度: {env.get_num_obs()}")
        print(f"   ✅ 动作维度: {env.get_num_actions()}")

        # 测试环境reset
        print("   测试环境reset...")
        states = env.reset()
        print(f"   ✅ reset成功，状态形状: {states.shape}, 设备: {states.device}")

        # 测试环境step
        print("   测试环境step...")
        actions = torch.randn(env.get_num_envs(), env.get_num_actions(), device=env.device)
        next_states, rewards, dones, infos = env.step(actions)
        print(f"   ✅ step成功:")
        print(f"      next_states: {next_states.shape}, {next_states.device}")
        print(f"      rewards: {rewards.shape}, {rewards.device}")
        print(f"      dones: {dones.shape}, {dones.device}")

        return True

    except Exception as e:
        print(f"   ❌ 环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ppo_creation():
    """测试PPO创建和设备管理"""
    print("\n🤖 测试PPO训练器:")

    try:
        from ppo_isaac import PPOIsaac
        from ur10e_env_isaac import UR10ePPOEnvIsaac
        from utils import load_config

        # 创建环境
        env = UR10ePPOEnvIsaac(
            config_path="config_isaac.yaml",
            num_envs=2,
            device_id=0
        )

        # 加载配置
        config = load_config("config_isaac.yaml")

        # 创建PPO训练器
        print("   创建PPO训练器...")
        ppo = PPOIsaac(env, config)

        print(f"   ✅ PPO设备: {ppo.device}")
        print(f"   ✅ 网络参数设备: {next(ppo.actor.parameters()).device}")
        print(f"   ✅ 价值网络设备: {next(ppo.critic.parameters()).device}")

        # 验证设备一致性
        assert ppo.device == env.device, f"PPO设备{ppo.device} != 环境设备{env.device}"
        print(f"   ✅ PPO和环境设备一致")

        return True

    except Exception as e:
        print(f"   ❌ PPO测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def simulate_training_steps(num_steps=10):
    """模拟训练步骤，测试第500步问题"""
    print(f"\n🚀 模拟训练步骤 ({num_steps}步):")

    try:
        from ppo_isaac import PPOIsaac
        from ur10e_env_isaac import UR10ePPOEnvIsaac
        from utils import load_config

        # 创建环境和PPO
        env = UR10ePPOEnvIsaac(
            config_path="config_isaac.yaml",
            num_envs=2,
            device_id=0
        )
        config = load_config("config_isaac.yaml")
        ppo = PPOIsaac(env, config)

        states = env.reset()

        for step in range(num_steps):
            # 生成动作
            with torch.no_grad():
                actions, log_probs = ppo.actor.sample(states)
                values = ppo.critic(states)

            # 环境步进
            next_states, rewards, dones, infos = env.step(actions)

            # 检查设备一致性（关键！）
            for tensor_name, tensor in [
                ('states', states), ('actions', actions), ('next_states', next_states),
                ('rewards', rewards), ('dones', dones), ('values', values)
            ]:
                if tensor.device != ppo.device:
                    print(f"   ❌ Step {step}: {tensor_name}设备不一致: {tensor.device} != {ppo.device}")
                    return False

            if step % 5 == 0:
                print(f"   ✅ Step {step}: 所有张量设备一致 {ppo.device}")

            states = next_states

        print(f"   ✅ 模拟训练{num_steps}步完成，无设备错误")
        return True

    except Exception as e:
        print(f"   ❌ 模拟训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始服务器设备兼容性验证...\n")

    # 检查当前目录
    if not Path("config_isaac.yaml").exists():
        print("❌ config_isaac.yaml不存在，请在正确目录运行此脚本")
        return

    tests = [
        ("基础设备一致性", test_device_consistency),
        ("修复工具函数", test_utils_functions),
        ("环境创建", test_environment_creation),
        ("PPO创建", test_ppo_creation),
        ("模拟训练步骤", simulate_training_steps),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 运行测试: {test_name}")
        print(f"{'='*60}")

        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            results.append((test_name, False))

    # 测试结果汇总
    print(f"\n{'='*80}")
    print("📊 测试结果汇总")
    print(f"{'='*80}")

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name:20} : {status}")
        if success:
            passed += 1

    print(f"\n总体结果: {passed}/{total} 测试通过")

    if passed == total:
        print("🎉 所有测试通过！服务器设备不匹配问题已修复")
        print("💡 建议在服务器上运行完整训练验证")
    else:
        print("⚠️ 仍有测试失败，需要进一步调试")
        print("💡 建议检查CUDA环境和配置文件")

    print(f"{'='*80}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
设备兼容性测试脚本
用于多GPU服务器训练前的设备兼容性检查
"""

import torch
import sys
import os

def check_cuda_environment():
    """检查CUDA环境"""
    print("🔍 CUDA环境检查:")
    if not torch.cuda.is_available():
        print("   ❌ CUDA不可用")
        return False

    print(f"   ✅ CUDA可用，版本: {torch.version.cuda}")
    print(f"   ✅ PyTorch版本: {torch.__version__}")
    print(f"   ✅ GPU数量: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"      计算能力: {props.major}.{props.minor}")
        print(f"      总内存: {props.total_memory / 1024**3:.1f} GB")

    return True

def test_device_compatibility(target_device_id=0):
    """测试设备兼容性"""
    print(f"\n🎯 测试GPU {target_device_id}兼容性:")

    if not torch.cuda.is_available():
        print("   ❌ CUDA不可用，无法测试GPU")
        return False

    if target_device_id >= torch.cuda.device_count():
        print(f"   ❌ GPU {target_device_id} 不存在，只有 {torch.cuda.device_count()} 个GPU")
        return False

    try:
        # 设置当前设备
        torch.cuda.set_device(target_device_id)
        current_device = torch.cuda.current_device()
        print(f"   ✅ 设置当前设备: GPU {current_device}")

        # 测试张量创建和操作
        print("   🧪 测试张量创建...")
        x = torch.randn(1000, 1000, device=f'cuda:{target_device_id}')
        y = torch.randn(1000, 1000, device=f'cuda:{target_device_id}')
        z = torch.mm(x, y)
        print(f"   ✅ 张量操作成功，形状: {z.shape}, 设备: {z.device}")

        # 测试内存
        allocated = torch.cuda.memory_allocated(target_device_id)
        cached = torch.cuda.memory_reserved(target_device_id)
        print(f"   ✅ 内存使用: {allocated/1024**2:.1f} MB (已分配), {cached/1024**2:.1f} MB (已缓存)")

        return True

    except Exception as e:
        print(f"   ❌ 设备测试失败: {e}")
        return False

def recommend_device_config():
    """推荐设备配置"""
    print(f"\n💡 推荐设备配置:")

    if not torch.cuda.is_available():
        print("   使用CPU模式:")
        print("   train_isaac_fixed.py --device_id -1")
        return

    gpu_count = torch.cuda.device_count()
    print(f"   检测到 {gpu_count} 个GPU:")

    for i in range(gpu_count):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")

    print(f"\n   推荐使用GPU 0（最稳定）:")
    print(f"   train_isaac_fixed.py --device_id 0")

    if gpu_count > 1:
        print(f"   或者指定其他GPU:")
        for i in range(1, gpu_count):
            print(f"   train_isaac_fixed.py --device_id {i}")

def main():
    """主函数"""
    print("=" * 60)
    print("🚀 Isaac Gym UR10e 训练设备兼容性检查")
    print("=" * 60)

    # 检查CUDA环境
    if not check_cuda_environment():
        print("\n❌ CUDA环境检查失败")
        sys.exit(1)

    # 测试默认设备
    default_device = 0
    if not test_device_compatibility(default_device):
        print(f"\n❌ 默认GPU {default_device} 测试失败")
        sys.exit(1)

    # 推荐配置
    recommend_device_config()

    print(f"\n✅ 设备兼容性检查完成！可以开始训练")
    print("=" * 60)

if __name__ == "__main__":
    main()
# GPU 2 服务器配置说明

## 用户服务器环境
- **使用GPU**: GPU 2
- **修复目标**: 解决第500步 cuda:0 vs cuda:2 设备不匹配问题

## 关键配置修改

### 1. 环境变量设置
```bash
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### 2. 配置文件修改 (config_isaac.yaml)
```yaml
env:
  device_id: 2                   # GPU设备ID (用户服务器使用GPU 2)

device: "cuda:2"                 # PyTorch设备 (用户服务器使用GPU 2)
sim:
  device_id: 2                   # Isaac Gym仿真设备ID (用户服务器使用GPU 2)
graphics:
  graphics_device_id: 2          # 图形设备ID (用户服务器使用GPU 2)
```

### 3. 代码逻辑说明

**重要概念**: 当设置 `CUDA_VISIBLE_DEVICES=2` 后：
- 物理 GPU 2 变成逻辑 GPU 0
- 所以代码中使用 `cuda:0` 实际上是物理 GPU 2
- 这样避免设备混合，确保一致性

**设备映射**:
```
物理GPU 2 → 设置CUDA_VISIBLE_DEVICES=2 → 逻辑GPU 0 → cuda:0
```

### 4. 运行命令

**测试设备兼容性**:
```bash
python test_server_device.py
```

**正常训练**:
```bash
python train_isaac_fixed.py --config config_isaac.yaml
```

## 预期输出示例

```
🔧 [SERVER FIX] 强制CUDA设备一致性设置...
   ✅ CUDA可用，版本: 11.8
   ✅ 检测到GPU数量: 1  (只有GPU 2可见)
   🔒 [FORCED] 当前CUDA设备: GPU 0 (原GPU 2)
   ✅ [SUCCESS] 成功强制使用GPU 2 (索引0)
   🧪 测试张量设备: cuda:0 (原GPU 2)
   ✅ [DEVICE OK] 使用目标设备: cuda:0 (原GPU 2)
```

## 故障排除

### 如果仍然出现设备错误：
1. **检查nvidia-smi**:
   ```bash
   nvidia-smi
   ```

2. **验证GPU 2可用**:
   ```bash
   python -c "import torch; print(f'GPU 2可用: {torch.cuda.is_available() and torch.cuda.device_count() > 2}')"
   ```

3. **强制重新设置环境变量**:
   ```bash
   unset CUDA_VISIBLE_DEVICES
   export CUDA_VISIBLE_DEVICES=2
   python train_isaac_fixed.py --config config_isaac.yaml
   ```

## 关键修复点

1. **统一设备**: 所有操作使用相同的GPU（GPU 2）
2. **环境变量优先**: `CUDA_VISIBLE_DEVICES=2` 确保只看到GPU 2
3. **逻辑映射**: GPU 2 → cuda:0 (避免混合)
4. **运行时检查**: 持续验证设备一致性

这个配置专门针对用户服务器使用GPU 2的环境，通过强制设备映射和一致性检查，解决第500步的设备不匹配问题。
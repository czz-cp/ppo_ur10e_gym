"""
UR10e PPO环境 - Isaac Gym版本

基于Isaac Gym实现的UR10e机械臂RL-PID混合控制环境
将原本的MuJoCo实现迁移到Isaac Gym以获得更好的并行性能
"""

# IMPORTANT: Isaac Gym must be imported before PyTorch
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from isaacgym.torch_utils import *

import torch
import numpy as np
import math
import sys
from typing import Dict, Any, List, Tuple, Optional
import os

# 导入运动学（延迟导入以避免导入顺序问题）
ur10e_kinematics_fixed = None
reward_normalizer_module = None

# 导入设备一致性检查工具
from utils import assert_same_device, check_tensor_devices, get_tensor_device, ensure_device

def get_kinematics():
    """延迟导入运动学模块"""
    global ur10e_kinematics_fixed
    if ur10e_kinematics_fixed is None:
        try:
            from ur10e_kinematics_fixed import UR10eKinematicsFixed
            ur10e_kinematics_fixed = UR10eKinematicsFixed
        except ImportError:
            print("⚠️ 无法导入运动学模块，使用简化实现")
            ur10e_kinematics_fixed = None
    return ur10e_kinematics_fixed

def get_reward_normalizer():
    """获取奖励归一化模块 - 使用utils.py中的完整版本"""
    # 优先尝试从当前utils导入（修复device问题的版本）
    try:
        from utils import RewardNormalizer
        print("✅ 使用utils.py中的RewardNormalizer（支持reset和device参数）")
        return RewardNormalizer
    except ImportError as e:
        print(f"⚠️ 无法导入utils.py中的RewardNormalizer: {e}")
        print("⚠️ 尝试从父目录导入...")
        # 备选方案：尝试父目录
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ppo_ur10e'))
        try:
            from utils import RewardNormalizer
            print("✅ 使用父目录的RewardNormalizer")
            return RewardNormalizer
        except ImportError:
            print("❌ 无法导入任何RewardNormalizer，训练可能失败")
            return None


class UR10ePPOEnvIsaac:
    """
    UR10e PPO环境 - Isaac Gym版本

    特性:
    - 支持大规模并行仿真
    - GPU加速计算
    - RL-PID混合控制架构
    - 基于雅可比的精确控制
    - 奖励归一化
    """

    def __init__(self,
                 config_path: str = "config.yaml",
                 num_envs: int = 512,
                 device_id: int = 0):
        """
        初始化Isaac Gym环境

        Args:
            config_path: 配置文件路径
            num_envs: 并行环境数量
            device_id: GPU设备ID
        """
        # 加载配置
        self.config = self._load_config(config_path)
        self.num_envs = num_envs

        # 目标：把单步奖励控制在 [-50, 0] 左右
        self.reward_scale = 1e-5  # 你可以后面微调，比如 5e-4, 2e-3 之类

        # 🎯 优先使用传入的device_id，覆盖配置文件中的设置（多GPU服务器兼容）
        self.device_id = device_id
        if 'device_id' in self.config.get('env', {}):
            config_device_id = self.config['env']['device_id']
            if device_id != config_device_id:
                print(f"⚠️ 覆盖配置文件中的设备ID: {config_device_id} -> {device_id}")

        # 🎯 设备配置（参考isaac_gym_manipulator成功方案）
        device_str = self.config.get('device', 'cuda:0')
        self.device = torch.device(device_str)

        # 图形设备配置（修复config参数传递）
        viz_config = self.config.get('visualization', {})
        if viz_config.get('enable', False):
            graphics_device_id = self.config.get('graphics', {}).get('graphics_device_id', 0)
        else:
            graphics_device_id = -1

        # 仿真设备ID（从config读取，保持与PyTorch设备一致）
        self.sim_device_id = self.config.get('sim', {}).get('device_id', 0)

        print(f"🎯 设备配置 (isaac_gym_manipulator方案):")
        print(f"   PyTorch设备: {self.device}")
        print(f"   仿真设备ID: {self.sim_device_id}")
        print(f"   图形设备ID: {graphics_device_id}")

        # 🔍 调试：打印config关键参数
        print(f"🔍 Config调试信息:")
        print(f"   config['device']: {self.config.get('device', 'NOT_FOUND')}")
        print(f"   config['sim']: {self.config.get('sim', 'NOT_FOUND')}")
        print(f"   config['graphics']: {self.config.get('graphics', 'NOT_FOUND')}")
        print(f"   config['visualization']: {self.config.get('visualization', 'NOT_FOUND')}")

        # 强制CUDA设备一致性
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
            print(f"   ✅ 强制设置CUDA设备: {self.device}")

        # UR10e机器人参数
        self.num_dofs = 6  # UR10e有6个自由度

        # ��境参数
        self.max_steps = self.config['env']['max_steps']
        self.dt = self.config['env']['dt']

        # UR10e官方关节限制（基于isaac_gym_manipulator中的官方URDF配置）
        self.joint_limits = np.array([
            [-6.28319, 6.28319],   # shoulder_pan_joint: ±360° = ±2π rad
            [-6.28319, 6.28319],   # shoulder_lift_joint: ±360° = ±2π rad
            [-3.14159, 3.14159],   # elbow_joint: ±180° = ±π rad (人为限制避免规划问题)
            [-6.28319, 6.28319],   # wrist_1_joint: ±360° = ±2π rad
            [-6.28319, 6.28319],   # wrist_2_joint: ±360° = ±2π rad
            [-6.28319, 6.28319]    # wrist_3_joint: ±360° = ±2π rad
        ])

        print("📐 UR10e官方关节限制:")
        joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_joint', 'wrist_1', 'wrist_2', 'wrist_3']
        for i, (name, limits) in enumerate(zip(joint_names, self.joint_limits)):
            limits_deg = np.degrees(limits)
            print(f"  {i+1}. {name:12}: [{limits[0]:.2f}, {limits[1]:.2f}] rad ({limits_deg[0]:.0f}°, {limits_deg[1]:.0f}°)")

        # 动作空间限制 (RL补偿力矩)
        max_compensation_torque = 30.0  # 30 N⋅m补偿力矩 (约为最大力矩的10%)
        self.action_space_high = np.array([max_compensation_torque] * 6)  # [τ1, τ2, τ3, τ4, τ5, τ6]
        self.action_space_low = np.array([-max_compensation_torque] * 6)

        # 状态空间 (18维：当前关节角度6 + 目标关节角度6 + 当前末端位置3 + 目标位置3)
        self.state_dim = 18
        self.action_dim = 6

        # 初始化Isaac Gym
        self.gym = gymapi.acquire_gym()
        self._init_simulator()

        # 创建环境
        self._create_environments()

        # 运动学解算器
        kinematics_class = get_kinematics()
        if kinematics_class is not None:
            self.kinematics = kinematics_class()
        else:
            print("⚠️ 使用简化运动学实现")
            self.kinematics = None

        # 🎯 奖励归一化器 (每个环境独立)
        reward_config = self.config.get('reward_normalization', {})

        if reward_config.get('enabled', True):
            reward_normalizer_class = get_reward_normalizer()
            if reward_normalizer_class is not None:
                self.reward_normalizers = [
                    reward_normalizer_class(
                        gamma=reward_config.get('gamma', 0.99),
                        clip_range=reward_config.get('clip_range', 5.0),
                        normalize_method=reward_config.get('normalize_method', 'running_stats'),
                        warmup_steps=reward_config.get('warmup_steps', 100)
                    ) for _ in range(num_envs)
                ]
                print(f"✅ 启用奖励归一化: gamma={reward_config.get('gamma', 0.99)}, clip_range={reward_config.get('clip_range', 5.0)}")
            else:
                print("⚠️ 无法加载奖励归一化器，使用原始奖励")
                self.reward_normalizers = [None] * num_envs
        else:
            print("📝 奖励归一化已禁用")
            self.reward_normalizers = [None] * num_envs

        # 状态变量
        self.current_step = 0
        self.episode_steps = torch.zeros(num_envs, device=self.device)  # 每个环境的当前episode步数
        self.debug_step = 0  # 调试步数计数器
        self.start_joint_angles = None

        # 🎯 稳定性跟踪变量
        self.on_goal_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.stability_required_steps = 100  # 需要连续100步在目标范围内
        self.target_positions = None
        self.target_joint_angles = None  # 🎯 新增：目标关节角度
        self.prev_position_errors = None
        self.prev_joint_errors = None  # 🎯 新增：上次关节角度误差

        # 🎯 二次型奖励函数参数（基于论文设计）
        # Q矩阵：对角正定矩阵，位置误差权重远大于速度误差权重
        self.Q_position_weight = 8e4   # 位置误差权重 (论文示例值)
        self.Q_velocity_weight = 10.0  # 速度误差权重 (论文示例值)
        # 可以针对不同关节设置不同权重
        self.Q_weights = torch.ones(6, device=self.device) * self.Q_position_weight
        self.Q_velocity_weights = torch.ones(6, device=self.device) * self.Q_velocity_weight

        print(f"✅ Isaac Gym UR10e环境初始化完成")
        print(f"   并行环境数: {num_envs}")
        print(f"   设备ID: {device_id}")
        print(f"   状态空间: {self.state_dim}维 (RL-PID混合��制)")
        print(f"   动作空间: {self.action_dim}维 (PID参数调度)")

        # 🎯 显示官方UR10e PID参数
        if 'pid_params' in self.config and 'base_gains' in self.config['pid_params']:
            pid_params = self.config['pid_params']['base_gains']
            print(f"🎯 使用官方UR10e PID参数:")
            print(f"   Kp: {pid_params['p']}")
            print(f"   Kd: {pid_params['d']}")
            print(f"   Ki: {pid_params['i']}")

        # 🔍 设备兼容性���查（多GPU服务器）
        self._device_consistency_check()

    def _device_consistency_check(self):
        """设备一致性检查和修复（多GPU服务器兼容）"""
        if torch.cuda.is_available():
            print(f"🔍 设备兼容性检查:")
            print(f"   PyTorch当前设备: {torch.cuda.current_device()}")
            print(f"   PyTorch设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")

            # 强制所有后续CUDA操作都在指定GPU上
            if self.device.type == 'cuda':
                torch.cuda.set_device(self.device)
                print(f"   ✅ 强制所有CUDA操作使用GPU {self.device.index}")
        else:
            print("   ℹ️ CUDA不可用，使用CPU模式")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        import yaml
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"⚠️ 配置文件 {config_path} 未找到，使用默认配置")
            config = self._get_default_config()
        return config

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'env': {
                'max_steps': 1000,
                'dt': 0.01,
                'action_bound': 0.03,
                'xml_path': '../universal_robots_ur10e/ur10e_mujoco/scene.xml'
            },
            'reward': {
                'accuracy': {'weight': 5.0, 'threshold': 0.005},
                'stability': {'weight': 0.5},
                'speed': {'weight': 1.0},
                'energy': {'weight': 0.001},
                'extra': {'success_reward': 10.0}
            }
        }

    def _init_simulator(self):
        """初始化Isaac Gym仿真器"""
        # 创建仿真器 - 基于成功的isaac_gym_manipulator配置
        sim_params = gymapi.SimParams()
        sim_params.dt = self.dt
        sim_params.substeps = 2  # 使用成功配置的子步数
        sim_params.up_axis = gymapi.UP_AXIS_Z

        # 设置重力（参考成功配置）
        sim_params.gravity.x = 0
        sim_params.gravity.y = 0
        sim_params.gravity.z = -9.81

        # 设置物理引擎参数 - 使用成功配置
        sim_params.physx.solver_type = 1  # 使用solver_type=1（成功配置）
        sim_params.physx.num_position_iterations = 4  # 位置迭代次数
        # 获取仿真配置
        simulator_config = self.config.get('simulator', {})

        sim_params.physx.num_velocity_iterations = 1  # 速度迭代次数
        sim_params.physx.num_threads = 0  # 线程数（0=自动）
        sim_params.physx.use_gpu = simulator_config.get('use_gpu', True)
        # 修复：正确设置GPU渲染管线，避免CUDA内存问题
        sim_params.use_gpu_pipeline = simulator_config.get('use_gpu_pipeline', False)
        # 🎯 修复渲染配置读取（使用新的config结构）
        viz_config = self.config.get('visualization', {})
        graphics_config = self.config.get('graphics', {})

        enable_rendering = viz_config.get('enable', False)
        graphics_device_id = graphics_config.get('graphics_device_id', self.device_id) if enable_rendering else -1

        # 根据渲染设置选择图形设备
        if enable_rendering:
            print(f"🎬 启用渲染模式，图形设备: {graphics_device_id}")
        else:
            graphics_device_id = -1  # 无头模式
            print("🖥️ 无头模式，禁用渲染")

        # 🎯 Isaac Gym仿真器创建（isaac_gym_manipulator方案）
        print(f"🎮 创建Isaac Gym仿真器 - 计算设备: {self.sim_device_id}, 图形设备: {graphics_device_id}")
        self.sim = self.gym.create_sim(
            compute_device=self.sim_device_id,  # 使用配置中的仿真设备ID
            graphics_device=graphics_device_id,
            type=gymapi.SIM_PHYSX,  # 关键：使用PhysX而不是默认的FleX
            params=sim_params
        )

        if self.sim is None:
            raise Exception("Failed to create Isaac Gym simulator")

    def _create_environments(self):
        """创建并行环境 - 参考isaac_gym_manipulator实现"""

        # 添加地面 - 参考isaac_gym_manipulator
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0
        self.gym.add_ground(self.sim, plane_params)
        print("✅ 已添加地面")

        # 🎯 渲染配置（使用新的config结构）
        viz_config = self.config.get('visualization', {})
        graphics_config = self.config.get('graphics', {})

        self.enable_rendering = viz_config.get('enable', False)
        self.graphics_device = graphics_config.get('graphics_device_id', self.device_id)

        if self.enable_rendering:
            print(f"🎬 启用Isaac Gym渲染，图形设备: {self.graphics_device}")
        else:
            print("🖥️  无头模式运行，禁用渲染")

        # 获取UR10e资产路径
        asset_root = "."  # 当前目录，包含ur10e_isaac.urdf
        asset_file = "scene.xml"

        # 加载UR10e资产 - 参考isaac_gym_manipulator设置
        ur10e_asset_options = gymapi.AssetOptions()
        ur10e_asset_options.flip_visual_attachments = True  # 启用以正确显示mesh
        ur10e_asset_options.fix_base_link = True
        ur10e_asset_options.use_mesh_materials = True  # 启用材质
        ur10e_asset_options.override_com = True
        ur10e_asset_options.override_inertia = True
        ur10e_asset_options.vhacd_enabled = True  # 启用VHACD处理凸包碰撞
        ur10e_asset_options.vhacd_params = gymapi.VhacdParams()
        ur10e_asset_options.vhacd_params.resolution = 300000
        ur10e_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT  # 修复：默认力矩控制模式

        # 使用我们创建的URDF文件
        urdf_path = os.path.join(asset_root, "ur10e_isaac.urdf")
        if not os.path.exists(urdf_path):
            # 如果URDF不存在，回退到创建简单URDF
            urdf_path = self._create_ur10e_urdf(asset_root)
        else:
            print(f"✅ 使用URDF文件: {urdf_path}")

        try:
            # 使用load_asset而不是load_urdf，参考isaac_gym_manipulator
            self.ur10e_asset = self.gym.load_asset(
                self.sim, asset_root, "ur10e_isaac.urdf", ur10e_asset_options
            )
            print(f"✅ UR10e资产加载成功")
        except Exception as e:
            print(f"❌ UR10e资产加载失败: {e}")
            print(f"   资产路径: {asset_root}/ur10e_isaac.urdf")
            raise

        # 设置环境间距
        env_spacing = 2.0
        env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

        # 创建环境
        self.envs = []
        self.ur10e_handles = []

        for i in range(self.num_envs):
            # 创建环境
            env = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            self.envs.append(env)

            # 创建UR10e机器人
            ur10e_handle = self.gym.create_actor(
                env, self.ur10e_asset, gymapi.Transform(), f"ur10e_{i}"
            )
            self.ur10e_handles.append(ur10e_handle)

            # 设置UR10e属性
            self.gym.set_actor_dof_properties(env, ur10e_handle, self._get_ur10e_dof_props())

        # 创建张量视图
        self._create_tensor_views()

        # 设置渲染器（如果启用渲染）
        if self.enable_rendering:
            self._setup_renderer()

    def _create_ur10e_urdf(self, asset_root: str) -> str:
        """创建UR10e URDF文件（如果不存在）"""
        # 这里应该有一个URDF到MuJoCo XML的转换
        # 暂时返回一个占位符路径
        urdf_path = os.path.join(asset_root, "ur10e.urdf")
        if not os.path.exists(urdf_path):
            # 创建一个简单的URDF文件
            urdf_content = self._generate_simple_urdf()
            with open(urdf_path, 'w') as f:
                f.write(urdf_content)
        return urdf_path

    def _generate_simple_urdf(self) -> str:
        """生成简单的UR10e URDF"""
        return """<?xml version="1.0"?>
<robot name="ur10e">
  <link name="base_link">
    <inertial>
      <mass value="10"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>

  <!-- UR10e 6个关节 -->
  <link name="shoulder_pan_joint">
    <inertial>
      <mass value="5"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <link name="shoulder_lift_joint">
    <inertial>
      <mass value="5"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <link name="elbow_joint">
    <inertial>
      <mass value="5"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <link name="wrist_1_joint">
    <inertial>
      <mass value="3"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
    </inertial>
  </link>

  <link name="wrist_2_joint">
    <inertial>
      <mass value="3"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
    </inertial>
  </link>

  <link name="wrist_3_joint">
    <inertial>
      <mass value="2"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>
  </link>

  <!-- 关节连接 -->
  <joint name="shoulder_pan_joint" type="revolute" parent="base_link" child="shoulder_pan_joint">
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" effort="100"/>
  </joint>

  <joint name="shoulder_lift_joint" type="revolute" parent="shoulder_pan_joint" child="shoulder_lift_joint">
    <axis xyz="0 1 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="100"/>
  </joint>

  <joint name="elbow_joint" type="revolute" parent="shoulder_lift_joint" child="elbow_joint">
    <axis xyz="0 1 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="100"/>
  </joint>

  <joint name="wrist_1_joint" type="revolute" parent="elbow_joint" child="wrist_1_joint">
    <axis xyz="0 1 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="50"/>
  </joint>

  <joint name="wrist_2_joint" type="revolute" parent="wrist_1_joint" child="wrist_2_joint">
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" effort="50"/>
  </joint>

  <joint name="wrist_3_joint" type="revolute" parent="wrist_2_joint" child="wrist_3_joint">
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" effort="50"/>
  </joint>
</robot>"""

    def _get_ur10e_dof_props(self):
        """获取UR10e DOF属性"""
        dof_props = self.gym.get_actor_dof_properties(self.envs[0], self.ur10e_handles[0])

        # 设置关节属性（修复：使用EFFORT模式支持力矩控制）
        dof_props["driveMode"] = gymapi.DOF_MODE_EFFORT  # 关键修复：力矩控制模式
        dof_props["stiffness"] = 0.0
        dof_props["damping"] = 0.0

        # 设置关节限制
        for i in range(6):
            dof_props["lower"][i] = self.joint_limits[i][0]
            dof_props["upper"][i] = self.joint_limits[i][1]

        return dof_props

    def _create_tensor_views(self):
        """创建GPU张量视图"""
        # 观测空间 (更新为18维)
        self.obs_buf = torch.zeros(
            (self.num_envs, self.state_dim),
            device=self.device, dtype=torch.float32
        )

        # 动作空间
        self.actions_buf = torch.zeros(
            (self.num_envs, self.action_dim),
            device=self.device, dtype=torch.float32
        )

        # 奖励
        self.rewards_buf = torch.zeros(
            (self.num_envs,),
            device=self.device, dtype=torch.float32
        )

        # 完成标志
        self.dones_buf = torch.zeros(
            (self.num_envs,),
            device=self.device, dtype=torch.bool
        )

        # 🎯 获取Isaac Gym张量视图并强制设备一致性
        self.root_states = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_states = self.gym.acquire_dof_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(self.root_states)
        self.dof_states = gymtorch.wrap_tensor(self.dof_states)

        # 🚨 强制Isaac Gym张量移动到指定设备（修复多GPU设备不匹配问题）
        if self.device.type == 'cuda':
            self.root_states = self.root_states.to(self.device)
            self.dof_states = self.dof_states.to(self.device)
            print(f"🔧 Isaac Gym张量已移动到GPU {self.device.index}: {self.device}")

    def _setup_renderer(self):
        """设置Isaac Gym渲染器 - 参考isaac_gym_manipulator实现"""
        try:
            # 创建viewer - 使用标准Isaac Gym viewer
            self.viewer = self.gym.create_viewer(
                self.sim,
                gymapi.CameraProperties()
            )

            if self.viewer is None:
                print("⚠️ 无法创建viewer，使用无头模式")
                self.enable_rendering = False
                return

            # 设置相机视角 - 参考isaac_gym_manipulator实现
            cam_pos = gymapi.Vec3(2.0, 0.0, 2.0)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
            # 使用None作为环境参数，参考isaac_gym_manipulator
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

            print(f"✅ 渲染器设置完成，使用标准Isaac Gym viewer")
            print("   参考isaac_gym_manipulator成功实现")

            # 测试渲染（使用简单方式）
            self._test_render_simple()

        except Exception as e:
            print(f"⚠️ 渲染器设置失败: {e}")
            print("   继续使用无头模式")
            self.enable_rendering = False
            self.viewer = None

    def _test_render_simple(self):
        """测试渲染功能（参考isaac_gym_manipulator实现）"""
        try:
            # 运行几步仿真进行测试
            for i in range(3):
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)

                # 使用标准图形更新
                self.gym.step_graphics(self.sim)

                # 绘制viewer
                if self.viewer is not None:
                    self.gym.draw_viewer(self.viewer, self.sim, True)

                # 短暂延迟让窗口显示
                if i == 0:
                    import time
                    time.sleep(0.1)

            print("✅ 渲染测试通过")
        except Exception as e:
            print(f"⚠️ 渲染测试失败: {e}")
            print("   可能是环境或驱动问题，但训练仍可继续")
            # 不禁用渲染，可能在实际运行时可以工作

    def get_num_envs(self) -> int:
        """获取环境数量"""
        return self.num_envs

    def reset(self) -> torch.Tensor:
        """
        重置所有环境

        Returns:
            obs: 初始观测张量 [num_envs, state_dim]
        """
        # 随机生成起始关节角度
        self.start_joint_angles = self._sample_random_joint_angles_batch()

        # 🎯 随机生成目标关节角度，然后用正运动学生成目标位置
        self.target_joint_angles = self._sample_target_joint_angles_batch()
        self.target_positions = self._compute_positions_from_joint_angles(self.target_joint_angles)

        # 设置初始状态（参考 isaac_gym_manipulator 模式，避免CUDA内存错误）
        # 直接使用 start_idx:end_idx 批量设置，而不是逐个索引
        for i in range(self.num_envs):
            # 确保关节角度是6维的
            joint_angles = self.start_joint_angles[i]
            if len(joint_angles) != 6:
                if len(joint_angles) > 6:
                    joint_angles = joint_angles[:6]
                else:
                    joint_angles = torch.cat([joint_angles, torch.zeros(6-len(joint_angles), device=self.device)])

            # 使用批量切片操作（isaac_gym_manipulator 成功模式）
            start_idx = i * self.num_dofs
            end_idx = (i + 1) * self.num_dofs
            self.dof_states[start_idx:end_idx, 0] = joint_angles.to(self.device)  # 位置 - 确保在正确设备上
            self.dof_states[start_idx:end_idx, 1] = 0.0  # 速度

        # 🎯 修复DOF状态张量设备问题（确保CPU张量再unwrap）
        if self.dof_states.device.type != 'cpu':
            dof_states_cpu = self.dof_states.cpu()
        else:
            dof_states_cpu = self.dof_states
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(dof_states_cpu))

        # 运行几步simulation让机械臂稳定（参考isaac_gym_manipulator）
        for _ in range(10):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

        # 刷新状态张量（isaac_gym_manipulator 模式）
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        # 重置内部状态
        self.current_step = 0
        self.episode_steps.zero_()  # 重置每个环境的episode步数
        self.prev_position_errors = torch.ones(self.num_envs, device=self.device) * 10.0
        self.prev_joint_errors = torch.ones(self.num_envs, device=self.device) * 10.0  # 🎯 重置关节误差

        # 🎯 重置稳定性跟踪
        self.on_goal_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # 重置奖励归一化器
        for normalizer in self.reward_normalizers:
            normalizer.reset()

        # 推进一步
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # 获取初始观测
        obs = self._get_states()

        return obs

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        执行一步仿真

        Args:
            actions: RL补偿力矩动作 [num_envs, 6] [τ1, τ2, τ3, τ4, τ5, τ6]

        Returns:
            obs: 下一步状态 [num_envs, state_dim]
            rewards: 奖励 [num_envs]
            dones: 完成标志 [num_envs]
            info: 额外信息
        """
        # 增加调试步数计数器
        self.debug_step += 1

        # 设备一致性检查
        actions = ensure_device(actions, self.device)

        # 调试信息（可选择性启用）
        if hasattr(self, '_debug_mode') and self._debug_mode:
            actual_device = actions.device
            expected_device = self.device
            if actual_device != expected_device:
                print(f"⚠️ Step输入设备不匹配: actions在{actual_device}, 期望在{expected_device}")

        self.actions_buf = actions

        # 执行RL-PID控制
        self._apply_rl_pid_control(actions)

        # 推进一步
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)

        # 渲染（如果启用）- 参考isaac_gym_manipulator成功实现
        if self.enable_rendering:
            try:
                # 绘制目标点为红色球体
                self._draw_target_sphere()

                # 使用Isaac Gym标准的图形更新方式
                self.gym.step_graphics(self.sim)

                # 如果有viewer，绘制viewer
                if hasattr(self, 'viewer') and self.viewer is not None:
                    self.gym.draw_viewer(self.viewer, self.sim, True)

            except Exception as e:
                # 静默处理渲染错误，避免中断训练
                if hasattr(self, '_debug_mode') and self._debug_mode:
                    print(f"⚠️ 渲染错误: {e}")
                pass

        # 获取新状态
        obs = self._get_states()

        # 计算奖励
        rewards = self._compute_rewards_batch(actions)

        # 检查完成条件
        dones = self._check_done_batch()

        # 处理完成的episode - 重置相关状态
        for i in range(self.num_envs):
            if dones[i]:
                self.episode_steps[i] = 0  # 重置该环境的episode步数

        # 更新奖励归一化器
        """for i in range(self.num_envs):
            if not dones[i]:
                if self.reward_normalizers[i] is not None:
                    self.reward_normalizers[i].update(rewards[i].item())
                    rewards[i] = self.reward_normalizers[i].normalize(rewards[i].item())
                # 如果没有归一化器，使用原始奖励"""

        self.current_step += 1
        self.episode_steps += 1  # 每个环境的episode步数+1

        # 构建信息字典
        info = {
            'step': self.current_step,
            'episode_steps': self.episode_steps.clone(),  # 添加episode步数信息
            'target_positions': self.target_positions.detach().cpu().numpy()
        }

        return obs, rewards, dones, info

    def _sample_random_joint_angles_batch(self) -> torch.Tensor:
        """批量采样随机关节角度"""
        angles = torch.zeros((self.num_envs, 6), device=self.device)

        for i in range(6):
            low, high = self.joint_limits[i]
            angles[:, i] = torch.rand(
                self.num_envs, device=self.device
            ) * (high - low) * 0.5 + low * 0.5  # 使用较小的范围

        return angles

    def _sample_random_target_positions_batch(self) -> torch.Tensor:
        """
        🎯 批量采样随机目标位置（基于config配置）

        从config中读取目标位置范围，便于调整UR10e工作空间
        """
        # 从config中读取目标位置范围
        target_range = self.config['env']['target_range']
        x_range = target_range['x']
        y_range = target_range['y']
        z_range = target_range['z']

        target_positions = torch.zeros((self.num_envs, 3), device=self.device)

        # 在指定范围内随机生成目标位置
        target_positions[:, 0] = torch.rand(self.num_envs, device=self.device) * (x_range[1] - x_range[0]) + x_range[0]
        target_positions[:, 1] = torch.rand(self.num_envs, device=self.device) * (y_range[1] - y_range[0]) + y_range[0]
        target_positions[:, 2] = torch.rand(self.num_envs, device=self.device) * (z_range[1] - z_range[0]) + z_range[0]

        # 调试信息：打印第一个环境的目标位置
        if hasattr(self, 'debug_step') and self.debug_step % 500 == 0:  # 每500步打印一次
            print(f"🎯 目标位置更新: [{target_positions[0].cpu().numpy().tolist()}]")

        return target_positions

    def _sample_target_joint_angles_batch(self) -> torch.Tensor:
        """
        🎯 直接采样随机的目标关节角度，然后用正运动学生成目标位置

        避免复杂的逆运动学计算，确保目标位置是可达的
        """
        target_joint_angles = torch.zeros((self.num_envs, 6), device=self.device)

        # 在工作空间内随机生成关节角度
        for i in range(6):
            low, high = self.joint_limits[i]
            # 使用较小的范围确保在工作空间内
            safe_low = low * 0.5
            safe_high = high * 0.5
            target_joint_angles[:, i] = torch.rand(self.num_envs, device=self.device) * (safe_high - safe_low) + safe_low

        return target_joint_angles

    def _compute_positions_from_joint_angles(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """
        🎯 通过正运动学从关节角度计算末端位置

        Args:
            joint_angles: [num_envs, 6] 关节角度

        Returns:
            positions: [num_envs, 3] 末端位置
        """
        return self._compute_end_effector_positions_batch(joint_angles)

    def _get_states(self) -> torch.Tensor:
        """获取所有环境的当前状态"""
        states = torch.zeros((self.num_envs, self.state_dim), device=self.device)

        # 获取当前关节角度和速度
        current_angles, current_velocities = self._get_joint_angles_and_velocities()

        # 计算末端位置
        current_positions = self._compute_end_effector_positions_batch(current_angles)

        # 🎯 构建状态向量 (18维：当前关节角度6 + 目标关节角度6 + 当前末端位置3 + 目标位置3)
        # [current_angles(6), target_joint_angles(6), current_position(3), target_position(3)]
        states[:, 0:6] = current_angles
        states[:, 6:12] = self.target_joint_angles
        states[:, 12:15] = current_positions
        states[:, 15:18] = self.target_positions

        # 设备一致性检查
        if hasattr(self, '_debug_mode') and self._debug_mode:
            if not check_tensor_devices({'states': states, 'target_positions': self.target_positions, 'target_joint_angles': self.target_joint_angles}, "_get_states"):
                print(f"⚠️ _get_states设备不一致")

        return states

    def _compute_end_effector_positions_batch(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """批量计算末端执行器位置 - 使用完整的UR10e DH参数"""
        # 确保输入张量在正确的设备上
        joint_angles = joint_angles.to(self.device)
        # 使用运动学解算器计算末端位置
        positions = torch.zeros((self.num_envs, 3), device=self.device)

        for i in range(self.num_envs):
            if self.kinematics is not None:
                # 使用运动学解算器（如果可用）
                angles_np = joint_angles[i].detach().cpu().numpy()
                T = self.kinematics.forward_kinematics(angles_np)
                positions[i] = torch.tensor(T[:3, 3], device=self.device)
            else:
                # 使用完整的UR10e DH参数正运动学
                positions[i] = self._forward_kinematics(joint_angles[i])

        return positions

    def _forward_kinematics(self, joint_positions: torch.Tensor) -> torch.Tensor:
        """
        UR10e forward kinematics using all 6 joints (q1-q6) with complete DH parameters.

        Args:
            joint_positions: [6] 关节角度张量

        Returns:
            ee_pos: [3] 末端执行器位置
        """
        import math

        # 保证是 1D 向量 [6]
        joint_positions = joint_positions.view(-1)
        device = joint_positions.device
        dtype = joint_positions.dtype

        # UR10e DH参数 (基于官方规格)
        d = torch.tensor(
            [0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655],
            device=device, dtype=dtype
        )
        a = torch.tensor(
            [0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0],
            device=device, dtype=dtype
        )
        alpha = torch.tensor(
            [math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0],
            device=device, dtype=dtype
        )

        # DH 变换函数
        def dh_transform(theta, d_i, a_i, alpha_i):
            ct = torch.cos(theta)
            st = torch.sin(theta)
            ca = torch.cos(alpha_i)
            sa = torch.sin(alpha_i)

            T = torch.zeros((4, 4), device=device, dtype=dtype)
            T[0, 0] = ct
            T[0, 1] = -st * ca
            T[0, 2] = st * sa
            T[0, 3] = a_i * ct

            T[1, 0] = st
            T[1, 1] = ct * ca
            T[1, 2] = -ct * sa
            T[1, 3] = a_i * st

            T[2, 0] = 0.0
            T[2, 1] = sa
            T[2, 2] = ca
            T[2, 3] = d_i

            T[3, 3] = 1.0
            return T

        # 累积变换
        T_cum = torch.eye(4, device=device, dtype=dtype)
        for i in range(6):
            T_i = dh_transform(joint_positions[i], d[i], a[i], alpha[i])
            T_cum = T_cum @ T_i

        # 返回末端位置
        ee_pos = T_cum[:3, 3]
        return ee_pos

    def _apply_rl_pid_control(self, actions: torch.Tensor):
        """应用RL补偿力矩控制：基础PID力矩 + RL补偿力矩"""
        # 验证动作维度 (现在应该是6维)
        if actions.shape[-1] != 6:
            raise ValueError(f"期望6维力矩补偿动作，得到{actions.shape[-1]}维")

        # 获取当前状态
        current_angles, current_velocities = self._get_joint_angles_and_velocities()
        joint_errors = self.target_joint_angles - current_angles  # [num_envs, 6]

        # 从config获取基础PID参数
        base_kp = self.config['pid_params']['base_gains']['p']  # 基础P增益
        base_kd = self.config['pid_params']['base_gains']['d']  # 基础D增益

        # UR10e力矩限制
        ur10e_torque_limits = [330.0, 330.0, 150.0, 54.0, 54.0, 54.0]
        ur10e_torque_limits_tensor = torch.tensor(ur10e_torque_limits, device=self.device)

        # 🎯 计算基础PID力矩
        pid_torques = torch.zeros_like(actions)  # [num_envs, 6]
        for j in range(6):
            p_term = base_kp[j] * joint_errors[:, j]
            d_term = base_kd[j] * current_velocities[:, j]
            pid_torques[:, j] = p_term - d_term

        # 🤖 RL补偿力矩 (直接输出，已在动作范围内)
        rl_compensation = actions  # [num_envs, 6]

        # ⚡ 总力矩 = PID力矩 + RL补偿
        total_torques = pid_torques + rl_compensation

        # 🔒 力矩限制（确保安全）
        for j in range(6):
            total_torques[:, j] = torch.clamp(
                total_torques[:, j],
                -ur10e_torque_limits_tensor[j],
                ur10e_torque_limits_tensor[j]
            )

        # 批量计算所有环境的控制力矩（Isaac Gym期望CPU张量）
        all_dof_forces = torch.zeros(self.num_envs, 6, 1, device='cpu')

        # 🎯 转换到Isaac Gym格式
        for i in range(self.num_envs):
            for j in range(6):
                all_dof_forces[i, j, 0] = total_torques[i, j].cpu().item()

        # 📊 调试信息（每100步打印一次）
        if hasattr(self, 'debug_step') and self.debug_step % 100 == 0:
            print(f"\n📊 === 步骤 {self.debug_step} 力矩分解调试信息 ===")
            i = 0  # 显示第一个环境
            print(f"🤖 环境{i}:")
            print(f"   关节误差: [{joint_errors[i].cpu().numpy().tolist()}] rad")
            print(f"   🔧 PID力矩:   [{pid_torques[i].cpu().numpy().tolist()}] N⋅m")
            print(f"   🤖 RL补偿:   [{rl_compensation[i].cpu().numpy().tolist()}] N⋅m")
            print(f"   ⚡ 总力矩:   [{total_torques[i].cpu().numpy().tolist()}] N⋅m")

            joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_joint', 'wrist_1', 'wrist_2', 'wrist_3']
            for j, (name, total, limit) in enumerate(zip(joint_names, total_torques[i].cpu().numpy(), ur10e_torque_limits)):
                saturation = abs(total) / limit * 100
                print(f"      {j+1}. {name:12}: {total:7.2f} N⋅m (限制: ±{limit:5.1f}, 饱和度: {saturation:5.1f}%)")

        # 🎯 Isaac Gym官方API：确保力矩张量在CPU上再unwrap（修复设备不匹配）
        # 参考: gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(u))
        try:
            # 确保力矩张量在CPU上（gymtorch.unwrap_tensor需要CPU张量）
            if all_dof_forces.device.type != 'cpu':
                all_dof_forces_cpu = all_dof_forces.cpu()
            else:
                all_dof_forces_cpu = all_dof_forces

            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(all_dof_forces_cpu))
            if hasattr(self, 'debug_step') and self.debug_step % 100 == 0:
                print(f"✅ Isaac Gym力矩设置成功: 形状={all_dof_forces.shape}, 原始设备={all_dof_forces.device}, 传输到CPU")
        except Exception as e:
            print(f"❌ Isaac Gym力矩设置失败: {e}")
            print(f"   力矩张量形状: {all_dof_forces.shape}")
            print(f"   力矩张量设备: {all_dof_forces.device}")
            print(f"   力矩张量类型: {all_dof_forces.dtype}")
            print(f"   力矩范数: {torch.norm(all_dof_forces)}")
    
    def _compute_rewards_batch(self, actions):
        """
        🎯 二次型奖励函数（基于论文设计）

        奖励函数: ρ(e_i, ė_i) = Q_i[1,1]·(e_i)² + Q_i[2,2]·(ė_i)²
        其中 e_i 是位置误差，ė_i 是速度误差
        """
        current_angles, current_vels = self._get_joint_angles_and_velocities()

        # 🎯 关节空间误差计算
        position_errors = self.target_joint_angles - current_angles  # [num_envs, 6]
        velocity_errors = -current_vels  # 目标速度为0，所以误差 = -当前速度 [num_envs, 6]

        # 🎯 二次型奖励函数 (论文公式)
        # ρ(e_i, ė_i) = Q_i[1,1]·(e_i)² + Q_i[2,2]·(ė_i)²
        position_rewards = -torch.sum(self.Q_weights.unsqueeze(0) * position_errors**2, dim=1)  # [num_envs]
        velocity_rewards = -torch.sum(self.Q_velocity_weights.unsqueeze(0) * velocity_errors**2, dim=1)  # [num_envs]

        # 总奖励 = 位置奖励 + 速度奖励
        total_rewards = position_rewards + velocity_rewards
        total_rewards = self.reward_scale*total_rewards

        # 📊 调试信息（每100步打印一次）
        if hasattr(self, 'debug_step') and self.debug_step % 100 == 0:
            avg_position_error = torch.norm(position_errors, dim=1).mean().item()
            avg_velocity_error = torch.norm(velocity_errors, dim=1).mean().item()
            avg_reward = total_rewards.mean().item()

            print(f"📈 步骤{self.debug_step}:")
            print(f"   平均关节位置误差: {avg_position_error:.4f} rad ({avg_position_error*180/3.14159:.1f}°)")
            print(f"   平均关节速度误差: {avg_velocity_error:.4f} rad/s")
            print(f"   平均奖励: {avg_reward:.2f}")
            print(f"   位置奖励分量: {position_rewards.mean().item():.2f}")
            print(f"   速度奖励分量: {velocity_rewards.mean().item():.2f}")

        self.debug_step += 1
        return total_rewards


    def _check_done_batch(self) -> torch.Tensor:
        """检查完成条件（稳定性要求 - 连续100步在目标范围内）"""
        # 获取当前状态
        current_angles, _ = self._get_joint_angles_and_velocities()
        current_positions = self._compute_end_effector_positions_batch(current_angles)

        # 🎯 同时满足关节和位置精度
        joint_errors = self.target_joint_angles - current_angles
        joint_error_norms = torch.norm(joint_errors, dim=1)
        joint_success_threshold = 0.052  # 3度 (3 * π/180 ≈ 0.052弧度)
        joint_success = joint_error_norms < joint_success_threshold

        position_errors = torch.norm(self.target_positions - current_positions, dim=1)
        position_success_threshold = 0.05  # 5cm保持不变
        position_success = position_errors < position_success_threshold

        # 🎯 同时满足关节和位置精度
        success_this_step = joint_success & position_success

        # 更新稳定性计数器
        self.on_goal_count = torch.where(
            success_this_step,
            self.on_goal_count + 1,  # 成功则增加计数
            torch.zeros_like(self.on_goal_count)  # 失败则重置计数器
        )

        # ⏰ 超时条件
        timeout_done = (self.episode_steps + 1) >= self.max_steps

        # 🎯 完成条件：连续成功达到要求步数 OR 超时
        stability_done = self.on_goal_count >= self.stability_required_steps
        done = stability_done | timeout_done

        # 📊 调试信息（每100步打印一次）
        if hasattr(self, 'debug_step') and self.debug_step % 100 == 0:
            joint_success_rate = joint_success.float().mean().item()
            position_success_rate = position_success.float().mean().item()
            combined_success_rate = success_this_step.float().mean().item()
            avg_stability_count = self.on_goal_count.float().mean().item()
            timeout_rate = timeout_done.float().mean().item()

            print(f"🏁 步骤{self.debug_step} Done状态:")
            print(f"   关节成功(3°): {joint_success_rate:.2%}")
            print(f"   位置成功(5cm): {position_success_rate:.2%}")
            print(f"   综合成功: {combined_success_rate:.2%}")
            print(f"   平均稳定性计数: {avg_stability_count:.1f}/{self.stability_required_steps}")
            print(f"   超时: {timeout_rate:.2%}")

        return done

    def _get_dof_state_indices(self, env_idx: int):
        """获取指定环境的DOF状态索引"""
        # 这里需要根据Isaac Gym的具体实现来获取索引
        # 暂时返回占位符
        return torch.arange(env_idx * 6, (env_idx + 1) * 6, device=self.device)

    def _get_joint_angles_and_velocities(self) -> tuple:
        """正确获取所有环境的关节角度和速度"""
        dof_positions = self.dof_states.view(-1, 2)  # [num_envs * 6, 2]
        current_angles_list = []
        current_velocities_list = []
        for i in range(self.num_envs):
            start_idx = i * 6
            env_angles = dof_positions[start_idx:start_idx+6, 0]  # 6个关节的位置
            env_vels = dof_positions[start_idx:start_idx+6, 1]   # 6个关节的速度
            current_angles_list.append(env_angles)
            current_velocities_list.append(env_vels)

        current_angles = torch.stack(current_angles_list)  # [num_envs, 6]
        current_velocities = torch.stack(current_velocities_list)  # [num_envs, 6]

        # 确保张量在正确的设备上
        current_angles = current_angles.to(self.device)
        current_velocities = current_velocities.to(self.device)

        return current_angles, current_velocities

    def get_num_envs(self) -> int:
        """获取环境数量"""
        return self.num_envs

    def get_num_actions(self) -> int:
        """获取动作维度"""
        return self.action_dim

    def get_num_obs(self) -> int:
        """获取观测维度"""
        return self.state_dim

    def _draw_target_sphere(self):
        """绘制红色目标球体（使用Isaac Gym官方gymutil API）"""
        try:
            # 导入gymutil（Isaac Gym官方调试绘制工具）
            from isaacgym import gymutil

            # 创建红色球体几何体（只创建一次）
            if not hasattr(self, '_target_sphere_geom'):
                sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
                sphere_pose = gymapi.Transform(r=sphere_rot)
                # 创建红色线框球体，半径0.05m
                self._target_sphere_geom = gymutil.WireframeSphereGeometry(0.05, 12, 12, sphere_pose, color=(1, 0, 0))

            # 为每个环境绘制目标点
            for i in range(self.num_envs):
                target_pos = self.target_positions[i]

                # 创建目标点的变换（位置）
                sphere_pose = gymapi.Transform()
                sphere_pose.p = gymapi.Vec3(target_pos[0].item(), target_pos[1].item(), target_pos[2].item())
                sphere_pose.r = gymapi.Quat(0, 0, 0, 1)  # 无旋转

                # 使用Isaac Gym官方调试绘制API绘制红色球体
                if hasattr(self, 'viewer') and self.viewer is not None:
                    gymutil.draw_lines(self._target_sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
        except Exception as e:
            # 如果绘制失败，静默处理（避免中断训练）
            if hasattr(self, 'debug_step') and self.debug_step % 1000 == 0:  # 偶尔报告错误
                print(f"⚠️ 目标点绘制失败: {e}")
            pass

    def close(self):
        """关闭环境"""
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
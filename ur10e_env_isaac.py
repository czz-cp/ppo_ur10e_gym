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
        #self.reward_scale = 1e-3  # 你可以后面微调，比如 5e-4, 2e-3 之类

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
        #max_compensation_torque = 30.0  # 30 N⋅m补偿力矩 (约为最大力矩的10%)
        #self.action_space_high = np.array([max_compensation_torque] * 6)  # [τ1, τ2, τ3, τ4, τ5, τ6]
        #self.action_space_low = np.array([-max_compensation_torque] * 6)
        

        # 状态空间 (25维：紧凑位姿误差表示 + 障碍物距离)
        # 状态结构：[关节角6 + 当前位姿7 + 目标位姿7 + 位姿误差2 + dobs3]
        self.state_dim = 25  # 包含紧凑位姿误差表示和dobs
        self.action_dim = 6

        # 🎯 障碍物参数
        self.num_obstacles = 3  # 每个环境3个障碍物
        self.obstacle_radius = 0.025  # 🎯 障碍物半径 2.5cm (直径5cm，按论文建议)

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
        self.stability_required_steps = 5  # 需要连续100步在目标范围内
        self.target_positions = None
        self.target_joint_angles = None  # 🎯 新增：目标关节角度
        self.prev_position_errors = None
        self.prev_joint_errors = None  # 🎯 新增：上次关节角度误差

        # 🔧 修复：显式初始化积分器状态
        self.desired_joint_angles = None  # PD积分器的期望关节角度

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
        self.obstacle_handles = []  # 🎯 新增：障碍物handles
        self.obstacle_positions = []  # 🎯 新增：存储实际障碍物位置

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

            # 🎯 创建球体障碍物
            env_obstacles = []
            env_obstacle_positions = []  # 存储当前环境的障碍物位置
            obstacle_asset_options = gymapi.AssetOptions()
            obstacle_asset_options.fix_base_link = True  # 固定障碍物

            # 创建球体障碍物资产
            self.obstacle_asset = self.gym.create_sphere(
                self.sim, self.obstacle_radius, obstacle_asset_options
            )

            for j in range(self.num_obstacles):
                # 随机采样障碍物位置（在论文工作空间内）
                obstacle_pos = self._sample_obstacle_position()

                obstacle_transform = gymapi.Transform()
                obstacle_transform.p = obstacle_pos

                obstacle_handle = self.gym.create_actor(
                    env, self.obstacle_asset, obstacle_transform, f"obstacle_{i}_{j}"
                )
                env_obstacles.append(obstacle_handle)
                env_obstacle_positions.append([obstacle_pos.x, obstacle_pos.y, obstacle_pos.z])

                # 设置障碍物颜色为红色
                self.gym.set_rigid_body_color(
                    env, obstacle_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                    gymapi.Vec3(1.0, 0.0, 0.0)  # 红色
                )

            self.obstacle_handles.append(env_obstacles)
            self.obstacle_positions.append(env_obstacle_positions)  # 存储每个环境的障碍物位置

        print(f"✅ 创建了 {self.num_envs} 个环境，每个环境有 {self.num_obstacles} 个障碍物")

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

        # 🎯 重置新的误差跟踪变量（用于增强奖励函数）
        if hasattr(self, '_prev_position_errors'):
            delattr(self, '_prev_position_errors')
        if hasattr(self, 'target_orientations'):
            delattr(self, 'target_orientations')

        # 初始化期望关节角度（用于速度控制）
        self.desired_joint_angles = self.start_joint_angles.clone()
        print(f"🔧 Reset: 初始化��望关节角度为起始角度")

        # 🎯 重置稳定性跟踪
        self.on_goal_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # 重置奖励归一化器
        #for normalizer in self.reward_normalizers:
            #normalizer.reset()
        
        for normalizer in self.reward_normalizers:
            if normalizer is not None:
                normalizer.reset()

        # 推进一步
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # 获取初始观测
        obs = self._get_states()

        return obs
    
    def _reset_done_envs(self, dones: torch.Tensor):
        """只重置 dones == True 的那些环境"""
        done_indices = torch.nonzero(dones, as_tuple=False).squeeze(-1)
        if done_indices.numel() == 0:
            return

        # 1) 为这些 env 重新采样起始关节角 & 目标关节角/位置
        new_start_angles = self._sample_random_joint_angles_batch()[done_indices]
        new_target_joint_angles = self._sample_target_joint_angles_batch()[done_indices]
        new_target_positions = self._compute_positions_from_joint_angles(new_target_joint_angles)

        # 2) 写回 DOF 状态
        for env_idx, joint_angles in zip(done_indices, new_start_angles):
            env_idx = int(env_idx.item())
            # 保证长度为 6
            joint_angles = joint_angles.view(-1)
            if joint_angles.numel() != 6:
                if joint_angles.numel() > 6:
                    joint_angles = joint_angles[:6]
                else:
                    pad = torch.zeros(6 - joint_angles.numel(), device=self.device)
                    joint_angles = torch.cat([joint_angles, pad], dim=0)

            start = env_idx * self.num_dofs
            end = (env_idx + 1) * self.num_dofs
            self.dof_states[start:end, 0] = joint_angles.to(self.device)  # 位置
            self.dof_states[start:end, 1] = 0.0                           # 速度置零

        # 3) 更新这些 env 的 target 变量
        self.target_joint_angles[done_indices] = new_target_joint_angles
        self.target_positions[done_indices] = new_target_positions

        # 4) 把 DOF 状态写回 Isaac Gym
        if self.dof_states.device.type != 'cpu':
            dof_states_cpu = self.dof_states.cpu()
        else:
            dof_states_cpu = self.dof_states
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(dof_states_cpu))

        # 5) 为新 episode 稍微稳定几步
        for _ in range(10):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        # 6) 重置这些 env 的内部计数器
        self.episode_steps[done_indices] = 0
        self.on_goal_count[done_indices] = 0
        if self.prev_position_errors is not None:
            self.prev_position_errors[done_indices] = 10.0
        if self.prev_joint_errors is not None:
            self.prev_joint_errors[done_indices] = 10.0

        # 🎯 重置新的误差跟踪变量（用于增强奖励函数）
        if hasattr(self, '_prev_position_errors'):
            self._prev_position_errors[done_indices] = float('inf')
        if hasattr(self, 'target_orientations'):
            # 重新采样完成环境的姿态
            new_orientations = self._sample_random_orientations_batch()[done_indices.cpu().numpy()]
            self.target_orientations[done_indices] = new_orientations

        # 7) 🔧 修复：重置这些环境的desired_joint_angles（关键修复！）
        if self.desired_joint_angles is not None:
            # 获取当前所有环境的关节角度
            current_angles, _ = self._get_joint_angles_and_velocities()
            # 只重置完成的环境
            self.desired_joint_angles[done_indices] = current_angles[done_indices]
            print(f"🔧 Reset {len(done_indices)} 个完成环境的desired_joint_angles为当前角度")

        # 8) 🎯 重置障碍物位置（Domain Randomization - 防止智能体"背答案"）
        if hasattr(self, 'obstacle_positions') and len(self.obstacle_positions) > 0:
            print(f"🎯 重置 {len(done_indices)} 个环境的障碍物位置...")

            for env_idx in done_indices.cpu().tolist():
                if 0 <= env_idx < len(self.obstacle_positions):
                    # 🎯 更新障碍物在Isaac Gym中的位置
                    for obs_idx, obs_handle in enumerate(self.obstacle_handles[env_idx]):
                        # 为每个障碍物单独采样位置
                        new_obstacle_pos = self._sample_obstacle_position()

                        obs_pose = gymapi.Transform()
                        # _sample_obstacle_position() 返回 gymapi.Vec3，直接使用
                        if isinstance(new_obstacle_pos, gymapi.Vec3):
                            obs_pose.p = new_obstacle_pos
                        else:
                            # 如果返回的是tensor，需要转换
                            if hasattr(new_obstacle_pos, '__getitem__'):
                                obs_pose.p = gymapi.Vec3(
                                    new_obstacle_pos[0].item() if hasattr(new_obstacle_pos[0], 'item') else float(new_obstacle_pos[0]),
                                    new_obstacle_pos[1].item() if hasattr(new_obstacle_pos[1], 'item') else float(new_obstacle_pos[1]),
                                    new_obstacle_pos[2].item() if hasattr(new_obstacle_pos[2], 'item') else float(new_obstacle_pos[2])
                                )
                            else:
                                # 备用方案
                                obs_pose.p = gymapi.Vec3(0.4, 0.2, 0.3)
                        # 保持随机旋转 (Isaac Gym四元数不需要手动归一化)
                        obs_pose.r = gymapi.Quat(
                            np.random.uniform(-0.5, 0.5),
                            np.random.uniform(-0.5, 0.5),
                            np.random.uniform(-0.5, 0.5),
                            np.random.uniform(0.5, 1.0)
                        )

                        # 🎯 使用root_state tensor更新方法���参考isaac_gym_manipulator静态障碍物实现）
                        # 计算全局actor索引：robot(0) + target(1) + obstacles(3个)
                        global_actor_idx = env_idx * (2 + self.num_obstacles) + 2 + obs_idx

                        # 刷新root_state tensor
                        self.gym.refresh_actor_root_state_tensor(self.sim)

                        if global_actor_idx < self.root_states.shape[0]:
                            # 直接修改root_state tensor中的位置
                            self.root_states[global_actor_idx, 0:3] = torch.tensor([
                                obs_pose.p.x, obs_pose.p.y, obs_pose.p.z
                            ], device=self.device, dtype=torch.float32)
                            # 设置四元数 (x,y,z,w)
                            self.root_states[global_actor_idx, 3:7] = torch.tensor([
                                obs_pose.r.x, obs_pose.r.y, obs_pose.r.z, obs_pose.r.w
                            ], device=self.device, dtype=torch.float32)
                            # 速度清零
                            self.root_states[global_actor_idx, 7:13] = 0.0

                            # 使用批量更新API (需要CPU tensor)
                            indices_i32 = torch.tensor([global_actor_idx], dtype=torch.int32, device='cpu')
                            # 将root_states移动到CPU进行更新
                            root_states_cpu = self.root_states.cpu()
                            self.gym.set_actor_root_state_tensor_indexed(
                                self.sim,
                                gymtorch.unwrap_tensor(root_states_cpu),
                                gymtorch.unwrap_tensor(indices_i32),
                                1
                            )

                    # 更新内部存储（使用新采样的位置）
                    self.obstacle_positions[env_idx][obs_idx] = [obs_pose.p.x, obs_pose.p.y, obs_pose.p.z]

            # 刷新物理状态以确保障碍物位置更新生效
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            print(f"✅ 障碍物位置重新采样完成")

        # 9) 重置对应的奖励归一化器（如果你还在用的话）
        for env_idx in done_indices.cpu().tolist():
            if (0 <= env_idx < len(self.reward_normalizers)
                    and self.reward_normalizers[env_idx] is not None):
                self.reward_normalizers[env_idx].reset()


    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        执行一步仿真

        
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

        # 执行速度PD控制
        self._apply_velocity_pd_control(actions)

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
        #for i in range(self.num_envs):
        #    if dones[i]:
        #        self.episode_steps[i] = 0  # 重置该环境的episode步数
        
        # ⭐ 对 done 的环境做真正的 reset：重采样起点/目标，写回 dof_states 等
        self._reset_done_envs(dones)

        # 对于被 reset 的环境，把 obs 换成“新 episode 的初始观测”
        if dones.any():
            obs = self._get_states()

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

    """def _sample_random_joint_angles_batch(self) -> torch.Tensor:
        批量采样随机关节角度
        angles = torch.zeros((self.num_envs, 6), device=self.device)

        for i in range(6):
            low, high = self.joint_limits[i]
            angles[:, i] = torch.rand(
                self.num_envs, device=self.device
            ) * (high - low) * 0.5 + low * 0.5  # 使用较小的范围

        return angles"""
    
    def _sample_random_joint_angles_batch(self) -> torch.Tensor:
        """批量采样“可动”的随机关节角度（远离极限和奇异位）"""
        # joint_limits: 形状 [6, 2]，每行 [low, high]
        joint_limits = torch.tensor(self.joint_limits, device=self.device, dtype=torch.float32)  # [6,2]
        low = joint_limits[:, 0]   # [6]
        high = joint_limits[:, 1]  # [6]

        center = (low + high) / 2.0          # 中点
        half_range = (high - low) / 2.0      # 半范围

        # 只用中间 20% 的范围，确保TCP位置在工作空间内
        ratio = 0.2
        noise_range = half_range * ratio     # 每个关节的"活动半径"

        # 随机在 [-noise_range, +noise_range] 内扰动
        # angles 形状 [num_envs, 6]
        noise = (torch.rand(self.num_envs, 6, device=self.device) * 2.0 - 1.0) * noise_range  # [-1,1]*noise_range
        angles = center.unsqueeze(0) + noise  # [1,6] + [num_envs,6] -> [num_envs,6]

        # 进一步限制前三个关节的角度范围，确保TCP在工作空间内
        # shoulder_pan: 限制在±1.0 rad (±57°)
        angles[:, 0] = torch.clamp(angles[:, 0], -1.0, 1.0)
        # shoulder_lift: 限制在[0.8, 2.0] rad (确保TCP有足够高度，手臂向上)
        angles[:, 1] = torch.clamp(angles[:, 1], 0.8, 2.0)
        # elbow: 限制在[-0.5, 0.5] rad (适中的肘部角度)
        angles[:, 2] = torch.clamp(angles[:, 2], -0.5, 0.5)

        # 再保险一点，离上下限各留 10% 的 margin
        margin = 0.1 * (high - low)
        safe_low = low + margin
        safe_high = high - margin

        angles = torch.max(torch.min(angles, safe_high.unsqueeze(0)), safe_low.unsqueeze(0))
        return angles


    
    def _sample_target_joint_angles_batch(self) -> torch.Tensor:
        """
        目标关节角：从球体-圆柱工作空间中采样可达的关节配置

        方法：
        1. 随机采样关节角度配置
        2. 用前向运动学计算末端位置
        3. 检查位置是否在球体-圆柱工作空间内
        4. 如果不在，重新采样（拒绝采样）
        """
        # 确保 start_joint_angles 已经填好
        if not hasattr(self, "start_joint_angles"):
            self.start_joint_angles = self._sample_random_joint_angles_batch()

        target_angles = torch.empty((self.num_envs, 6), device=self.device)

        # 工作空间参数
        sphere_radius = 0.85  # 球体半径
        cylinder_radius = 0.30  # 圆柱半径
        max_attempts = 100  # 每个环境的最大采样尝试次数

        for i in range(self.num_envs):
            sampled = False

            for attempt in range(max_attempts):
                # 随机采样关节角度（在关节限制范围内）
                random_angles = self._sample_random_joint_angles_batch_single()

                # 用前向运动学计算末端位置
                end_effector_pos = self._compute_end_effector_positions_batch(random_angles.unsqueeze(0))[0]

                # 检查是否在工作空间内
                if self._is_position_in_workspace(end_effector_pos, sphere_radius, cylinder_radius):
                    target_angles[i] = random_angles
                    sampled = True
                    break

            # 如果采样失败，使用基于起始角的小偏移
            if not sampled:
                # 回退到原始方法：在起始角基础上加小偏移
                noise = torch.empty(6, device=self.device)
                noise[:3].uniform_(-0.3, 0.3)   # 前三个关节 ±0.3rad
                noise[3:].uniform_(-0.5, 0.5)   # 手腕关节 ±0.5rad

                fallback_angles = self.start_joint_angles[i] + noise

                # 应用关节限制
                low = torch.tensor(self.joint_limits[:, 0], device=self.device)
                high = torch.tensor(self.joint_limits[:, 1], device=self.device)
                fallback_angles = torch.clamp(fallback_angles, low, high)

                target_angles[i] = fallback_angles
                if attempt == max_attempts - 1:
                    print(f"⚠️ 环境 {i} 工作空间采样失败，使用回退方法")

        return target_angles

    def _sample_random_joint_angles_batch_single(self) -> torch.Tensor:
        """
        为单个环境采样随机关节角度

        Returns:
            angles: [6] 关节角度张量
        """
        angles = torch.empty(6, device=self.device)

        # 根据UR10e关节限制采样
        # UR10e关节限制（弧度）：[-2π, 2π], [-2π, 2π], [-π, π], [-2π, 2π], [-2π, 2π], [-2π, 2π]
        joint_limits = [
            (-2*np.pi, 2*np.pi),   # Base joint
            (-2*np.pi, 2*np.pi),   # Shoulder joint
            (-np.pi, np.pi),       # Elbow joint
            (-2*np.pi, 2*np.pi),   # Wrist 1 joint
            (-2*np.pi, 2*np.pi),   # Wrist 2 joint
            (-2*np.pi, 2*np.pi)    # Wrist 3 joint
        ]

        for j, (low, high) in enumerate(joint_limits):
            angles[j] = torch.rand(1, device=self.device).item() * (high - low) + low

        return angles

    def _is_position_in_workspace(self, position: torch.Tensor, sphere_radius: float, cylinder_radius: float) -> bool:
        """
        检查位置是否在球体-圆柱工作空间内

        Args:
            position: [3] 位置张量 [x, y, z]
            sphere_radius: 球体半径
            cylinder_radius: 圆柱半径

        Returns:
            bool: 是否在工作空间内
        """
        x, y, z = position[0].item(), position[1].item(), position[2].item()

        # 检查是否在球体内
        distance_from_origin = np.sqrt(x**2 + y**2 + z**2)
        if distance_from_origin > sphere_radius:
            return False

        # 检查是否在圆柱外
        radial_distance = np.sqrt(x**2 + y**2)
        if radial_distance <= cylinder_radius:
            return False

        # 检查z坐标不要太低（避免地面碰撞）
        if z <= 0.1:  # z > 0.1m
            return False

        return True


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

    

        # Check if any cached state variables are already NaN
        if hasattr(self, 'target_positions') and self.target_positions is not None:
            if torch.isnan(self.target_positions).any():
                print(f"🚨 [EMERGENCY] target_positions already contains NaN!")
                print(f"   target_positions: {self.target_positions}")
                self.target_positions = torch.zeros_like(self.target_positions) + 0.5  # Emergency fallback

        if hasattr(self, 'target_orientations') and self.target_orientations is not None:
            if torch.isnan(self.target_orientations).any():
                print(f"🚨 [EMERGENCY] target_orientations already contains NaN!")
                print(f"   target_orientations: {self.target_orientations}")
                # Reset to unit quaternions
                self.target_orientations = torch.zeros_like(self.target_orientations)
                self.target_orientations[:, 0] = 1.0  # w = 1, x=y=z = 0

        # 获取当前关节角度和速度
        current_angles, current_velocities = self._get_joint_angles_and_velocities()

        # 🔍 DEBUG: Check for NaN in joint angles
        if torch.isnan(current_angles).any():
            print(f"🚨 [DEBUG] NaN detected in current_angles!")
            print(f"   current_angles: {current_angles}")
            # 🚨 EMERGENCY FIX - Replace NaN with safe values
            current_angles = torch.zeros_like(current_angles)
            print(f"🚨 [EMERGENCY] Replaced NaN angles with zeros!")

        # 计算当前末端位姿（位置 + 姿态）
        current_positions = self._compute_end_effector_positions_batch(current_angles)
        current_orientations = self._compute_end_effector_orientations_batch(current_angles)

        # 🔍 DEBUG: Check for NaN in current poses
        if torch.isnan(current_positions).any():
            print(f"🚨 [DEBUG] NaN detected in current_positions!")
            print(f"   current_positions: {current_positions}")
        if torch.isnan(current_orientations).any():
            print(f"🚨 [DEBUG] NaN detected in current_orientations!")
            print(f"   current_orientations: {current_orientations}")

        # 🎯 获取目标末端位姿（位置 + 姿态）
        if not hasattr(self, "target_orientations"):
            # 懒初始化：采样随机目标姿态
            self.target_orientations = self._sample_random_orientations_batch()

        # 🎯 按论文附录A.2计算几何位姿误差
        pose_errors = torch.zeros((self.num_envs, 2), device=self.device)

        # 论文参数：轴长度ℓ（0.1m）和姿态权重λ_ori
        ell = 0.1  # 轴长度
        lambda_ori = float(self.config.get('trajectory_tracking', {}).get('lambda_ori', 0.5))

        for i in range(self.num_envs):
            # 当前位姿：位置p_e, 姿态q_e
            current_pos = current_positions[i]  # [3]
            current_quat = current_orientations[i]  # [w, x, y, z]
            current_R = self._quaternion_to_rotation_matrix(current_quat)  # [3, 3]

            # 目标位姿：位置p_t, 姿态q_t
            target_pos = self.target_positions[i]  # [3]
            target_quat = self.target_orientations[i]  # [w, x, y, z]
            target_R = self._quaternion_to_rotation_matrix(target_quat)  # [3, 3]

            # 定义单位向量
            x_hat = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            y_hat = torch.tensor([0.0, 1.0, 0.0], device=self.device)

            # 当前位姿下的3个点
            P_e0 = current_pos  # p_e
            P_e1 = current_pos + current_R @ (ell * x_hat)  # p_e + R_e * ℓ * x̂
            P_e2 = current_pos + current_R @ (ell * y_hat)  # p_e + R_e * ℓ * ŷ

            # 目标位姿下的3个点
            P_t0 = target_pos   # p_t
            P_t1 = target_pos + target_R @ (ell * x_hat)   # p_t + R_t * ℓ * x̂
            P_t2 = target_pos + target_R @ (ell * y_hat)   # p_t + R_t * ℓ * ŷ

            # 计算几何误差 e_shape = Σ_k ||P_e,k - P_t,k||²
            shape_error = (torch.norm(P_e0 - P_t0) ** 2 +
                          torch.norm(P_e1 - P_t1) ** 2 +
                          torch.norm(P_e2 - P_t2) ** 2)

            # 计算姿态误差 θ = 2 * arccos(|Δq_w|)
            delta_q = self._quaternion_multiply(target_quat, self._quaternion_inverse(current_quat))
            delta_q_w = delta_q[0]  # w分量
            theta = 2 * torch.arccos(torch.clamp(torch.abs(delta_q_w), 0.0, 1.0))

            # 组合误差向量 e = [e_shape, λ_ori * θ]
            pose_errors[i, 0] = shape_error
            pose_errors[i, 1] = lambda_ori * theta

        # 🎯 计算障碍物距离 dobs (批处理版本)
        dobs = self._compute_obstacle_distances_batch(current_angles)  # [num_envs, 3]

        # ���� DEBUG: Check for NaN in dobs and target values
        if torch.isnan(dobs).any():
            print(f"🚨 [DEBUG] NaN detected in dobs!")
            print(f"   dobs: {dobs}")
            print(f"   current_angles: {current_angles}")
        if torch.isnan(self.target_positions).any():
            print(f"🚨 [DEBUG] NaN detected in target_positions!")
            print(f"   target_positions: {self.target_positions}")
        if torch.isnan(self.target_orientations).any():
            print(f"🚨 [DEBUG] NaN detected in target_orientations!")
            print(f"   target_orientations: {self.target_orientations}")
        if torch.isnan(pose_errors).any():
            print(f"🚨 [DEBUG] NaN detected in pose_errors!")
            print(f"   pose_errors: {pose_errors}")

        # 🎯 新的状态向量 q_t = [关节角6 + 当前位姿7 + 目标位姿7 + 误差2 + dobs3]
        # 总维度：6 + 7 + 7 + 2 + 3 = 25维
        # 状态结构：[current_angles(6), current_pose(7), target_pose(7), pose_error(2), dobs(3)]
        states[:, 0:6] = current_angles                           # q_t: 当前6个关节角
        states[:, 6:9] = current_positions                        # p_e: 当前位置(3)
        states[:, 9:13] = current_orientations                     # p_e: 当前姿态(4)
        states[:, 13:16] = self.target_positions                  # p_t: 目标位置(3)
        states[:, 16:20] = self.target_orientations                # p_t: 目标姿态(4)
        states[:, 20:22] = pose_errors                             # error: (Dϕ+Dθ+Dψ, Δθ)
        states[:, 22:25] = dobs                                    # dobs: 到3个障碍物的最小距离

        # 🔍 FINAL DEBUG: Check final state vector for NaN values
        if torch.isnan(states).any():
            print(f"🚨 [DEBUG] NaN detected in final states!")
            nan_indices = torch.isnan(states).nonzero()
            print(f"   Total NaN values: {nan_indices.shape[0]}")
            for idx in nan_indices[:5]:  # Show first 5 NaN values
                env_idx, dim_idx = idx[0].item(), idx[1].item()
                print(f"   Env {env_idx}, Dim {dim_idx}: NaN")
                if dim_idx >= 22:  # dobs dimension
                    obs_idx = dim_idx - 22
                    print(f"      -> DOBS[{obs_idx}] for env {env_idx}: {dobs[env_idx, obs_idx]}")
                elif dim_idx >= 20:  # pose error dimension
                    error_idx = dim_idx - 20
                    print(f"      -> PoseError[{error_idx}] for env {env_idx}: {pose_errors[env_idx, error_idx]}")
                elif dim_idx >= 13:  # target dimension
                    target_idx = dim_idx - 13
                    if target_idx < 3:
                        print(f"      -> TargetPos[{target_idx}] for env {env_idx}: {self.target_positions[env_idx, target_idx]}")
                    else:
                        ori_idx = target_idx - 3
                        print(f"      -> TargetOri[{ori_idx}] for env {env_idx}: {self.target_orientations[env_idx, ori_idx]}")
                elif dim_idx >= 6:  # current dimension
                    current_idx = dim_idx - 6
                    if current_idx < 3:
                        print(f"      -> CurrentPos[{current_idx}] for env {env_idx}: {current_positions[env_idx, current_idx]}")
                    else:
                        ori_idx = current_idx - 3
                        print(f"      -> CurrentOri[{ori_idx}] for env {env_idx}: {current_orientations[env_idx, ori_idx]}")
                else:  # joint angle dimension
                    print(f"      -> JointAngle[{dim_idx}] for env {env_idx}: {current_angles[env_idx, dim_idx]}")
            # Stop training if NaN detected
            raise ValueError("NaN values detected in state vector!")

        # 更新状态维度
        self.state_dim = 25

        # 设备一致性检查
        if hasattr(self, '_debug_mode') and self._debug_mode:
            if not check_tensor_devices({
                'states': states,
                'target_positions': self.target_positions,
                'target_orientations': self.target_orientations
            }, "_get_states"):
                print(f"⚠️ _get_states设备不一致")
        
        states = torch.nan_to_num(states, nan=0.0, posinf=1e3, neginf=-1e3)
        states = torch.clamp(states, -1e3, 1e3)


        return states

    """def _compute_end_effector_positions_batch(self, joint_angles: torch.Tensor) -> torch.Tensor:
        批量计算末端执行器位置 - 使用完整的UR10e DH参数
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

        return positions"""
    
    def _compute_end_effector_positions_batch(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """批量计算末端执行器位置"""
        # joint_angles: [B, 6]，B 可以是 num_envs，也可以是 len(done_indices)

        joint_angles = joint_angles.to(self.device)
        batch_size = joint_angles.shape[0]

        positions = torch.zeros((batch_size, 3), device=self.device)

        for i in range(batch_size):
            if self.kinematics is not None:
                angles_np = joint_angles[i].detach().cpu().numpy()
                T = self.kinematics.forward_kinematics(angles_np)
                positions[i] = torch.tensor(T[:3, 3], device=self.device)
            else:
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

    def _forward_kinematics_with_orientation(self, joint_positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        UR10e forward kinematics with orientation (position + rotation)

        Args:
            joint_positions: [6] 关节角度张量

        Returns:
            ee_pos: [3] 末端执行器位置
            ee_quat: [4] 末端执行器姿态（四元数 [w, x, y, z]）
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

        # 提取位置
        ee_pos = T_cum[:3, 3]

        # 提取旋转矩阵
        R = T_cum[:3, :3]

        # 旋转矩阵转四元数
        ee_quat = self._rotation_matrix_to_quaternion(R)

        return ee_pos, ee_quat

    def _rotation_matrix_to_quaternion(self, R: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        trace = torch.trace(R)

        if trace > 0:
            S = torch.sqrt(trace + 1.0 + eps) * 2.0
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / (S + eps)
            qy = (R[0, 2] - R[2, 0]) / (S + eps)
            qz = (R[1, 0] - R[0, 1]) / (S + eps)
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2] + eps) * 2.0
            qw = (R[2, 1] - R[1, 2]) / (S + eps)
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / (S + eps)
            qz = (R[0, 2] + R[2, 0]) / (S + eps)
        elif R[1, 1] > R[2, 2]:
            S = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2] + eps) * 2.0
            qw = (R[0, 2] - R[2, 0]) / (S + eps)
            qx = (R[0, 1] + R[1, 0]) / (S + eps)
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / (S + eps)
        else:
            S = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1] + eps) * 2.0
            qw = (R[1, 0] - R[0, 1]) / (S + eps)
            qx = (R[0, 2] + R[2, 0]) / (S + eps)
            qy = (R[1, 2] + R[2, 1]) / (S + eps)
            qz = 0.25 * S

        quat = torch.stack([qw, qx, qy, qz])
        quat = quat / torch.clamp(torch.norm(quat), min=eps)
        return quat


    def _sample_random_orientations_batch(self) -> torch.Tensor:
        """
        批量采样随机目标姿态（四元数格式）

        Returns:
            orientations: [num_envs, 4] 四元数 [w, x, y, z]
        """
        orientations = torch.zeros((self.num_envs, 4), device=self.device)

        for i in range(self.num_envs):
            # 生成随机旋转轴
            # 使用球坐标均匀采样单位球面 - 纯tensor实现
            theta = torch.rand(1, device=self.device) * 2 * torch.pi  # 方位角 [0, 2π]
            phi = torch.acos(1 - 2 * torch.rand(1, device=self.device))  # 极角 [0, π]

            # 旋转轴 - 保持tensor计算
            axis_x = torch.sin(phi) * torch.cos(theta)
            axis_y = torch.sin(phi) * torch.sin(theta)
            axis_z = torch.cos(phi)
            axis = torch.cat([axis_x, axis_y, axis_z])  # 直接拼接为tensor

            # 随机旋转角度 [0, π] - 保持tensor
            angle = torch.rand(1, device=self.device) * torch.pi

            # 旋转轴-角转四元数 - 纯tensor计算
            half_angle = angle / 2
            w = torch.cos(half_angle)
            xyz = axis * torch.sin(half_angle)

            # 直接设置到数组，避免不必要的转换
            orientations[i, 0] = w
            orientations[i, 1:4] = xyz

        return orientations

    def _quaternion_distance(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        计算两个四元数之间的距离（最小旋转角度）

        Args:
            q1, q2: 四元数 [w, x, y, z]

        Returns:
            四元数距离（0到π之间）
        """
        # 确保四元数归一化
        q1 = q1 / torch.norm(q1)
        q2 = q2 / torch.norm(q2)

        q1 = q1 / torch.clamp(torch.norm(q1), min=1e-8)
        q2 = q2 / torch.clamp(torch.norm(q2), min=1e-8)

        # 计算点积
        dot_product = torch.dot(q1, q2).clamp(-1.0, 1.0)

        # 四元数距离 = arccos(|dot_product|)
        distance = torch.acos(torch.abs(dot_product))

        return distance

    def _quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        四元数乘法 q1 ⊗ q2

        Args:
            q1, q2: 四元数 [w, x, y, z]

        Returns:
            result: 四元数乘法结果 [w, x, y, z]
        """
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return torch.tensor([w, x, y, z], device=q1.device, dtype=q1.dtype)

    def _quaternion_inverse(self, q: torch.Tensor) -> torch.Tensor:
        """
        四元数求逆（对于单位四元数等于共轭）

        Args:
            q: 四元数 [w, x, y, z]

        Returns:
            inverse: 四元数的逆 [w, x, y, z]
        """
        # 对于单位四元数，逆等于共轭 [w, -x, -y, -z]
        return torch.tensor([q[0], -q[1], -q[2], -q[3]], device=q.device, dtype=q.dtype)

    def _quaternion_to_rotation_matrix(self, quat: torch.Tensor) -> torch.Tensor:
        """
        将四元数转换为旋转矩阵

        Args:
            quat: 四元数 [w, x, y, z]

        Returns:
            R: 3x3旋转矩阵
        """
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]

        # 四元数归一化
        quat_norm = torch.sqrt(w**2 + x**2 + y**2 + z**2)
        quat_norm = torch.clamp(quat_norm, min=1e-8)
        w, x, y, z = w/quat_norm, x/quat_norm, y/quat_norm, z/quat_norm

        # 构建旋转矩阵
        R = torch.zeros((3, 3), device=quat.device, dtype=quat.dtype)

        R[0, 0] = 1 - 2*(y**2 + z**2)
        R[0, 1] = 2*(x*y - z*w)
        R[0, 2] = 2*(x*z + y*w)

        R[1, 0] = 2*(x*y + z*w)
        R[1, 1] = 1 - 2*(x**2 + z**2)
        R[1, 2] = 2*(y*z - x*w)

        R[2, 0] = 2*(x*z - y*w)
        R[2, 1] = 2*(y*z + x*w)
        R[2, 2] = 1 - 2*(x**2 + y**2)

        return R

    def _rotation_matrix_to_axis_angle(self, R: torch.Tensor) -> torch.Tensor:
        """
        将旋转矩阵转换为轴角表示

        Args:
            R: 3x3旋转矩阵

        Returns:
            axis_angle: 3D轴角向量 [rx, ry, rz]
        """
        # 使用Rodrigues公式转换
        angle = torch.acos(torch.clamp((torch.trace(R) - 1) / 2, -1.0, 1.0))

        if angle < 1e-6:
            # 如果角度很小，返回零向量
            return torch.zeros(3, device=R.device, dtype=R.dtype)

        # 计算旋转轴
        rx = R[2, 1] - R[1, 2]
        ry = R[0, 2] - R[2, 0]
        rz = R[1, 0] - R[0, 1]

        axis = torch.tensor([rx, ry, rz], device=R.device, dtype=R.dtype)
        axis = axis / (2 * torch.sin(angle))

        # 轴角向量
        axis_angle = angle * axis

        return axis_angle

    def _compute_end_effector_orientations_batch(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """
        使用真实运动学计算末端执行器姿态的批处理版本

        Args:
            joint_angles: 关节角度 (num_envs, 6)

        Returns:
            末端执行器姿态 (num_envs, 4) 四元数格式 [w, x, y, z]
        """
        num_envs = joint_angles.shape[0]
        orientations = torch.zeros((num_envs, 4), device=self.device)

        for i in range(num_envs):
            try:
                # 使用扩展的正向运动学函数计算位置和姿态
                _, quat = self._forward_kinematics_with_orientation(joint_angles[i])
                orientations[i] = quat
            except Exception as e:
                print(f"⚠️ 姿态计算失败，使用单位四元数: {e}")
                # 使用单位四元数作为默认值
                orientations[i] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        return orientations

    def _apply_velocity_pd_control(self, normalized_velocities: torch.Tensor):
        """
        应用基于速度的PD控制：
        1. 将归一化速度[-1,1]转换为物理速度
        2. 积分得到期望关节角度
        3. 应用PD控制生成力矩
        4. 强制执行力矩限制
        """
        # 确保输入是2D tensor: [num_envs, 6]
        if normalized_velocities.ndim == 1:
            normalized_velocities = normalized_velocities.unsqueeze(0)  # [6] -> [1, 6]

        # 验证动作维度
        if normalized_velocities.shape[-1] != 6:
            raise ValueError(f"期望6维归一化速度，得到{normalized_velocities.shape[-1]}维")

        # 检查归一化速度是否在范围内
        if not torch.all((normalized_velocities >= -1.0) & (normalized_velocities <= 1.0)):
            print(f"⚠️ 归一化速度超出[-1,1]范围: min={normalized_velocities.min().item():.3f}, max={normalized_velocities.max().item():.3f}")
            normalized_velocities = torch.clamp(normalized_velocities, -1.0, 1.0)

        # 获取当前状态
        current_angles, current_velocities = self._get_joint_angles_and_velocities()

        # 初始化期望关节角度（第一次调用时）
        if not hasattr(self, 'desired_joint_angles') or self.desired_joint_angles is None:
            self.desired_joint_angles = current_angles.clone()
            print(f"🔧 初始化期望关节角度: {self.desired_joint_angles[0].detach().cpu().numpy()}")

        # 1. 速度反归一化：[-1,1] -> 物理速度范围
        if not hasattr(self, 'velocity_limits_tensor'):
            # 如果没有在子类中定义，使用默认值
            self.velocity_limits_tensor = torch.tensor([2.094, 2.094, 3.142, 3.142, 3.142, 3.142], device=self.device)

        physical_velocities = normalized_velocities * self.velocity_limits_tensor  # [num_envs, 6]

        # 2. 积分得到期望关节角度 q_des(t+1) = clamp(q_des(t) + q̇_cmd * dt, joint_limits)
        dt = self.config['env']['dt']  # 0.01s
        self.desired_joint_angles = self.desired_joint_angles + physical_velocities * dt

        # 关节限制（如果存在）
        if hasattr(self, 'joint_lower_limits_tensor') and hasattr(self, 'joint_upper_limits_tensor'):
            self.desired_joint_angles = torch.clamp(
                self.desired_joint_angles,
                self.joint_lower_limits_tensor,
                self.joint_upper_limits_tensor
            )

        # 3. PD控制律：τ = Kp * (q_des - q) + Kd * (-qdot)
        # 从config获取PD增益，如果没有则使用默认值
        if 'pid_params' in self.config and 'base_gains' in self.config['pid_params']:
            kp_gains = self.config['pid_params']['base_gains']['p']
            kd_gains = self.config['pid_params']['base_gains']['d']
        else:
            # 默认PD增益（针对速度控制优化）
            kp_gains = [1000.0, 1000.0, 800.0, 400.0, 200.0, 100.0]
            kd_gains = [50.0, 50.0, 30.0, 20.0, 10.0, 5.0]

        kp_tensor = torch.tensor(kp_gains, device=self.device)
        kd_tensor = torch.tensor(kd_gains, device=self.device)

        # 计算PD力矩
        position_errors = self.desired_joint_angles - current_angles  # [num_envs, 6]
        pd_torques = kp_tensor * position_errors - kd_tensor * current_velocities  # [num_envs, 6]

        # 4. 力矩限制（UR10e规格）
        ur10e_torque_limits = [330.0, 330.0, 150.0, 54.0, 54.0, 54.0]
        ur10e_torque_limits_tensor = torch.tensor(ur10e_torque_limits, device=self.device)

        total_torques = torch.clamp(
            pd_torques,
            -ur10e_torque_limits_tensor,
            ur10e_torque_limits_tensor
        )

        # 5. 转换到Isaac Gym格式并应用
        # Isaac Gym期望CPU张量 [num_envs, 6, 1]
        all_dof_forces = torch.zeros(self.num_envs, 6, 1, device='cpu')
        for i in range(self.num_envs):
            for j in range(6):
                all_dof_forces[i, j, 0] = total_torques[i, j].detach().cpu()

        # 应用到仿真
        try:
            if all_dof_forces.device.type != 'cpu':
                all_dof_forces_cpu = all_dof_forces.cpu()
            else:
                all_dof_forces_cpu = all_dof_forces
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(all_dof_forces_cpu))
        except Exception as e:
            print(f"❌ Isaac Gym力矩设置失败: {e}")
            print(f"   力矩张量形状: {all_dof_forces.shape}")
            print(f"   力矩张量设备: {all_dof_forces.device}")

        # 调试信息
        """if hasattr(self, 'debug_step') and self.debug_step % 100 == 0:
            print(f"\n🎯 === 步骤 {self.debug_step} 速度PD控制调试信息 ===")
            i = 0  # 显示第一个环境
            print(f"🤖 环境{i}:")
            print(f"   归一化速度: [{normalized_velocities[i].detach().cpu().numpy()}]")
            print(f"   物理速度:   [{physical_velocities[i].detach().cpu().numpy()}] rad/s")
            print(f"   当前角度:   [{current_angles[i].detach().cpu().numpy()}] rad")
            print(f"   期望角度:   [{self.desired_joint_angles[i].detach().cpu().numpy()}] rad")
            print(f"   位置误差:   [{position_errors[i].detach().cpu().numpy()}] rad")
            print(f"   PD力矩:     [{pd_torques[i].detach().cpu().numpy()}] N⋅m")
            print(f"   限制后力矩: [{total_torques[i].detach().cpu().numpy()}] N⋅m")

            joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_joint', 'wrist_1', 'wrist_2', 'wrist_3']
            for j, (name, total, limit) in enumerate(zip(joint_names, total_torques[i].detach().cpu().numpy(), ur10e_torque_limits)):
                saturation = abs(total) / limit * 100
                print(f"      {j+1}. {name:12}: {total:7.2f} N⋅m (限制: ±{limit:5.1f}, 饱和度: {saturation:5.1f}%)")"""

    def _compute_rewards_batch(self, actions):
        """
        🎯 论文奖励函数：轨迹跟踪 + 障碍物避免

        根据论文公式：
        r = -ω1*e² - log(e² + τ) - ω2*ψ_sum

        其中：
        - e: 几何位姿误差（附录A.2）
        - τ: 小常数防止log(0)
        - ψ_sum: 障碍物避免项（所有障碍物的ψ函数之和）
        """
        # 确保actions是2D tensor
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)  # [6] -> [1, 6]

        # 1. 获取当前末端执行器位置
        current_angles, current_vels = self._get_joint_angles_and_velocities()
        current_positions = self._compute_end_effector_positions_batch(current_angles)  # [N, 3]

        # 2. 计算位置误差
        position_errors = torch.norm(self.target_positions - current_positions, dim=1)  # [N]

        # 🎯 计算姿态误差
        if not hasattr(self, "target_orientations"):
            # 懒初始化：采样随机目标姿态
            self.target_orientations = self._sample_random_orientations_batch()

        # 计算当前姿态
        current_orientations = self._compute_end_effector_orientations_batch(current_angles)  # [N, 4]

        # 计算姿态误差（四元数距离）
        orientation_errors = torch.zeros(self.num_envs, device=self.device)
        for i in range(self.num_envs):
            orientation_errors[i] = self._quaternion_distance(
                current_orientations[i], self.target_orientations[i]
            )

        # 3. 初始化误差跟踪变量（用于进步奖励）
        #if not hasattr(self, "_prev_position_errors"):
        #    self._prev_position_errors = torch.full((self.num_envs,), float('inf'), device=self.device)

        # 4. 使用轨迹跟踪环境相同的奖励函数参数
        w1 = self.trajectory_config.get("w1", 0.001) if hasattr(self, 'trajectory_config') else 0.001
        lambda_ori = self.trajectory_config.get("lambda_ori", 0.7) if hasattr(self, 'trajectory_config') else 0.5
        tau = self.trajectory_config.get("log_tau", 0.0001) if hasattr(self, 'trajectory_config') else 0.1

        # 🎯 5. 使用论文附录A.2的几何位姿误差计算奖励函数
        rewards = torch.zeros(self.num_envs, device=self.device)

        # 论文参数：轴长度ℓ（0.1m）
        ell = 0.1

        # 🎯 论文奖励函数参数（完全按照论文设置）
        w1 = 1e-3  # ω1 = 10^-3
        tau = 1e-4  # τ = 10^-4
        w2 = 0.1   # ω2 = 0.1
        dmax = 0.08  # d_max = 0.08m

        for i in range(self.num_envs):
            # 当前位姿：位置p_e, 姿态q_e
            current_pos = current_positions[i]  # [3]
            current_quat = current_orientations[i]  # [w, x, y, z]
            current_R = self._quaternion_to_rotation_matrix(current_quat)  # [3, 3]

            # 目标位姿：位置p_t, 姿态q_t
            target_pos = self.target_positions[i]  # [3]
            target_quat = self.target_orientations[i]  # [w, x, y, z]
            target_R = self._quaternion_to_rotation_matrix(target_quat)  # [3, 3]

            # 定义单位向量
            x_hat = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            y_hat = torch.tensor([0.0, 1.0, 0.0], device=self.device)

            # 当前位姿下的3个点
            P_e0 = current_pos  # p_e
            P_e1 = current_pos + current_R @ (ell * x_hat)  # p_e + R_e * ℓ * x̂
            P_e2 = current_pos + current_R @ (ell * y_hat)  # p_e + R_e * ℓ * ŷ

            # 目标位姿下的3个点
            P_t0 = target_pos   # p_t
            P_t1 = target_pos + target_R @ (ell * x_hat)   # p_t + R_t * ℓ * x̂
            P_t2 = target_pos + target_R @ (ell * y_hat)   # p_t + R_t * ℓ * ŷ

            # 🎯 计算shape_error（论文中的几何位置误差）
            shape_error = (torch.norm(P_e0 - P_t0) ** 2 +
                          torch.norm(P_e1 - P_t1) ** 2 +
                          torch.norm(P_e2 - P_t2) ** 2)

            # 计算姿态误差 θ = 2 * arccos(|Δq_w|)
            delta_q = self._quaternion_multiply(target_quat, self._quaternion_inverse(current_quat))
            delta_q_w = delta_q[0]  # w分量
            theta = 2 * torch.arccos(torch.clamp(torch.abs(delta_q_w), 0.0, 1.0))

            # 🎯 按照论文：||e||² = shape_error + θ²
            e2 = shape_error + theta * theta  # ✅ ||e||²（直接加和，不加权）

            # ✅ 计算归一化形状误差（用于奖励计算）
            # 这里使用当前误差与之前误差的归一化差值
            error_norm_curr = shape_error
            error_norm_prev = self.prev_error_norm[i] if hasattr(self, 'prev_error_norm') else shape_error
            e_shape = (error_norm_curr - error_norm_prev) / (error_norm_curr + error_norm_prev + 1e-8)

            # 🔍 DEBUG: Check for potential NaN sources before computing reward
            if torch.isnan(e_shape):
                print(f"🚨 [REWARD DEBUG] NaN in e_shape for env {i}!")
                print(f"   P_e0: {P_e0}, P_t0: {P_t0}")
                print(f"   P_e1: {P_e1}, P_t1: {P_t1}")
                print(f"   P_e2: {P_e2}, P_t2: {P_t2}")
                e_shape = 0.0  # Fallback

            if torch.isnan(theta):
                print(f"🚨 [REWARD DEBUG] NaN in theta for env {i}!")
                print(f"   delta_q: {delta_q}, delta_q_w: {delta_q_w}")
                theta = 0.0  # Fallback

            e = e_shape + lambda_ori * theta  # 综合位置 + 姿态误差
            e_sq = e * e  # e²

            # 🔍 DEBUG: Check e_sq before log
            if torch.isnan(e_sq):
                print(f"🚨 [REWARD DEBUG] NaN in e_sq for env {i}!")
                print(f"   e_shape: {e_shape}, theta: {theta}")
                print(f"   e: {e}, lambda_ori: {lambda_ori}")
                e_sq = 0.0  # Fallback

            if e_sq < 0:
                print(f"🚨 [REWARD DEBUG] Negative e_sq for env {i}: {e_sq}")
                e_sq = 0.0  # Fallback

            # 🎯 根据论文公式：R(s,a) = -[ω1 * e² + ln(e² + τ) + ω2*ψ_sum]
            # 注意：当前reward_i只包含轨迹跟踪部分，障碍物惩罚在后面统一减去
            log_arg = e_sq + tau
            if torch.isnan(torch.log(log_arg)):
                print(f"🚨 [REWARD DEBUG] NaN in log({log_arg}) for env {i}!")
                print(f"   e_sq: {e_sq}, tau: {tau}")
                log_term = 0.0  # Fallback
            else:
                log_term = torch.log(log_arg)

            reward_i = -(w1 * e_sq + log_term)

            # 🔍 DEBUG: Final reward check
            if torch.isnan(reward_i):
                print(f"🚨 [REWARD DEBUG] NaN in final reward for env {i}!")
                print(f"   w1: {w1}, e_sq: {e_sq}, log_term: {log_term}")
                reward_i = 0.0  # Fallback

            rewards[i] = reward_i

        # 🎯 添加障碍物避免项 ψ_sum
        # 根据论文：ψ_sum = Σ_i Σ_j ψ(d_obs(i,j))，其中d是障碍物到link的距离
        w2 = 0.1  # 障碍物避免权重（可调参数）
        tau = 1e-4  # 🎯 论文指定的τ值
        psi_sum = torch.zeros(self.num_envs, device=self.device)

        for i in range(self.num_envs):
            # 计算当前关节配置的障碍物距离
            obs_distances = self._compute_obstacle_distances(current_angles[i])  # [3]

            # 对每个障碍物计算ψ函数
            for j in range(self.num_obstacles):
                d = obs_distances[j]  # 第j个障碍物的最小距离

                # 🎯 根据论文ψ(d) = max(0, 1 - d/d_max)
                psi = torch.clamp(1.0 - d / dmax, min=0.0)  # ✅ 论文 ψ 函数

                psi_sum[i] += psi

        # 🔍 DEBUG: Check psi_sum for NaN
        if torch.isnan(psi_sum).any():
            print(f"🚨 [REWARD DEBUG] NaN in psi_sum!")
            for i in range(self.num_envs):
                if torch.isnan(psi_sum[i]):
                    print(f"   Env {i}: psi_sum NaN")
                    # Recalculate with debugging
                    try:
                        obs_distances = self._compute_obstacle_distances(current_angles[i])
                        print(f"   Obs distances: {obs_distances}")
                    except Exception as e:
                        print(f"   Obs distance calculation failed: {e}")
            psi_sum = torch.nan_to_num(psi_sum, nan=0.0)  # Fallback

        # 添加障碍物避免项到奖励函数
        rewards -= w2 * psi_sum

        # 🔍 FINAL DEBUG: Check final rewards
        if torch.isnan(rewards).any():
            print(f"🚨 [REWARD DEBUG] NaN in final rewards!")
            nan_indices = torch.isnan(rewards).nonzero()
            for idx in nan_indices[:5]:  # Show first 5 NaN values
                env_idx = idx[0].item()
                print(f"   Env {env_idx}: NaN reward")
                # Try to identify the source
                print(f"   e_shape: {e_shape if 'e_shape' in locals() else 'N/A'}")
                print(f"   theta: {theta if 'theta' in locals() else 'N/A'}")
                print(f"   psi_sum: {psi_sum[env_idx] if env_idx < len(psi_sum) else 'N/A'}")
            # Replace NaN with zero reward
            rewards = torch.nan_to_num(rewards, nan=0.0)

        # 6. 进步奖励：比上一帧更靠近目标就加分
        """progress_weight = self.trajectory_config.get("progress_weight", 5.0) if hasattr(self, 'trajectory_config') else 5.0
        prev_errors = self._prev_position_errors

        # 计算进步（正数表示误差变小了）
        progress = prev_errors - position_errors
        progress_reward = progress_weight * torch.clamp(progress, min=0.0)  # 只奖励正向进步
        rewards += progress_reward"""

        # 7. 成功奖励：到达目标位置
        """success_threshold = 0.05  # 5cm
        self.waypoint_bonus = 50.0
        success_bonus = self.waypoint_bonus if hasattr(self, 'waypoint_bonus') else 10.0
        success = position_errors < success_threshold
        rewards += success.float() * success_bonus"""

        # 8. 更新误差跟踪
        #self._prev_position_errors = position_errors.detach()

        # 9. 应用奖励缩放
        #rewards = self.reward_scale * rewards

        # 10. 调试信息（每100步打印一次）- 显示论文误差类型
        """if hasattr(self, 'debug_step') and self.debug_step % 100 == 0:
            avg_pos_error = position_errors.mean().item()
            avg_ori_error = orientation_errors.mean().item()
            avg_reward = rewards.mean().item()

            print(f"📈 步骤{self.debug_step} (论文A.2几何误差):")
            print(f"   平均位置误差: {avg_pos_error:.4f} m")
            print(f"   平均姿态误差: {avg_ori_error:.4f} rad")
            print(f"   平均奖励: {avg_reward:.4f}")
            print(f"   λ_ori: {lambda_ori:.3f}, τ: {tau:.3f}")"""

        return rewards


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
        """if hasattr(self, 'debug_step') and self.debug_step % 100 == 0:
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
            print(f"   超时: {timeout_rate:.2%}")"""

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

    # 🎯 新增：障碍物相关方法
    def _sample_obstacle_position(self):
        """
        根据论文附录A.4采样障碍物位置
        工作空间：四分之一球环区域，major=0.6m��minor=0.15m，外加圆柱半径0.30m
        """
        # 论文参数
        major_radius = 0.6   # 主环半径
        minor_radius = 0.15  # 次环半径
        cylinder_radius = 0.30  # 圆柱半径

        max_attempts = 100
        for _ in range(max_attempts):
            # 在球环区域采样
            # 随机采样球坐标
            theta = np.random.uniform(0, 2 * np.pi)  # 方位角
            phi = np.random.uniform(0, np.pi/2)       # 极角（只采样上半球）
            r = np.random.uniform(major_radius - minor_radius, major_radius + minor_radius)  # 径向距离

            # 转换为笛卡尔坐标
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)

            # 检查是否在圆柱外
            radial_dist = np.sqrt(x**2 + y**2)
            if radial_dist > cylinder_radius and z > 0.1:  # 确保在地面上方且不在圆柱内
                return gymapi.Vec3(x, y, z)

        # 如果采样失败，返回默认位置
        return gymapi.Vec3(0.5, 0.5, 0.5)

    def _compute_link_positions(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """
        计算6个关节点（形成5条link）的位置

        Returns:
            link_points: [6, 3] 6个关节点的位置（包括基座和末端）
        """
        link_points = torch.zeros((7, 3), device=joint_angles.device)  # 6个关节 + 末端
        joint_angles = joint_angles.view(-1)

        # UR10e DH参数 (与forward kinematics保持一致)
        d = torch.tensor([0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655], device=joint_angles.device)
        a = torch.tensor([0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0], device=joint_angles.device)
        alpha = torch.tensor([math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0], device=joint_angles.device)

        # 累积变换
        T_cum = torch.eye(4, device=joint_angles.device, dtype=joint_angles.dtype)
        link_points[0] = T_cum[:3, 3]  # 基座位置

        for i in range(6):
            # DH变换
            theta = joint_angles[i]
            ct = torch.cos(theta)
            st = torch.sin(theta)
            ca = torch.cos(alpha[i])
            sa = torch.sin(alpha[i])

            T_i = torch.zeros((4, 4), device=joint_angles.device, dtype=joint_angles.dtype)
            T_i[0, 0] = ct
            T_i[0, 1] = -st * ca
            T_i[0, 2] = st * sa
            T_i[0, 3] = a[i] * ct
            T_i[1, 0] = st
            T_i[1, 1] = ct * ca
            T_i[1, 2] = -ct * sa
            T_i[1, 3] = a[i] * st
            T_i[2, 0] = 0.0
            T_i[2, 1] = sa
            T_i[2, 2] = ca
            T_i[2, 3] = d[i]
            T_i[3, 3] = 1.0

            T_cum = T_cum @ T_i
            link_points[i+1] = T_cum[:3, 3]  # 第i+1个关节位置

        return link_points  # [7, 3]

    def _distance_point_to_segment(self, point: torch.Tensor, seg_start: torch.Tensor, seg_end: torch.Tensor) -> torch.Tensor:
        """
        计算点到线段的最短距离（论文附录A.23/A.24几何公式）

        Args:
            point: [3] 点坐标
            seg_start: [3] 线段起点
            seg_end: [3] 线段终点

        Returns:
            distance: 最短距离
        """
        # 计算线段向量
        seg_vec = seg_end - seg_start  # [3]
        seg_len_sq = torch.sum(seg_vec ** 2)  # 线段长度平方

        # 如果线段长度接近0，返回点到起点的距离
        if seg_len_sq < 1e-8:
            return torch.norm(point - seg_start)

        # 计算投影系���t
        point_vec = point - seg_start  # [3]
        t = torch.dot(point_vec, seg_vec) / seg_len_sq

        # 限制t在[0,1]范围内
        t = torch.clamp(t, 0.0, 1.0)

        # 计算最近点
        closest_point = seg_start + t * seg_vec

        # 计算距离
        distance = torch.norm(point - closest_point)

        return distance

    def _compute_obstacle_distances(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """
        计算障碍物到5-link的距离（dobs）

        Returns:
            dobs: [3] 每个障碍物对5条link取最小距离
        """
        # 获取当前7个点（基座+6关节）的位置
        link_points = self._compute_link_positions(joint_angles)  # [7, 3]

        # 形成5条link线段 (6个关节点形成5条link)
        link_segments = []
        for i in range(6):  # 6个关节点形成5条link
            link_segments.append((link_points[i], link_points[i+1]))

        # 初始化障碍物距离
        dobs = torch.zeros(self.num_obstacles, device=joint_angles.device)

        # 🎯 获取障碍物位置（优先使用实际存储的位置）
        obstacle_positions_to_use = None

        if hasattr(self, 'obstacle_positions') and len(self.obstacle_positions) > 0:
            # 使用第一个环境的障碍物位置作为默认值（单个环境调用时的回退方案）
            if isinstance(self.obstacle_positions[0], list):
                obstacle_positions_to_use = torch.tensor(
                    self.obstacle_positions[0], device=joint_angles.device, dtype=torch.float32
                )  # [3, 3]
            else:
                # 如果是tensor格式，直接使用
                obstacle_positions_to_use = self.obstacle_positions[0]  # [3, 3]
        else:
            # 如果还没有障碍物位置，使用默认值
            obstacle_positions_to_use = torch.tensor([
                [0.4, 0.4, 0.5],
                [0.4, -0.4, 0.5],
                [-0.4, 0.0, 0.5]
            ], device=joint_angles.device, dtype=torch.float32)  # [3, 3]

        # 对每个障碍物计算到所有link的最小距离
        for obs_idx in range(self.num_obstacles):
            obs_pos = obstacle_positions_to_use[obs_idx]  # [3]

            min_distance = float('inf')

            # 计算障碍物到每条link的距离，取最小值
            for link_idx, (seg_start, seg_end) in enumerate(link_segments):
                dist = self._distance_point_to_segment(obs_pos, seg_start, seg_end)
                min_distance = min(min_distance, dist.item())

            dobs[obs_idx] = min_distance

        return dobs

    def _compute_obstacle_distances_batch(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """
        批量计算障碍物到5-link的距离（dobs）

        Args:
            joint_angles: [num_envs, 6] 所有环境的关节角度

        Returns:
            dobs: [num_envs, 3] 每个环境每个障碍物的最小距离
        """
        # 🔍 DEBUG: Check input joint angles
        if torch.isnan(joint_angles).any():
            print(f"🚨 [DEBUG OBS_DIST] NaN in input joint_angles!")
            print(f"   joint_angles: {joint_angles}")

        # 🎯 确保joint_angles是2D tensor
        if joint_angles.ndim == 1:
            joint_angles = joint_angles.unsqueeze(0)  # [6] -> [1, 6]

        num_envs = joint_angles.shape[0]
        dobs = torch.zeros((num_envs, self.num_obstacles), device=joint_angles.device)

        # 🎯 使用实际存储的障碍物位置
        if hasattr(self, 'obstacle_positions') and len(self.obstacle_positions) > 0:
            # 将障碍物位置转换为tensor [num_envs, num_obstacles, 3]
            obstacle_positions_np = np.array(self.obstacle_positions)  # [num_envs, 3, 3]
            obstacle_positions_tensor = torch.tensor(
                obstacle_positions_np, device=joint_angles.device, dtype=torch.float32
            )  # [num_envs, num_obstacles, 3]
        else:
            # 如果还没有障碍物位置，使用默认值
            default_positions = np.array([
                [0.4, 0.4, 0.5],
                [0.4, -0.4, 0.5],
                [-0.4, 0.0, 0.5]
            ])
            obstacle_positions_tensor = torch.tensor(
                default_positions, device=joint_angles.device, dtype=torch.float32
            ).unsqueeze(0).expand(num_envs, -1, -1)  # [num_envs, 3, 3]

        # 🔍 DEBUG: Check obstacle positions
        if torch.isnan(obstacle_positions_tensor).any():
            print(f"🚨 [DEBUG OBS_DIST] NaN in obstacle_positions_tensor!")
            print(f"   obstacle_positions_tensor: {obstacle_positions_tensor}")

        # 对每个环境计算障碍物距离
        for env_idx in range(num_envs):
            env_angles = joint_angles[env_idx]  # [6]
            env_obstacle_positions = obstacle_positions_tensor[env_idx]  # [3, 3]

            # 🔍 DEBUG: Check individual environment values
            if torch.isnan(env_angles).any():
                print(f"🚨 [DEBUG OBS_DIST] NaN in env_angles for env {env_idx}!")
                print(f"   env_angles: {env_angles}")
            if torch.isnan(env_obstacle_positions).any():
                print(f"🚨 [DEBUG OBS_DIST] NaN in env_obstacle_positions for env {env_idx}!")
                print(f"   env_obstacle_positions: {env_obstacle_positions}")

            # 计算当前环境的link位置
            try:
                link_points = self._compute_link_positions(env_angles)  # [7, 3]

                # 🔍 DEBUG: Check link points
                if torch.isnan(link_points).any():
                    print(f"🚨 [DEBUG OBS_DIST] NaN in link_points for env {env_idx}!")
                    print(f"   env_angles: {env_angles}")
                    print(f"   link_points: {link_points}")
                    # Use fallback link points
                    link_points = torch.zeros((7, 3), device=env_angles.device)
                    link_points[0] = torch.tensor([0.0, 0.0, 0.0], device=env_angles.device)
                    for i in range(6):
                        link_points[i+1] = link_points[i] + torch.tensor([0.1, 0.0, 0.0], device=env_angles.device)
            except Exception as e:
                print(f"🚨 [DEBUG OBS_DIST] Exception in _compute_link_positions for env {env_idx}: {e}")
                # Use fallback link points
                link_points = torch.zeros((7, 3), device=env_angles.device)
                link_points[0] = torch.tensor([0.0, 0.0, 0.0], device=env_angles.device)
                for i in range(6):
                    link_points[i+1] = link_points[i] + torch.tensor([0.1, 0.0, 0.0], device=env_angles.device)

            # 形成5条link线段 (6个关节点形成5条线段)
            link_segments = []
            for i in range(6):  # 6个关节点形成5条link
                link_segments.append((link_points[i], link_points[i+1]))

            # 对每个障碍物计算到所有link的最小距离
            for obs_idx in range(self.num_obstacles):
                obs_pos = env_obstacle_positions[obs_idx]  # [3]

                # 🔍 DEBUG: Check obstacle position
                if torch.isnan(obs_pos).any():
                    print(f"🚨 [DEBUG OBS_DIST] NaN in obs_pos for env {env_idx}, obs {obs_idx}!")
                    print(f"   obs_pos: {obs_pos}")
                    obs_pos = torch.tensor([0.5, 0.5, 0.5], device=obs_pos.device)  # Fallback

                min_distance = float('inf')

                # 计算障碍物到每条link的距离，取最小值
                for link_idx, (seg_start, seg_end) in enumerate(link_segments):
                    try:
                        # 🔍 DEBUG: Check segment points
                        if torch.isnan(seg_start).any() or torch.isnan(seg_end).any():
                            print(f"🚨 [DEBUG OBS_DIST] NaN in segment for env {env_idx}, link {link_idx}!")
                            print(f"   seg_start: {seg_start}")
                            print(f"   seg_end: {seg_end}")
                            continue  # Skip this segment

                        dist = self._distance_point_to_segment(obs_pos, seg_start, seg_end)

                        # 🔍 DEBUG: Check distance calculation
                        if torch.isnan(dist):
                            print(f"🚨 [DEBUG OBS_DIST] NaN in distance calculation!")
                            print(f"   obs_pos: {obs_pos}")
                            print(f"   seg_start: {seg_start}")
                            print(f"   seg_end: {seg_end}")
                            dist = torch.tensor(1.0, device=obs_pos.device)  # Fallback distance

                        min_distance = min(min_distance, dist.item())
                    except Exception as e:
                        print(f"🚨 [DEBUG OBS_DIST] Exception in distance calculation for env {env_idx}, obs {obs_idx}, link {link_idx}: {e}")
                        continue  # Skip this problematic calculation

                # Ensure we have a valid distance value
                if min_distance == float('inf') or np.isnan(min_distance):
                    min_distance = 1.0  # Fallback distance

                dobs[env_idx, obs_idx] = min_distance

        # 🔍 DEBUG: Check final dobs result
        if torch.isnan(dobs).any():
            print(f"🚨 [DEBUG OBS_DIST] NaN in final dobs!")
            nan_indices = torch.isnan(dobs).nonzero()
            for idx in nan_indices[:5]:  # Show first 5 NaN values
                env_idx, obs_idx = idx[0].item(), idx[1].item()
                print(f"   Env {env_idx}, Obs {obs_idx}: NaN")
                print(f"   Env angles: {joint_angles[env_idx] if env_idx < joint_angles.shape[0] else 'N/A'}")
                print(f"   Obs position: {obstacle_positions_tensor[env_idx, obs_idx] if env_idx < obstacle_positions_tensor.shape[0] else 'N/A'}")
            # Replace NaN with fallback values
            dobs = torch.nan_to_num(dobs, nan=1.0, posinf=10.0, neginf=0.0)

        return dobs

    def close(self):
        """关闭环境"""
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
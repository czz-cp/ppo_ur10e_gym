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
        self.device_id = device_id

        # 🎯 GPU设备配置（参考 isaac_gym_manipulator 模式）
        if torch.cuda.is_available() and device_id >= 0:
            # 检查GPU设备是否可用
            if device_id < torch.cuda.device_count():
                self.device = torch.device(f'cuda:{device_id}')
                # 设置当前CUDA设备（确保所有操作都在指定的GPU上）
                torch.cuda.set_device(device_id)
            else:
                print(f"[Warning] GPU {device_id} not available, only {torch.cuda.device_count()} GPUs found. Using GPU 0.")
                self.device = torch.device('cuda:0')
                torch.cuda.set_device(0)
                device_id = 0  # 更新为实际使用的设备ID
        else:
            self.device = torch.device('cpu')
            device_id = -1  # CPU模式

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

        # 动作空间限制 (PID参数调度)
        self.action_space_high = np.array([0.5, 0.5, 1.0])  # [kp_scale, kd_scale, ki_enable]
        self.action_space_low = np.array([-0.5, -0.5, 0.0])

        # 状态空间 (16维RL-PID混合控制)
        self.state_dim = 16
        self.action_dim = 3

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
        self.target_positions = None
        self.prev_position_errors = None

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

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
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
        enable_rendering = simulator_config.get('enable_rendering', False)
        graphics_device = simulator_config.get('graphics_device', self.device_id)

        # 根据渲染设置选择图形设备
        if enable_rendering:
            graphics_device_id = graphics_device
            print(f"🎬 启用渲染模式，图形设备: {graphics_device_id}")
        else:
            graphics_device_id = -1  # 无头模式
            print("🖥️ 无头模式，禁用渲染")

        # 创建仿真器（使用PhysX而非FleX，参考isaac_gym_manipulator）
        self.sim = self.gym.create_sim(
            compute_device=self.device_id,
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

        # 渲染配置
        self.enable_rendering = self.config.get('simulator', {}).get('enable_rendering', False)
        self.graphics_device = self.config.get('simulator', {}).get('graphics_device', self.device_id)

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
        # 观测空间
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

        # 获取状态和动作的张量视图
        self.root_states = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_states = self.gym.acquire_dof_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(self.root_states)
        self.dof_states = gymtorch.wrap_tensor(self.dof_states)

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

        # 随机生成目标位置
        self.target_positions = self._sample_random_target_positions_batch()

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

        # 应用到仿真（isaac_gym_manipulator 模式）
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_states))

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
            actions: PID调度动作 [num_envs, 3] [kp_scale, kd_scale, ki_enable]

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
        for i in range(self.num_envs):
            if not dones[i]:
                if self.reward_normalizers[i] is not None:
                    self.reward_normalizers[i].update(rewards[i].item())
                    rewards[i] = self.reward_normalizers[i].normalize(rewards[i].item())
                # 如果没有归一化器，使用原始奖励

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

    def _get_states(self) -> torch.Tensor:
        """获取所有环境的当前状态"""
        states = torch.zeros((self.num_envs, self.state_dim), device=self.device)

        # 获取当前关节角度和速度
        current_angles, current_velocities = self._get_joint_angles_and_velocities()

        # 计算末端位置（这里需要Isaac Gym的前向动力学）
        current_positions = self._compute_end_effector_positions_batch(current_angles)

        # 构建状态向量 (16维RL-PID混合控制)
        # [current_angles(6), current_velocities(6), current_position(3), distance_to_target(1)]
        distance_to_target = torch.norm(current_positions - self.target_positions, dim=1, keepdim=True)

        states[:, 0:6] = current_angles
        states[:, 6:12] = current_velocities
        states[:, 12:15] = current_positions
        states[:, 15] = distance_to_target.squeeze()

        # 设备一致性检查
        if hasattr(self, '_debug_mode') and self._debug_mode:
            if not check_tensor_devices({'states': states, 'target_positions': self.target_positions}, "_get_states"):
                print(f"⚠️ _get_states设备不一致: states在{states.device}, target_positions在{self.target_positions.device}")

        return states

    def _compute_end_effector_positions_batch(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """批量计算末端执���器位置"""
        # 确保输入张量在正确的设备上
        joint_angles = joint_angles.to(self.device)
        # 使用运动学解算器计算末端位置
        positions = torch.zeros((self.num_envs, 3), device=self.device)

        for i in range(self.num_envs):
            angles_np = joint_angles[i].detach().cpu().numpy()
            if self.kinematics is not None:
                T = self.kinematics.forward_kinematics(angles_np)
                positions[i] = torch.tensor(T[:3, 3], device=self.device)
            else:
                # 简化位置计算（近似）
                positions[i] = torch.tensor([
                    0.8 * np.cos(angles_np[0]) * np.cos(angles_np[1]),
                    0.8 * np.cos(angles_np[0]) * np.sin(angles_np[1]),
                    0.8 * np.sin(angles_np[1]) + 0.3
                ], device=self.device)

        return positions

    def _apply_rl_pid_control(self, actions: torch.Tensor):
        """应用RL-PID控制（使用Isaac Gym官方franka_osc.py的正确模式）"""
        # 解析动作
        kp_scale = actions[:, 0]
        kd_scale = actions[:, 1]
        ki_enable = actions[:, 2]

        # 获取当前状态
        current_angles, current_velocities = self._get_joint_angles_and_velocities()
        current_positions = self._compute_end_effector_positions_batch(current_angles)

        # 计算位置误差
        position_errors = self.target_positions - current_positions
        distance_errors = torch.norm(position_errors, dim=1)

        # 🎯 调试信息：每100步打印一次位置和误差信息
        if hasattr(self, 'debug_step') and self.debug_step % 100 == 0:
            print(f"\n📊 === 步骤 {self.debug_step} 调试信息 ===")
            for i in range(min(self.num_envs, 2)):  # 只打印前2个环境
                print(f"🤖 环境{i}:")
                print(f"   当前末端位置: [{current_positions[i].cpu().numpy().tolist()}]")
                print(f"   目标位置: [{self.target_positions[i].cpu().numpy().tolist()}]")
                print(f"   位置误差: [{position_errors[i].cpu().numpy().tolist()}]")
                print(f"   距离误差: {distance_errors[i].item():.4f}m")
                print(f"   关节角度: [{current_angles[i].cpu().numpy().tolist()}]")

        # 批量计算所有环境的控制力矩（修复设备问题：Isaac Gym期望CPU张量）
        # 初始化力矩张量 [num_envs, num_dofs, 1] - 必须是CPU张量！
        all_dof_forces = torch.zeros(self.num_envs, 6, 1, device='cpu')

        # 为每个环境计算控制力矩
        for i in range(self.num_envs):
            if distance_errors[i] > 1e-4:  # 只在有效时计算
                joint_control = self._compute_jacobian_control(
                    current_angles[i], current_velocities[i],
                    position_errors[i], actions[i]
                )

                # UR10e官方力矩限制 (N⋅m)
                ur10e_torque_limits = [330.0, 330.0, 150.0, 54.0, 54.0, 54.0]  # 基于官方URDF配置
                joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_joint', 'wrist_1', 'wrist_2', 'wrist_3']

                # 将计算出的力矩放入张量（处理设备转移：GPU计算 -> CPU存储）
                for j in range(6):  # 6个关节
                    # 提取力矩值并确保张量格式
                    if isinstance(joint_control[j], torch.Tensor):
                        force_value = joint_control[j]
                    else:
                        force_value = torch.tensor(float(joint_control[j]), device=self.device)

                    # 使用UR10e官方力矩限制
                    max_torque = ur10e_torque_limits[j]
                    min_torque = -max_torque
                    force_value = torch.clamp(force_value, min_torque, max_torque)

                    # 关键修复：转移到CPU标量值以匹配CPU张量
                    all_dof_forces[i, j, 0] = force_value.cpu().item()

                # 调试信息（仅第一个环境，每100步输出一次）
                if i == 0 and hasattr(self, 'debug_step') and self.debug_step % 100 == 0:
                    forces_list = all_dof_forces[i, :, 0].numpy()  # 已经在CPU上
                    print(f"   🔧 应用力矩: [{forces_list.tolist()}] N⋅m")
                    for j, (name, force, limit) in enumerate(zip(joint_names, forces_list, ur10e_torque_limits)):
                        saturation = abs(force) / limit * 100
                        print(f"      {j+1}. {name:12}: {force:7.2f} N⋅m (限制: ±{limit:5.1f}, 饱和度: {saturation:5.1f}%)")

        # 使用Isaac Gym官方示例的正确API：一次性设置所有环境的力矩
        # 参考: gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(u))
        try:
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(all_dof_forces))
            if hasattr(self, 'debug_step') and self.debug_step % 100 == 0:
                print(f"✅ Isaac Gym力矩设置成功: 形状={all_dof_forces.shape}, 设备={all_dof_forces.device}")
        except Exception as e:
            print(f"❌ Isaac Gym力矩设置失败: {e}")
            print(f"   力矩张量形状: {all_dof_forces.shape}")
            print(f"   力矩张量设备: {all_dof_forces.device}")
            print(f"   力矩张量类型: {all_dof_forces.dtype}")
            print(f"   力矩范数: {torch.norm(all_dof_forces)}")

    def _compute_jacobian_control(self, current_angles: torch.Tensor,
                                current_velocities: torch.Tensor,
                                position_error: torch.Tensor,
                                action: torch.Tensor) -> torch.Tensor:
        """计算基于雅可比的控制"""
        kp_scale, kd_scale, ki_enable = action

        # 映射缩放因子
        kp_scale = 0.1 + kp_scale * 2.0  # [0.1, 2.1]
        kd_scale = 0.1 + kd_scale * 2.0  # [0.1, 2.1]
        ki_enable = max(0.0, ki_enable)   # [0.0, 1.0+]

        # 🎯 使用官方UR10e PID参数（来自isaac_gym_manipulator官方配置）
        # 参考: /isaac_gym_manipulator/ros_sources/universal_robot/ur_gazebo/config/ur10e_controllers.yaml
        base_kp = self.config['pid_params']['base_gains']['p']
        base_kd = self.config['pid_params']['base_gains']['d']
        base_ki = self.config['pid_params']['base_gains']['i']

        # RL调度参数（每个关节分别计算）
        kp = [base_kp[i] * kp_scale for i in range(6)]
        kd = [base_kd[i] * kd_scale for i in range(6)]
        ki = [base_ki[i] * ki_enable for i in range(6)]

        # 计算雅可比矩阵（简化版本）
        jacobian = self._compute_jacobian_batch(current_angles.unsqueeze(0))[0]

        # 🎯 ��化：每个关节使用各自的kp进行控制（参考原始MuJoCo实现）
        joint_control = torch.zeros(6, device=self.device)

        # 先转换任务空间误差到关节空间
        joint_position_errors = jacobian.T @ position_error

        # 每个关节使用各自的kp、kd进行控制（ki参数预留，未来可添加积分项）
        for i in range(6):
            # 比例���：每个关节使用各自的kp
            p_term = kp[i] * joint_position_errors[i]

            # 阻尼项：每个关节使用各自的kd
            d_term = kd[i] * current_velocities[i]

            # 🎯 官方PID参数已配置，积分项预留（需要误差累积状态）
            # i_term = ki[i] * self.integral_errors[i]  # 未来可添加

            # 关节控制力矩 = 比例项 - 阻尼项 (+ 积分项)
            joint_control[i] = p_term - d_term

        return joint_control

    def _compute_jacobian_batch(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """批量计算雅可比矩阵"""
        # 确保输入张量在正确的设备上
        joint_angles = joint_angles.to(self.device)
        batch_size = joint_angles.shape[0]
        jacobian = torch.zeros((batch_size, 3, 6), device=self.device)
        epsilon = 1e-6

        for i in range(batch_size):
            angles_np = joint_angles[i].detach().cpu().numpy()
            jacobian_np = self._compute_jacobian_single(angles_np)
            jacobian[i] = torch.tensor(jacobian_np, device=self.device)

        return jacobian

    def _compute_jacobian_single(self, joint_angles: np.ndarray) -> np.ndarray:
        """计算单个雅可比矩阵"""
        if self.kinematics is not None:
            current_pos = self.kinematics.get_end_effector_position(joint_angles)
            jacobian = np.zeros((3, 6))

            # 数值微分
            for i in range(6):
                delta_q = np.zeros(6)
                delta_q[i] = 1e-6

                perturbed_pos = self.kinematics.get_end_effector_position(joint_angles + delta_q)
                jacobian[:, i] = (perturbed_pos - current_pos) / 1e-6

            return jacobian
        else:
            # 简化雅可比矩阵近似
            return np.eye(3, 6) * 0.1

    def _compute_rewards_batch(self, actions: torch.Tensor) -> torch.Tensor:
        """
        🎯 基于原始MuJoCo实现的批量奖励计算（去除能耗奖励）

        设计思路参考论文：r(s_t, a_t) = r_a^t + r_s^t + r_ex^t

        Args:
            actions: PID调度动作 [num_envs, 3]

        Returns:
            rewards: 奖励值 [num_envs]
        """
        # 获取当前状态
        current_angles, current_velocities = self._get_joint_angles_and_velocities()
        current_positions = self._compute_end_effector_positions_batch(current_angles)

        # 计算位置误差
        position_errors = torch.norm(self.target_positions - current_positions, dim=1)

        # 1. 🎯 精度奖励 r_a^t = -w_a * exp(σ_a * f_a(θ^t))
        # 基于原始MuJoCo实现的指数惩罚设计
        f_a_theta = position_errors ** 2  # f_a(θ^t) = ||p_d - p||^2

        # 使用指数惩罚：误差小时惩罚温和，误差大时惩罚急剧增加
        # 使用config中的sigma参数，便于调整惩罚的陡峭程度
        sigma = self.config['reward']['accuracy']['sigma']
        accuracy_reward = -self.config['reward']['accuracy']['weight'] * torch.exp(sigma * f_a_theta)

        # 2. 🏃 速度奖励 r_s^t（奖励误差减少速度）
        if self.prev_position_errors is not None:
            error_change = self.prev_position_errors - position_errors
            speed_reward = self.config['reward']['speed']['weight'] * torch.clamp(error_change, min=0.0)
        else:
            speed_reward = torch.zeros_like(position_errors)

        # 3. 🔧 稳定性奖励（PID参数变化幅度控制）
        stability_reward = -self.config['reward']['stability']['weight'] * (
            torch.abs(actions[:, 0]) + torch.abs(actions[:, 1])  # kp_scale + kd_scale
        )

        # 📝 注释掉能耗奖励，专注于位置控制性能
        # # 4. 能耗奖励（已移除）
        # energy_cost = torch.sum(current_velocities ** 2, dim=1)
        # energy_reward = -self.config['reward']['energy']['weight'] * energy_cost

        # 🏁 总奖励（去除能耗奖励）
        total_reward = accuracy_reward + speed_reward + stability_reward

        # 🎊 稀疏成功奖励（到达目标时的额外奖励）
        success_mask = position_errors < self.config['reward']['accuracy']['threshold']
        total_reward[success_mask] += self.config['reward']['extra']['success_reward']

        # 💾 保存误差历史用于下次计算速度奖励
        self.prev_position_errors = position_errors.clone()

        # 📊 调试信息（每100步打印一次）
        if hasattr(self, 'debug_step') and self.debug_step % 100 == 0:
            avg_error = position_errors.mean().item()
            avg_reward = total_reward.mean().item()
            success_rate = success_mask.float().mean().item()
            print(f"📈 步骤{self.debug_step}: 平均误差={avg_error:.4f}m, 平均奖励={avg_reward:.4f}, 成功率={success_rate:.2%}")

        return total_reward

    def _check_done_batch(self) -> torch.Tensor:
        """检查完成条件"""
        # 获取当前位置
        current_angles, _ = self._get_joint_angles_and_velocities()
        current_positions = self._compute_end_effector_positions_batch(current_angles)

        # 计算位置误差
        position_errors = torch.norm(self.target_positions - current_positions, dim=1)

        # 完成条件：成功到达或超过最大步数
        success_done = position_errors < self.config['reward']['accuracy']['threshold']
        timeout_done = self.episode_steps >= self.max_steps  # 使用每个环境的episode步数

        return success_done | timeout_done

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
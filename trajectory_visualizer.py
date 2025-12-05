#!/usr/bin/env python3
"""
轨迹可视化模块
用于可视化RRT*规划的路径和机器人实际运动轨迹
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from typing import List, Optional, Tuple, Dict, Any
import os
import time

class TrajectoryVisualizer:
    """轨迹可视化器"""

    def __init__(self, workspace_bounds: np.ndarray, save_dir: str = "./trajectory_plots"):
        """
        初始化可视化器

        Args:
            workspace_bounds: 工作空间边界 [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
            save_dir: 图片保存目录
        """
        self.workspace_bounds = workspace_bounds
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 存储轨迹数据
        self.planned_paths = []      # 规划的路径点
        self.actual_trajectories = []  # 实际运动轨迹
        self.tcp_history = []         # TCP位置历史
        self.joint_history = []        # 关节角度历史

        # 统计信息
        self.stats = {
            'total_plans': 0,
            'successful_plans': 0,
            'total_waypoints': 0,
            'total_distance': 0.0
        }

    def add_planned_path(self, waypoints: List, plan_info: Dict[str, Any] = None):
        """
        添加规划路径

        Args:
            waypoints: 路径点列表
            plan_info: 规划信息（规划时间、距离等）
        """
        if waypoints:
            path_points = np.array([wp.cartesian_position for wp in waypoints])
            self.planned_paths.append({
                'points': path_points,
                'waypoints': waypoints,
                'info': plan_info or {},
                'timestamp': time.time()
            })

            self.stats['total_plans'] += 1
            self.stats['successful_plans'] += 1
            self.stats['total_waypoints'] += len(waypoints)

            if plan_info and 'total_distance' in plan_info:
                self.stats['total_distance'] += plan_info['total_distance']

    def add_trajectory_point(self, tcp_pos: np.ndarray, joint_angles: np.ndarray, action: np.ndarray = None):
        """
        添加实际轨迹点

        Args:
            tcp_pos: TCP位置 [x, y, z]
            joint_angles: 关节角度 [6]
            action: 动作向量 [6] (可选)
        """
        self.tcp_history.append(tcp_pos.copy())
        self.joint_history.append(joint_angles.copy())

        # 为实际轨迹添加动作信息
        if self.actual_trajectories:
            self.actual_trajectories[-1]['actions'].append(action.copy() if action is not None else None)

    def start_new_trajectory(self, plan_index: int = 0):
        """
        开始新的实际轨迹记录

        Args:
            plan_index: 对应的规划路径索引
        """
        self.actual_trajectories.append({
            'plan_index': plan_index,
            'tcp_points': [],
            'joint_angles': [],
            'actions': [],
            'start_time': time.time()
        })

    def plot_planned_path_3d(self, path_index: int = -1, show: bool = True, save: bool = True) -> str:
        """
        绘制3D规划路径

        Args:
            path_index: 路径索引，-1表示最新路径
            show: 是否显示图形
            save: 是否保存图片

        Returns:
            保存的文件路径
        """
        if not self.planned_paths:
            print("❌ 没有规划路径可绘制")
            return ""

        path_data = self.planned_paths[path_index]
        points = path_data['points']
        info = path_data['info']

        # 创建3D图形
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制工作空间边界
        self._draw_workspace_bounds(ax)

        # 绘制规划路径
        ax.plot(points[:, 0], points[:, 1], points[:, 2],
               'b-', linewidth=2, label='RRT* 规划路径', alpha=0.8)

        # 标记路径点
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c='red', s=50, alpha=0.6, label='路径点')

        # 标记起点和终点
        ax.scatter(points[0, 0], points[0, 1], points[0, 2],
                  c='green', s=200, marker='o', label='起点')
        ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2],
                  c='red', s=200, marker='*', label='终点')

        # 添加路径点编号
        for i, point in enumerate(points[::2]):  # 每2个点显示一个标签
            ax.text(point[0], point[1], point[2], f'  {i*2}', fontsize=8)

        # 设置标签和标题
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        title = f'RRT* 规划路径可视化'
        if info and 'planning_time' in info:
            title += f'\n规划时间: {info["planning_time"]:.3f}s'
        if info and 'total_distance' in info:
            title += f'\n路径长度: {info["total_distance"]:.3f}m'
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.legend(loc='upper right')

        # 设置视角
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()

        # 保存图片
        if save:
            timestamp = int(time.time())
            filename = f"{self.save_dir}/planned_path_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 规划路径图已保存: {filename}")

        if show:
            plt.show()
        else:
            plt.close()

        return filename if save else ""

    def plot_actual_trajectory_3d(self, trajectory_index: int = -1, show: bool = True, save: bool = True) -> str:
        """
        绘制3D实际轨迹

        Args:
            trajectory_index: 轨迹索引，-1表示最新轨迹
            show: 是否显示图形
            save: 是否保存图片

        Returns:
            保存的文件路径
        """
        if not self.actual_trajectories:
            print("❌ 没有实际轨迹可绘制")
            return ""

        traj_data = self.actual_trajectories[trajectory_index]
        tcp_points = np.array(traj_data['tcp_points'])
        plan_index = traj_data['plan_index']

        # 创建3D图形
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制工作空间边界
        self._draw_workspace_bounds(ax)

        # 绘制对应的规划路径
        if plan_index < len(self.planned_paths):
            planned_points = self.planned_paths[plan_index]['points']
            ax.plot(planned_points[:, 0], planned_points[:, 1], planned_points[:, 2],
                   'b--', linewidth=1, alpha=0.5, label='规划路径')

            # 标记规划的起点和终点
            ax.scatter(planned_points[0, 0], planned_points[0, 1], planned_points[0, 2],
                      c='green', s=100, marker='o', alpha=0.5)
            ax.scatter(planned_points[-1, 0], planned_points[-1, 1], planned_points[-1, 2],
                      c='red', s=100, marker='*', alpha=0.5)

        # 绘制实际轨迹
        if len(tcp_points) > 0:
            ax.plot(tcp_points[:, 0], tcp_points[:, 1], tcp_points[:, 2],
                   'r-', linewidth=2, label='实际轨迹', alpha=0.8)

            # 标记实际起点和当前位置
            ax.scatter(tcp_points[0, 0], tcp_points[0, 1], tcp_points[0, 2],
                      c='blue', s=200, marker='o', label='实际起点')
            ax.scatter(tcp_points[-1, 0], tcp_points[-1, 1], tcp_points[-1, 2],
                      c='orange', s=200, marker='^', label='当前位置')

            # 轨迹点
            ax.scatter(tcp_points[:, 0], tcp_points[:, 1], tcp_points[:, 2],
                      c='orange', s=10, alpha=0.4)

        # 设置标签和标题
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        duration = time.time() - traj_data['start_time']
        title = f'机器人实际轨迹可视化\n轨迹点数: {len(tcp_points)}, 时长: {duration:.2f}s'
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.legend(loc='upper right')

        # 设置视角
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()

        # 保存图片
        if save:
            timestamp = int(time.time())
            filename = f"{self.save_dir}/actual_trajectory_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 实际轨迹图已保存: {filename}")

        if show:
            plt.show()
        else:
            plt.close()

        return filename if save else ""

    def plot_comparison_3d(self, trajectory_index: int = -1, show: bool = True, save: bool = True) -> str:
        """
        绘制规划路径与实际轨迹对比图

        Args:
            trajectory_index: 轨迹索引
            show: 是否显示图形
            save: 是否保存图片

        Returns:
            保存的文件路径
        """
        if not self.actual_trajectories or not self.planned_paths:
            print("❌ 缺少规划路径或实际轨迹")
            return ""

        traj_data = self.actual_trajectories[trajectory_index]
        plan_index = traj_data['plan_index']

        if plan_index >= len(self.planned_paths):
            print("❌ 轨迹对应的规划路径不存在")
            return ""

        # 创建图形
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制工作空间边界
        self._draw_workspace_bounds(ax)

        # 绘制规划路径
        planned_points = self.planned_paths[plan_index]['points']
        ax.plot(planned_points[:, 0], planned_points[:, 1], planned_points[:, 2],
               'b--', linewidth=3, label='RRT* 规划路径', alpha=0.7)

        # 标记规划路径点
        ax.scatter(planned_points[:, 0], planned_points[:, 1], planned_points[:, 2],
                  c='blue', s=30, alpha=0.4, label='规划路径点')

        # 绘制实际轨迹
        if traj_data['tcp_points']:
            tcp_points = np.array(traj_data['tcp_points'])
            ax.plot(tcp_points[:, 0], tcp_points[:, 1], tcp_points[:, 2],
                   'r-', linewidth=2, label='实际轨迹', alpha=0.9)

            # 标记实际轨迹点
            ax.scatter(tcp_points[:, 0], tcp_points[:, 1], tcp_points[:, 2],
                      c='red', s=20, alpha=0.6, label='实际轨迹点')

            # 标记起点、终点和当前位置
            ax.scatter(planned_points[0, 0], planned_points[0, 1], planned_points[0, 2],
                      c='green', s=200, marker='o', label='共同起点')
            ax.scatter(planned_points[-1, 0], planned_points[-1, 1], planned_points[-1, 2],
                      c='purple', s=200, marker='*', label='规划终点')
            ax.scatter(tcp_points[-1, 0], tcp_points[-1, 1], tcp_points[-1, 2],
                      c='orange', s=200, marker='^', label='当前位置')

        # 计算跟踪误差
        if traj_data['tcp_points']:
            current_pos = np.array(traj_data['tcp_points'][-1])
            # 找到最近的规划路径点
            distances = np.linalg.norm(planned_points - current_pos, axis=1)
            min_idx = np.argmin(distances)
            error = distances[min_idx]

            # 在图中标注误差信息
            title = f'规划路径 vs 实际轨迹对比\n当前跟踪误差: {error:.4f}m'
        else:
            title = '规划路径 vs 实际轨迹对比'

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')

        # 设置视角
        ax.view_init(elev=25, azim=45)

        plt.tight_layout()

        # 保存图片
        if save:
            timestamp = int(time.time())
            filename = f"{self.save_dir}/comparison_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 对比图已保存: {filename}")

        if show:
            plt.show()
        else:
            plt.close()

        return filename if save else ""

    def plot_joint_angles(self, trajectory_index: int = -1, show: bool = True, save: bool = True) -> str:
        """
        绘制关节角度时间序列

        Args:
            trajectory_index: 轨迹索引
            show: 是否显示图形
            save: 是否保存图片

        Returns:
            保存的文件路径
        """
        if not self.actual_trajectories:
            print("❌ 没有关节角度数据")
            return ""

        traj_data = self.actual_trajectories[trajectory_index]
        joint_angles = np.array(traj_data['joint_angles'])

        if len(joint_angles) == 0:
            print("❌ 关节角度数据为空")
            return ""

        # 创建图形
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        axes = axes.flatten()

        joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_joint', 'wrist_1', 'wrist_2', 'wrist_3']
        time_steps = np.arange(len(joint_angles))

        for i in range(6):
            axes[i].plot(time_steps, np.degrees(joint_angles[:, i]), 'b-', linewidth=2)
            axes[i].set_title(f'{joint_names[i]}')
            axes[i].set_xlabel('时间步')
            axes[i].set_ylabel('角度 (度)')
            axes[i].grid(True, alpha=0.3)

        plt.suptitle('关节角度变化时间序列', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 保存图片
        if save:
            timestamp = int(time.time())
            filename = f"{self.save_dir}/joint_angles_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 关节角度图已保存: {filename}")

        if show:
            plt.show()
        else:
            plt.close()

        return filename if save else ""

    def create_trajectory_animation(self, trajectory_index: int = -1, interval: int = 100) -> animation.FuncAnimation:
        """
        创建轨迹动画

        Args:
            trajectory_index: 轨迹索引
            interval: 动画帧间隔(毫秒)

        Returns:
            matplotlib动画对象
        """
        if not self.actual_trajectories:
            print("❌ 没有轨迹数据可创建动画")
            return None

        traj_data = self.actual_trajectories[trajectory_index]
        plan_index = traj_data['plan_index']

        if plan_index >= len(self.planned_paths):
            print("❌ 对应的规划路径不存在")
            return None

        tcp_points = np.array(traj_data['tcp_points'])
        planned_points = self.planned_paths[plan_index]['points']

        # 创建图形
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制工作空间边界
        self._draw_workspace_bounds(ax)

        # 绘制规划路径
        ax.plot(planned_points[:, 0], planned_points[:, 1], planned_points[:, 2],
               'b--', linewidth=1, alpha=0.5, label='规划路径')

        # 标记规划路径点
        ax.scatter(planned_points[:, 0], planned_points[:, 1], planned_points[:, 2],
                  c='blue', s=20, alpha=0.3)

        # 标记起点和终点
        ax.scatter(planned_points[0, 0], planned_points[0, 1], planned_points[0, 2],
                  c='green', s=100, marker='o', label='起点')
        ax.scatter(planned_points[-1, 0], planned_points[-1, 1], planned_points[-1, 2],
                  c='red', s=100, marker='*', label='终点')

        # 初始化轨迹线
        trajectory_line, = ax.plot([], [], [], 'r-', linewidth=2, label='实际轨迹')
        current_point, = ax.plot([], [], [], 'ro', markersize=8, label='当前位置')

        # 设置标签
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('机器人轨迹动画', fontsize=14, fontweight='bold')
        ax.legend()

        # 设置视角
        ax.view_init(elev=20, azim=45)

        def animate(frame):
            if frame < len(tcp_points):
                # 更新轨迹线
                trajectory_line.set_data(tcp_points[:frame+1, 0], tcp_points[:frame+1, 1])
                trajectory_line.set_3d_properties(tcp_points[:frame+1, 2])

                # 更新当前位置
                current_point.set_data([tcp_points[frame, 0]], [tcp_points[frame, 1]])
                current_point.set_3d_properties([tcp_points[frame, 2]])

                # 更新标题
                ax.set_title(f'机器人轨迹动画 - 步骤 {frame+1}/{len(tcp_points)}',
                             fontsize=14, fontweight='bold')

            return trajectory_line, current_point

        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=len(tcp_points),
                                     interval=interval, blit=False, repeat=True)

        return anim

    def _draw_workspace_bounds(self, ax):
        """绘制工作空间边界"""
        x_bounds = self.workspace_bounds[0]
        y_bounds = self.workspace_bounds[1]
        z_bounds = self.workspace_bounds[2]

        # 绘制工作空间的边界框
        xx, yy = np.meshgrid([x_bounds[0], x_bounds[1]], [y_bounds[0], y_bounds[1]])
        ax.plot_surface(xx, yy, np.ones_like(xx) * z_bounds[0], alpha=0.1, color='gray')
        ax.plot_surface(xx, yy, np.ones_like(xx) * z_bounds[1], alpha=0.1, color='gray')

        # 绘制侧面的边界线
        # X方向的边界面
        for y in y_bounds:
            for z in z_bounds:
                ax.plot([x_bounds[0], x_bounds[1]], [y, y], [z, z], 'k-', alpha=0.3)

        # Y方向的边界面
        for x in x_bounds:
            for z in z_bounds:
                ax.plot([x, x], [y_bounds[0], y_bounds[1]], [z, z], 'k-', alpha=0.3)

    def print_statistics(self):
        """打印统计信息"""
        print("\n📊 轨迹可视化统计:")
        print("=" * 50)
        print(f"总规划次数: {self.stats['total_plans']}")
        print(f"成功规划次数: {self.stats['successful_plans']}")
        print(f"成功率: {self.stats['successful_plans']/max(1, self.stats['total_plans'])*100:.1f}%")
        print(f"总路径点数: {self.stats['total_waypoints']}")
        print(f"总路径长度: {self.stats['total_distance']:.3f}m")
        print(f"保存的规划路径: {len(self.planned_paths)}")
        print(f"记录的实际轨迹: {len(self.actual_trajectories)}")
        print(f"图片保存目录: {self.save_dir}")

    def save_all_plots(self):
        """保存所有类型的图表"""
        print("🖼️ 生成所有可视化图表...")

        # 保存规划路径图
        for i in range(min(3, len(self.planned_paths))):  # 最多保存3个规划路径
            self.plot_planned_path_3d(path_index=i, show=False, save=True)

        # 保存实际轨迹图
        for i in range(min(3, len(self.actual_trajectories))):  # 最多保存3个实际轨迹
            self.plot_actual_trajectory_3d(trajectory_index=i, show=False, save=True)

        # 保存对比图
        for i in range(min(2, len(self.actual_trajectories))):
            self.plot_comparison_3d(trajectory_index=i, show=False, save=True)

        # 保存关节角度图
        for i in range(min(2, len(self.actual_trajectories))):
            self.plot_joint_angles(trajectory_index=i, show=False, save=True)

        print(f"✅ 所有图表已保存到 {self.save_dir}")


if __name__ == "__main__":
    # 测试可视化器
    workspace_bounds = np.array([
        [-1.2, 1.2],  # X bounds
        [-1.2, 1.2],  # Y bounds
        [0.0, 1.5]    # Z bounds
    ])

    visualizer = TrajectoryVisualizer(workspace_bounds)
    visualizer.print_statistics()
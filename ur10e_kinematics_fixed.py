"""
UR10e Forward and Inverse Kinematics - Fixed Version

基于官方UR10e运动学代码实现精确稳定的运动学解算
参考: /home/zar/Downloads/NavRL-main/isaac_gym_manipulator/ros_sources/universal_robot/ur_kinematics/src/ur_kin.cpp

单位：米
"""

import numpy as np
import math
from typing import List, Tuple, Optional


class UR10eKinematicsFixed:
    """
    UR10e运动学解算器 - 基于官方UR代码实现

    使用官方UR10e D-H参数和解析IK算法，提供高精度和稳定性
    """

    def __init__(self):
        # UR10e D-H参数 (单位：米) - 来自官方代码
        # #define UR10e_PARAMS
        self.d1 = 0.1807
        self.a2 = -0.6127  # 注意：负值
        self.a3 = -0.57155  # 注意：负值
        self.d4 = 0.17415
        self.d5 = 0.11985
        self.d6 = 0.11655

        # 关节限制 (弧度)
        self.joint_limits = [
            [-3.14159, 3.14159],   # shoulder_pan_joint
            [-3.14159, 3.14159],   # shoulder_lift_joint
            [-3.14159, 3.14159],   # elbow_joint
            [-3.14159, 3.14159],   # wrist_1_joint
            [-3.14159, 3.14159],   # wrist_2_joint
            [-3.14159, 3.14159]    # wrist_3_joint
        ]

        # 工作空间边界 (米) - UR10e工作空间
        self.workspace_radius = 1.3  # 大约工作半径
        self.workspace_min_height = 0.2
        self.workspace_max_height = 1.8

        # 数值计算阈值
        self.ZERO_THRESH = 1e-8
        self.PI = math.pi

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """
        正向运动学 - 基于官方代码实现

        Args:
            q: 6个关节角度 [q1, q2, q3, q4, q5, q6] (弧度)

        Returns:
            T: 4x4齐次变换矩阵
        """
        # 更鲁棒的长度检查（支持numpy数组和list）
        if hasattr(q, 'shape'):
            actual_length = q.shape[0] if len(q.shape) > 0 else len(q)
        else:
            actual_length = len(q)

        if actual_length != 6:
            raise ValueError(f"关节角度数组长度必须为6，实际为{actual_length}")

        q1, q2, q3, q4, q5, q6 = q

        # 预计算三角函数值（与官方代码一致）
        s1 = math.sin(q1); c1 = math.cos(q1)
        q23 = q2; q234 = q2
        s2 = math.sin(q2); c2 = math.cos(q2)
        s3 = math.sin(q3); c3 = math.cos(q3)
        q23 += q3; q234 += q3
        s4 = math.sin(q4); c4 = math.cos(q4)
        q234 += q4
        s5 = math.sin(q5); c5 = math.cos(q5)
        s6 = math.sin(q6); c6 = math.cos(q6)
        s23 = math.sin(q23); c23 = math.cos(q23)
        s234 = math.sin(q234); c234 = math.cos(q234)

        # 构造4x4变换矩阵（与官方代码一致）
        T = np.zeros((4, 4))

        # 第一行
        T[0, 0] = c234*c1*s5 - c5*s1
        T[0, 1] = c6*(s1*s5 + c234*c1*c5) - s234*c1*s6
        T[0, 2] = -s6*(s1*s5 + c234*c1*c5) - s234*c1*c6
        T[0, 3] = (self.d6*c234*c1*s5 - self.a3*c23*c1 - self.a2*c1*c2 -
                   self.d6*c5*s1 - self.d5*s234*c1 - self.d4*s1)

        # 第二行
        T[1, 0] = c1*c5 + c234*s1*s5
        T[1, 1] = -c6*(c1*s5 - c234*c5*s1) - s234*s1*s6
        T[1, 2] = s6*(c1*s5 - c234*c5*s1) - s234*c6*s1
        T[1, 3] = (self.d6*(c1*c5 + c234*s1*s5) + self.d4*c1 -
                   self.a3*c23*s1 - self.a2*c2*s1 - self.d5*s234*s1)

        # 第三行
        T[2, 0] = -s234*s5
        T[2, 1] = -c234*s6 - s234*c5*c6
        T[2, 2] = s234*c5*s6 - c234*c6
        T[2, 3] = (self.d1 + self.a3*s23 + self.a2*s2 -
                   self.d5*(c23*c4 - s23*s4) - self.d6*s5*(c23*s4 + s23*c4))

        # 第四行
        T[3, 0] = 0.0
        T[3, 1] = 0.0
        T[3, 2] = 0.0
        T[3, 3] = 1.0

        return T

    def inverse_kinematics(self, T: np.ndarray, q6_des: float = 0.0) -> List[np.ndarray]:
        """
        逆向运动学 - 基于官方代码实现

        Args:
            T: 4x4齐次变换矩阵
            q6_des: 期望的第6关节角度（默认0.0）

        Returns:
            solutions: 所有可能的关节角解列表
        """
        # 检查工作空间
        pos = T[:3, 3]
        if not self._is_in_workspace(pos):
            return []

        # 提取变换矩阵元素（与官方代码一致）
        T02 = -T[0, 2]; T00 = T[0, 0]; T01 = T[0, 1]; T03 = -T[0, 3]
        T12 = -T[1, 2]; T10 = T[1, 0]; T11 = T[1, 1]; T13 = -T[1, 3]
        T22 = T[2, 2]; T20 = -T[2, 0]; T21 = -T[2, 1]; T23 = T[2, 3]

        solutions = []

        # ////////////////////////////// shoulder rotate joint (q1) //////////////////////////////
        q1_solutions = self._solve_q1(T02, T03, T12, T13)

        for q1 in q1_solutions:
            # ////////////////////////////// wrist 2 joint (q5) //////////////////////////////
            q5_solutions = self._solve_q5(T03, T13, q1)

            for q5 in q5_solutions:
                # ////////////////////////////// wrist 3 joint (q6) //////////////////////////////
                q6 = self._solve_q6(q1, q5, T01, T11, T00, T10, q6_des)

                # ////////////////////////////// RRR joints (q2,q3,q4) //////////////////////////////
                q234_solutions = self._solve_q234(q1, q5, q6, T, T02, T12, T01, T11, T00, T10,
                                               T03, T13, T23, T20, T21)

                for q2, q3, q4 in q234_solutions:
                    solution = np.array([q1, q2, q3, q4, q5, q6])

                    # 检查关节限制
                    if self._check_joint_limits(solution):
                        solutions.append(solution)

        return solutions

    def inverse_kinematics_position(self, target_pos: np.ndarray, q6_des: float = 0.0) -> List[np.ndarray]:
        """
        仅基于位置的逆向运动学（为兼容环境接口）

        Args:
            target_pos: 目标位置 [x, y, z]
            q6_des: 期望的第6关节角度（默认0.0）

        Returns:
            solutions: 所有可能的关节角解列表
        """
        # 构造目标变换矩阵（姿态设为单位矩阵）
        T_target = np.eye(4)
        T_target[:3, 3] = target_pos

        return self.inverse_kinematics(T_target, q6_des)

    def _solve_q1(self, T02: float, T03: float, T12: float, T13: float) -> List[float]:
        """求解q1 - 肩关节旋转"""
        q1_solutions = []

        A = self.d6 * T12 - T13
        B = self.d6 * T02 - T03
        R = A * A + B * B

        if abs(A) < self.ZERO_THRESH:
            div = -self.d4 / B if abs(abs(self.d4) - abs(B)) >= self.ZERO_THRESH else -math.copysign(1, self.d4) * math.copysign(1, B)
            arcsin = math.asin(max(-1, min(1, div)))
            if abs(arcsin) < self.ZERO_THRESH:
                arcsin = 0.0
            if arcsin < 0.0:
                q1_solutions.append(arcsin + 2.0 * self.PI)
            else:
                q1_solutions.append(arcsin)
            q1_solutions.append(self.PI - arcsin)

        elif abs(B) < self.ZERO_THRESH:
            div = self.d4 / A if abs(abs(self.d4) - abs(A)) >= self.ZERO_THRESH else math.copysign(1, self.d4) * math.copysign(1, A)
            arccos = math.acos(max(-1, min(1, div)))
            q1_solutions.append(arccos)
            q1_solutions.append(2.0 * self.PI - arccos)

        elif self.d4 * self.d4 > R:
            return q1_solutions  # 无解

        else:
            arccos = math.acos(self.d4 / math.sqrt(R))
            arctan = math.atan2(-B, A)
            pos = arccos + arctan
            neg = -arccos + arctan

            if abs(pos) < self.ZERO_THRESH:
                pos = 0.0
            if abs(neg) < self.ZERO_THRESH:
                neg = 0.0

            if pos >= 0.0:
                q1_solutions.append(pos)
            else:
                q1_solutions.append(2.0 * self.PI + pos)

            if neg >= 0.0:
                q1_solutions.append(neg)
            else:
                q1_solutions.append(2.0 * self.PI + neg)

        return q1_solutions

    def _solve_q5(self, T03: float, T13: float, q1: float) -> List[float]:
        """求解q5 - 腕关节2"""
        q5_solutions = []

        numer = (T03 * math.sin(q1) - T13 * math.cos(q1) - self.d4)

        if abs(abs(numer) - abs(self.d6)) < self.ZERO_THRESH:
            div = math.copysign(1, numer) * math.copysign(1, self.d6)
        else:
            div = numer / self.d6

        # 确保在[-1, 1]范围内
        div = max(-1, min(1, div))
        arccos = math.acos(div)

        q5_solutions.append(arccos)
        q5_solutions.append(2.0 * self.PI - arccos)

        return q5_solutions

    def _solve_q6(self, q1: float, q5: float, T01: float, T11: float,
                 T00: float, T10: float, q6_des: float) -> float:
        """求解q6 - 腕关节3"""
        if abs(math.sin(q5)) < self.ZERO_THRESH:
            return q6_des
        else:
            q6 = math.atan2(math.copysign(1, math.sin(q5)) * -(T01 * math.sin(q1) - T11 * math.cos(q1)),
                           math.copysign(1, math.sin(q5)) * (T00 * math.sin(q1) - T10 * math.cos(q1)))
            if abs(q6) < self.ZERO_THRESH:
                q6 = 0.0
            if q6 < 0.0:
                q6 += 2.0 * self.PI
            return q6

    def _solve_q234(self, q1: float, q5: float, q6: float, T: np.ndarray,
                  T02: float, T12: float, T01: float, T11: float,
                  T00: float, T10: float, T03: float, T13: float,
                  T23: float, T20: float, T21: float) -> List[Tuple[float, float, float]]:
        """求解q2,q3,q4 - RRR关节"""
        solutions = []

        c1 = math.cos(q1); s1 = math.sin(q1)
        c5 = math.cos(q5); s5 = math.sin(q5)
        c6 = math.cos(q6); s6 = math.sin(q6)

        x04x = -s5 * (T02 * c1 + T12 * s1) - c5 * (s6 * (T01 * c1 + T11 * s1) - c6 * (T00 * c1 + T10 * s1))
        x04y = c5 * (T20 * c6 - T21 * s6) - T[2, 2] * s5
        p13x = (self.d5 * (s6 * (T00 * c1 + T10 * s1) + c6 * (T01 * c1 + T11 * s1)) -
                self.d6 * (T02 * c1 + T12 * s1) + T03 * c1 + T13 * s1)
        p13y = T23 - self.d1 - self.d6 * T[2, 2] + self.d5 * (T21 * c6 + T20 * s6)

        c3 = (p13x * p13x + p13y * p13y - self.a2 * self.a2 - self.a3 * self.a3) / (2.0 * self.a2 * self.a3)

        if abs(abs(c3) - 1.0) < self.ZERO_THRESH:
            c3 = math.copysign(1, c3)
        elif abs(c3) > 1.0:
            return solutions  # 无解

        arccos = math.acos(max(-1, min(1, c3)))
        q3_solutions = [arccos, 2.0 * self.PI - arccos]

        denom = self.a2 * self.a2 + self.a3 * self.a3 + 2 * self.a2 * self.a3 * c3
        s3 = math.sin(arccos)
        A = (self.a2 + self.a3 * c3)
        B = self.a3 * s3

        for q3 in q3_solutions:
            current_s3 = math.sin(q3)
            current_B = self.a3 * current_s3

            q2_1 = math.atan2((A * p13y - current_B * p13x) / denom,
                             (A * p13x + current_B * p13y) / denom)
            q2_2 = math.atan2((A * p13y + current_B * p13x) / denom,
                             (A * p13x - current_B * p13y) / denom)

            for q2 in [q2_1, q2_2]:
                c23 = math.cos(q2 + q3)
                s23 = math.sin(q2 + q3)
                q4 = math.atan2(c23 * x04y - s23 * x04x, x04x * c23 + x04y * s23)

                solutions.append((q2, q3, q4))

        return solutions

    def _is_in_workspace(self, pos: np.ndarray) -> bool:
        """检查位置是否在UR10e工作空间内（更宽松的���查）"""
        distance = np.linalg.norm(pos[:2])  # XY平面距离
        height = pos[2]

        # 使用更宽松的工作空间检查
        # UR10e的实际工作空间不是完美的圆柱形
        max_radius = self.workspace_radius + 0.2  # 增加0.2m容差

        if distance > max_radius or height < 0.0 or height > 2.0:
            return False

        return True

    def _check_joint_limits(self, q: np.ndarray) -> bool:
        """检查关节角度是否在限制范围内"""
        for i, (q_i, (min_q, max_q)) in enumerate(zip(q, self.joint_limits)):
            if q_i < min_q or q_i > max_q:
                return False
        return True

    def select_best_solution(self, solutions: List[np.ndarray],
                           current_q: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        从多个IK解中选择最佳解

        Args:
            solutions: IK解列表
            current_q: 当前关节角度（用于选择最接近的解）

        Returns:
            best_solution: 最佳解或None
        """
        if not solutions:
            return None

        if current_q is None:
            # 返回第一个有效解
            return solutions[0]

        # 选择与当前关节角度最接近的解
        min_distance = float('inf')
        best_solution = None

        for solution in solutions:
            # 计算关节角度距离（考虑关节旋转的周期性）
            distance = 0.0
            for i in range(6):
                diff = abs(solution[i] - current_q[i])
                # 考虑2π周期性
                diff = min(diff, 2 * self.PI - diff)
                distance += diff

            if distance < min_distance:
                min_distance = distance
                best_solution = solution

        return best_solution

    def get_end_effector_position(self, q: np.ndarray) -> np.ndarray:
        """获取末端执行器位置"""
        T = self.forward_kinematics(q)
        return T[:3, 3]

    def get_end_effector_orientation(self, q: np.ndarray) -> np.ndarray:
        """获取末端执行器姿态（欧拉角）"""
        T = self.forward_kinematics(q)
        R = T[:3, :3]

        # 从旋转矩阵计算欧拉角 (ZYX顺序)
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        if sy < 1e-6:
            # 奇异情况
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        else:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])

        return np.array([x, y, z])
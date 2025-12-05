"""
Task-Space Planner Interface

High-level interface for Task-Space RRT* planning and waypoint management.
Coordinates between global 3D path planning and local RL trajectory tracking.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import time

from ts_rrt_star import TaskSpaceRRTStar, TSWaypoint


@dataclass
class TSPlanningRequest:
    """Request for Task-Space path planning"""
    start_tcp: np.ndarray         # Current 3D TCP position [x, y, z]
    target_tcp: np.ndarray        # Target 3D TCP position [x, y, z]
    max_planning_time: float = 5.0    # Maximum planning time (seconds)
    tolerance: float = 0.15           # Goal tolerance (meters)


@dataclass
class TSPlanningResult:
    """Result of Task-Space path planning"""
    success: bool                     # Whether planning succeeded
    waypoints: List[TSWaypoint] = None    # Planned waypoints
    planning_time: float = 0.0            # Time taken for planning
    total_distance: float = 0.0           # Total path distance
    error_message: str = ""               # Error message if failed


class TaskSpacePlannerInterface:
    """
    High-level interface for Task-Space planning and waypoint management

    Responsibilities:
    - Manage Task-Space planning requests
    - Generate 3D waypoints for RL agent
    - Handle waypoint progression and tracking
    - Provide progress statistics and replanning logic
    """

    def __init__(self,
                 workspace_bounds: np.ndarray,
                 waypoint_spacing: float = 0.1,
                 replanning_threshold: float = 0.1,
                 max_waypoints: int = 50):
        """
        Initialize Task-Space planner interface

        Args:
            workspace_bounds: 3x2 array defining workspace [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
            waypoint_spacing: Desired spacing between waypoints (meters)
            replanning_threshold: Distance threshold for replanning (meters)
            max_waypoints: Maximum number of waypoints in plan
        """
        # Initialize Task-Space RRT* planner
        self.ts_planner = TaskSpaceRRTStar(
            workspace_bounds=workspace_bounds,
            max_iterations=2000,
            step_size=0.05,
            goal_bias=0.1,
            goal_tolerance=0.15
        )

        # Store workspace bounds for validation
        self.workspace_bounds = workspace_bounds

        # Configuration
        self.waypoint_spacing = waypoint_spacing
        self.replanning_threshold = replanning_threshold
        self.max_waypoints = max_waypoints

        # Current plan
        self.current_waypoints: List[TSWaypoint] = []
        self.current_waypoint_index: int = 0
        self.last_planning_time: float = 0.0

        # Statistics
        self.stats = {
            'planning_requests': 0,
            'successful_plans': 0,
            'failed_plans': 0,
            'replanning_events': 0,
            'total_planning_time': 0.0,
            'average_planning_time': 0.0,
            'waypoints_completed': 0,
            'total_waypoint_distance': 0.0
        }

    def plan_to_target(self, request: TSPlanningRequest) -> TSPlanningResult:
        """
        Plan path to target TCP position

        Args:
            request: Planning request with start, target, and constraints

        Returns:
            TSPlanningResult with waypoints or error information
        """
        self.stats['planning_requests'] += 1

        start_time = time.time()

        print(f"🛤️  Task-Space planning request:")
        print(f"   Start TCP: {request.start_tcp}")
        print(f"   Target TCP: {request.target_tcp}")
        print(f"   Max time: {request.max_planning_time}s")
        print(f"   Tolerance: {request.tolerance}m")

        # Validate inputs
        if not self._validate_tcp_position(request.start_tcp):
            return TSPlanningResult(
                success=False,
                error_message=f"Start TCP {request.start_tcp} is outside workspace"
            )

        if not self._validate_tcp_position(request.target_tcp):
            return TSPlanningResult(
                success=False,
                error_message=f"Target TCP {request.target_tcp} is outside workspace"
            )

        # Update planner tolerance if different
        if abs(self.ts_planner.goal_tolerance - request.tolerance) > 1e-6:
            self.ts_planner.goal_tolerance = request.tolerance

        # Plan path using Task-Space RRT*
        waypoints = self.ts_planner.plan(request.start_tcp, request.target_tcp)

        planning_time = time.time() - start_time

        if waypoints:
            # Update statistics
            self.stats['successful_plans'] += 1
            self.stats['total_planning_time'] += planning_time
            self.stats['average_planning_time'] = self.stats['total_planning_time'] / self.stats['successful_plans']

            # Resample waypoints if too many
            if len(waypoints) > self.max_waypoints:
                waypoints = self._sample_waypoints(waypoints, self.max_waypoints)

            # Update current plan
            self.current_waypoints = waypoints
            self.current_waypoint_index = 0
            self.last_planning_time = planning_time

            # Calculate total distance
            total_distance = sum(
                np.linalg.norm(waypoints[i].cartesian_position - waypoints[i+1].cartesian_position)
                for i in range(len(waypoints)-1)
            )

            print(f"✅ Task-Space planning successful:")
            print(f"   Waypoints: {len(waypoints)}")
            print(f"   Planning time: {planning_time:.3f}s")
            print(f"   Total distance: {total_distance:.4f}m")

            return TSPlanningResult(
                success=True,
                waypoints=waypoints,
                planning_time=planning_time,
                total_distance=total_distance
            )
        else:
            # Update statistics
            self.stats['failed_plans'] += 1

            error_msg = "Task-Space RRT* planning failed to find path"
            print(f"❌ Task-Space planning failed: {error_msg}")

            return TSPlanningResult(
                success=False,
                planning_time=planning_time,
                error_message=error_msg
            )

    def get_current_waypoint(self) -> Optional[TSWaypoint]:
        """
        Get current waypoint for RL agent to target

        Returns:
            Current waypoint or None if no plan available
        """
        if not self.current_waypoints or self.current_waypoint_index >= len(self.current_waypoints):
            return None

        return self.current_waypoints[self.current_waypoint_index]

    def advance_to_next_waypoint(self) -> bool:
        """
        Advance to next waypoint in plan

        Returns:
            True if advanced successfully, False if no more waypoints
        """
        if self.current_waypoint_index < len(self.current_waypoints) - 1:
            self.current_waypoint_index += 1
            self.stats['waypoints_completed'] += 1
            print(f"📍 Advanced to waypoint {self.current_waypoint_index + 1}/{len(self.current_waypoints)}")
            return True
        else:
            print("🎯 Reached final waypoint - trajectory completed!")
            return False

    def check_waypoint_reached(self, current_tcp: np.ndarray) -> bool:
        """
        Check if current waypoint has been reached

        Args:
            current_tcp: Current TCP position [x, y, z]

        Returns:
            True if current waypoint is reached
        """
        waypoint = self.get_current_waypoint()
        if waypoint is None:
            return False

        distance = np.linalg.norm(current_tcp - waypoint.cartesian_position)
        return distance <= waypoint.tolerance

    def update_progress(self, current_tcp: np.ndarray) -> bool:
        """
        Update progress and advance waypoints if current one is reached

        Args:
            current_tcp: Current TCP position

        Returns:
            True if advanced to next waypoint
        """
        # 添加调试信息
        waypoint_reached = self.check_waypoint_reached(current_tcp)
        if waypoint_reached:
            print(f"🎯 Waypoint reached! Current TCP: [{current_tcp[0]:.4f}, {current_tcp[1]:.4f}, {current_tcp[2]:.4f}]")
            waypoint = self.get_current_waypoint()
            if waypoint:
                print(f"🚩 Target Waypoint: [{waypoint.cartesian_position[0]:.4f}, {waypoint.cartesian_position[1]:.4f}, {waypoint.cartesian_position[2]:.4f}], Tolerance: {waypoint.tolerance:.4f}")
        
        if waypoint_reached:
            return self.advance_to_next_waypoint()
        return False

    def check_replanning_needed(self, current_tcp: np.ndarray, current_target: np.ndarray) -> bool:
        """
        Check if replanning is needed due to deviation from plan or target change

        Args:
            current_tcp: Current TCP position
            current_target: Current target TCP position

        Returns:
            True if replanning is needed
        """
        if not self.current_waypoints:
            return True

        # Check if we've reached the final waypoint
        if self.current_waypoint_index >= len(self.current_waypoints) - 1:
            final_waypoint = self.current_waypoints[-1]
            distance_to_goal = np.linalg.norm(current_tcp - final_waypoint.cartesian_position)
            if distance_to_goal < final_waypoint.tolerance:
                return False  # Goal reached

        # Check if target has changed significantly
        current_waypoint = self.get_current_waypoint()
        if current_waypoint:
            target_distance = np.linalg.norm(current_target - current_waypoint.cartesian_position)
            if target_distance > self.replanning_threshold:
                print(f"🔄 Target deviation: {target_distance:.4f}m > {self.replanning_threshold:.4f}m, replanning needed")
                self.stats['replanning_events'] += 1
                return True

        # Check if we're too far from current waypoint
        if current_waypoint:
            waypoint_distance = np.linalg.norm(current_tcp - current_waypoint.cartesian_position)
            if waypoint_distance > current_waypoint.tolerance * 3:  # Allow some tolerance
                print(f"🔄 Waypoint deviation: {waypoint_distance:.4f}m, replanning may be needed")
                return True

        return False

    def get_distance_to_current_waypoint(self, current_tcp: np.ndarray) -> float:
        """
        Get distance to current waypoint

        Args:
            current_tcp: Current TCP position

        Returns:
            Distance to current waypoint, or infinity if no waypoint
        """
        waypoint = self.get_current_waypoint()
        if waypoint:
            return np.linalg.norm(current_tcp - waypoint.cartesian_position)
        return float('inf')

    def get_remaining_waypoints(self) -> int:
        """Get number of remaining waypoints"""
        if not self.current_waypoints:
            return 0
        return len(self.current_waypoints) - self.current_waypoint_index

    def reset_plan(self):
        """Reset current plan"""
        self.current_waypoints = []
        self.current_waypoint_index = 0
        print("🔄 Task-Space plan reset")

    def get_planning_progress(self) -> Dict[str, Any]:
        """
        Get current planning progress

        Returns:
            Dictionary with progress information
        """
        if not self.current_waypoints:
            return {
                'has_plan': False,
                'current_waypoint': None,
                'total_waypoints': 0,
                'remaining_waypoints': 0,
                'progress_percentage': 0.0
            }

        progress = (self.current_waypoint_index / len(self.current_waypoints)) * 100

        return {
            'has_plan': True,
            'current_waypoint': self.current_waypoint_index + 1,
            'total_waypoints': len(self.current_waypoints),
            'remaining_waypoints': self.get_remaining_waypoints(),
            'progress_percentage': progress
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get planning statistics"""
        stats = self.stats.copy()
        stats.update(self.ts_planner.get_statistics())

        # Add derived statistics
        if stats['planning_requests'] > 0:
            stats['success_rate'] = stats['successful_plans'] / stats['planning_requests']
        else:
            stats['success_rate'] = 0.0

        return stats

    def _validate_tcp_position(self, tcp_position: np.ndarray) -> bool:
        """
        Validate TCP position is within workspace bounds

        Args:
            tcp_position: 3D TCP position

        Returns:
            True if position is valid
        """
        if len(tcp_position) != 3:
            return False

        for i in range(3):
            if tcp_position[i] < self.workspace_bounds[i, 0] - 0.01 or \
               tcp_position[i] > self.workspace_bounds[i, 1] + 0.01:
                return False

        return True

    def _sample_waypoints(self, waypoints: List[TSWaypoint], target_count: int) -> List[TSWaypoint]:
        """
        Sample waypoints to reduce count while preserving path shape

        Args:
            waypoints: Original waypoints
            target_count: Target number of waypoints

        Returns:
            Sampled waypoints
        """
        if len(waypoints) <= target_count:
            return waypoints

        # Always include first and last waypoints
        sampled = [waypoints[0]]

        # Sample intermediate waypoints
        step = (len(waypoints) - 1) / (target_count - 1)
        for i in range(1, target_count - 1):
            index = int(i * step)
            sampled.append(waypoints[index])

        sampled.append(waypoints[-1])

        return sampled


def test_task_space_planner_interface():
    """Test Task-Space planner interface"""
    print("🧪 Testing Task-Space Planner Interface")

    # Define UR10e workspace bounds (conservative estimate)
    workspace_bounds = np.array([
        [-0.8, 0.8],    # X-axis range
        [-0.8, 0.8],    # Y-axis range
        [0.1, 1.0]      # Z-axis range (considering floor)
    ])

    # Create interface
    interface = TaskSpacePlannerInterface(
        workspace_bounds=workspace_bounds,
        waypoint_spacing=0.1,
        replanning_threshold=0.1
    )

    # Create planning request
    start_tcp = np.array([0.5, -0.2, 0.3])   # Start TCP position
    target_tcp = np.array([-0.3, 0.4, 0.7])  # Target TCP position

    request = TSPlanningRequest(
        start_tcp=start_tcp,
        target_tcp=target_tcp,
        max_planning_time=5.0,
        tolerance=0.15
    )

    # Plan path
    result = interface.plan_to_target(request)

    if result.success:
        print(f"✅ Planning successful with {len(result.waypoints)} waypoints")

        # Test waypoint progression
        waypoint = interface.get_current_waypoint()
        print(f"   Current waypoint: {waypoint}")

        # Test distance calculation
        distance = interface.get_distance_to_current_waypoint(start_tcp)
        print(f"   Distance to waypoint: {distance:.4f}m")

        # Test progress
        progress = interface.get_planning_progress()
        print(f"📍 Progress: {progress}")

        # Test statistics
        stats = interface.get_statistics()
        print(f"📊 Statistics: {stats}")
    else:
        print(f"❌ Planning failed: {result.error_message}")


if __name__ == "__main__":
    test_task_space_planner_interface()
"""
Task-Space RRT* (Rapidly-exploring Random Tree Star) Path Planner

Implementation of RRT* algorithm in 3D task space for UR10e robotic arm.
Plans collision-free paths in Cartesian TCP space, completely decoupled from joint space.

Key Features:
- 3D Cartesian space planning (x, y, z)
- Simple Euclidean distance metrics
- Workspace boundary constraints
- Fast convergence with goal biasing
- Cost optimization for shortest paths
"""

import numpy as np
import math
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@dataclass
class TSNode:
    """Task-Space RRT* tree node"""
    position: np.ndarray      # 3D TCP position [x, y, z]
    cost: float               # Cost from root (path length)
    parent: Optional['TSNode'] = None
    children: List['TSNode'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def __eq__(self, other):
        """Override equality comparison to use object identity"""
        if not isinstance(other, TSNode):
            return False
        return id(self) == id(other)  # Compare by object identity, not position
    
    def __hash__(self):
        """Override hash to use object identity"""
        return id(self)



@dataclass
class TSWaypoint:
    """Task-Space Waypoint for RL local planner"""
    cartesian_position: np.ndarray   # 3D TCP target position
    tolerance: float                 # Position tolerance (meters)

    def __str__(self):
        return f"TSWaypoint(pos={self.cartesian_position}, tol={self.tolerance:.3f}m)"


class TaskSpaceRRTStar:
    """
    Task-Space RRT* path planner for UR10e robotic arm

    Plans paths in 3D Cartesian space, completely decoupled from joint configurations.
    The RL agent learns to track these waypoints through torque control.
    """

    def __init__(self,
                 workspace_bounds: np.ndarray,
                 max_iterations: int = 2000,
                 step_size: float = 0.05,
                 goal_bias: float = 0.1,
                 goal_tolerance: float = 0.15,
                 rewire_radius: float = None):
        """
        Initialize Task-Space RRT* planner

        Args:
            workspace_bounds: 3x2 array defining workspace [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
            max_iterations: Maximum planning iterations
            step_size: Maximum step size for tree expansion (meters)
            goal_bias: Probability of sampling goal (0-1)
            goal_tolerance: Distance tolerance to goal (meters)
            rewire_radius: Radius for rewiring tree (meters)
        """
        self.workspace_bounds = workspace_bounds
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.goal_tolerance = goal_tolerance

        # Calculate rewire radius if not provided (based on workspace size)
        if rewire_radius is None:
            workspace_volume = np.prod(workspace_bounds[:, 1] - workspace_bounds[:, 0])
            self.rewire_radius = 0.1 * (workspace_volume ** (1/3))  # Heuristic based on workspace size
        else:
            self.rewire_radius = rewire_radius

        # Tree nodes
        self.nodes: List[TSNode] = []

        # Statistics
        self.stats = {
            'iterations': 0,
            'nodes_explored': 0,
            'path_length': 0.0,
            'planning_time': 0.0,
            'success': False
        }

    def plan(self,
             start_tcp: np.ndarray,
             goal_tcp: np.ndarray) -> Optional[List[TSWaypoint]]:
        """
        Plan path from start TCP position to goal TCP position

        Args:
            start_tcp: 3D start TCP position [x, y, z]
            goal_tcp: 3D goal TCP position [x, y, z]

        Returns:
            List of TSWaypoints if successful, None otherwise
        """
        import time
        start_time = time.time()

        print(f"🛤️  Starting Task-Space RRT* planning...")
        print(f"   Start TCP: {start_tcp}")
        print(f"   Goal TCP: {goal_tcp}")
        print(f"   Workspace: {self.workspace_bounds}")
        print(f"   Max iterations: {self.max_iterations}")
        print(f"   Step size: {self.step_size}m")

        # Validate inputs
        if not self._is_in_workspace(start_tcp):
            print(f"❌ Start TCP {start_tcp} is outside workspace")
            return None

        if not self._is_in_workspace(goal_tcp):
            print(f"❌ Goal TCP {goal_tcp} is outside workspace")
            return None

        # Check minimum reachability (simple heuristic)
        distance = np.linalg.norm(goal_tcp - start_tcp)
        if distance.item() < self.goal_tolerance:
            print("✅ Start and goal are already within tolerance")
            return [TSWaypoint(goal_tcp, self.goal_tolerance)]

        # Initialize tree with start node
        self.nodes = [TSNode(position=start_tcp.copy(), cost=0.0)]
        goal_node = None

        # RRT* main loop
        for iteration in range(self.max_iterations):
            self.stats['iterations'] = iteration + 1

            # Sample random configuration
            if np.random.random() < self.goal_bias:
                sample = goal_tcp.copy()
            else:
                sample = self._sample_random_tcp_position()

            # Find nearest node in tree
            nearest_node = self._find_nearest_node(sample)

            # Steer towards sample
            new_position = self._steer(nearest_node.position, sample)

            # Check if new position is valid (workspace constraints)
            if not self._is_in_workspace(new_position):
                continue

            # Simple collision check - extend this with proper collision detection
            if not self._is_path_collision_free(nearest_node.position, new_position):
                continue

            # Find best parent (cost optimization)
            best_parent = self._find_best_parent(new_position, nearest_node)

            # Create new node
            new_node = TSNode(
                position=new_position,
                cost=best_parent.cost + np.linalg.norm(best_parent.position - new_position),
                parent=best_parent
            )
            best_parent.children.append(new_node)
            self.nodes.append(new_node)

            # Rewire tree for cost optimization
            self._rewire_tree(new_node)

            # Check if goal is reached
            if np.linalg.norm(new_position - goal_tcp) < self.goal_tolerance:
                goal_node = new_node
                print(f"✅ Goal reached at iteration {iteration + 1}")
                break

            # Progress update
            if (iteration + 1) % 200 == 0:
                distance_to_goal = min(np.linalg.norm(node.position - goal_tcp) for node in self.nodes)
                print(f"   Iteration {iteration + 1}: {len(self.nodes)} nodes, min distance: {distance_to_goal:.4f}m")

        # Extract path if goal was reached
        if goal_node is not None:
            path = self._extract_path(goal_node)
            waypoints = self._create_waypoints(path)

            # Update statistics
            self.stats['planning_time'] = time.time() - start_time
            self.stats['nodes_explored'] = len(self.nodes)
            self.stats['path_length'] = sum(np.linalg.norm(waypoints[i].cartesian_position - waypoints[i+1].cartesian_position)
                                           for i in range(len(waypoints)-1))
            self.stats['success'] = True

            print(f"🎯 Task-Space planning successful!")
            print(f"   Path waypoints: {len(waypoints)}")
            print(f"   Path length: {self.stats['path_length']:.4f}m")
            print(f"   Planning time: {self.stats['planning_time']:.3f}s")

            return waypoints
        else:
            self.stats['planning_time'] = time.time() - start_time
            self.stats['success'] = False

            # Report best effort
            if self.nodes:
                best_distance = min(np.linalg.norm(node.position - goal_tcp) for node in self.nodes)
                print(f"❌ Planning failed: best distance to goal {best_distance:.4f}m after {self.max_iterations} iterations")
            else:
                print(f"❌ Planning failed: no nodes generated")
            return None

    def _sample_random_tcp_position(self) -> np.ndarray:
        """Sample random TCP position within workspace bounds"""
        position = np.zeros(3)
        for i in range(3):
            position[i] = np.random.uniform(self.workspace_bounds[i, 0], self.workspace_bounds[i, 1])
        return position

    def _find_nearest_node(self, position: np.ndarray) -> TSNode:
        """Find nearest node in tree to given position"""
        min_distance = float('inf')
        nearest_node = self.nodes[0]

        for node in self.nodes:
            distance = np.linalg.norm(node.position - position)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node

        return nearest_node

    def _steer(self, from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
        """Steer from one position towards another"""
        direction = to_pos - from_pos
        distance = np.linalg.norm(direction)

        if distance <= self.step_size:
            return to_pos
        else:
            return from_pos + (direction / distance) * self.step_size

    def _is_in_workspace(self, position: np.ndarray) -> bool:
        """Check if position is within workspace bounds"""
        for i in range(3):
            if position[i] < self.workspace_bounds[i, 0] - 0.01 or position[i] > self.workspace_bounds[i, 1] + 0.01:
                return False
        return True

    def _is_path_collision_free(self, from_pos: np.ndarray, to_pos: np.ndarray) -> bool:
        """
        Check if path between two positions is collision-free

        Simplified collision checking - can be extended with proper collision detection
        """
        # Simple collision check using intermediate positions
        steps = 5  # Check 5 intermediate points
        for i in range(steps + 1):
            t = i / steps
            intermediate = from_pos + t * (to_pos - from_pos)

            if not self._is_in_workspace(intermediate):
                return False

            # TODO: Add proper collision detection with obstacles
            # For now, only check workspace bounds

        return True

    def _find_best_parent(self, position: np.ndarray, default_parent: TSNode) -> TSNode:
        """Find best parent for new position (cost optimization)"""
        # Find nodes within rewire radius
        nearby_nodes = []
        for node in self.nodes:
            if np.linalg.norm(node.position - position) <= self.rewire_radius:
                if self._is_path_collision_free(node.position, position):
                    nearby_nodes.append(node)

        if not nearby_nodes:
            return default_parent

        # Find node with minimum cost
        best_parent = nearby_nodes[0]
        min_cost = nearby_nodes[0].cost + np.linalg.norm(nearby_nodes[0].position - position)

        for node in nearby_nodes[1:]:
            cost = node.cost + np.linalg.norm(node.position - position)
            if cost < min_cost:
                min_cost = cost
                best_parent = node

        return best_parent

    def _rewire_tree(self, new_node: TSNode):
        """Rewire tree for cost optimization"""
        # Find nodes within rewire radius
        for node in self.nodes:
            if id(node) == id(new_node):
                continue

            if np.linalg.norm(node.position - new_node.position) <= self.rewire_radius:
                if self._is_path_collision_free(new_node.position, node.position):
                    new_cost = new_node.cost + np.linalg.norm(new_node.position - node.position)

                    if new_cost < node.cost:
                        # Rewire - 安全地移除节点
                        if node.parent and node in node.parent.children:
                            # 使用索引安全移除
                            for i, child in enumerate(node.parent.children):
                                if id(child) == id(node):  # 明确的对象身份比较
                                    del node.parent.children[i]
                                    break

                        node.parent = new_node
                        node.cost = new_cost
                        new_node.children.append(node)
    def _extract_path(self, goal_node: TSNode) -> List[np.ndarray]:
        """Extract path from root to goal node"""
        path = []
        current = goal_node

        while current is not None:
            path.append(current.position.copy())
            current = current.parent

        path.reverse()
        return path

    def _create_waypoints(self, path: List[np.ndarray]) -> List[TSWaypoint]:
        """Create waypoints from path"""
        waypoints = []

        for i, pos in enumerate(path):
            # Tighter tolerance for final waypoint
            tolerance = 0.15 if i < len(path) - 1 else self.goal_tolerance

            waypoint = TSWaypoint(
                cartesian_position=pos,
                tolerance=tolerance
            )
            waypoints.append(waypoint)

        return waypoints

    def visualize_tree(self, goal_tcp: np.ndarray = None, save_path: str = None):
        """Visualize RRT* tree in 3D"""
        if not self.nodes:
            print("⚠️ No tree to visualize")
            return

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot tree nodes
        positions = np.array([node.position for node in self.nodes])
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  c='blue', s=10, alpha=0.6, label='Tree Nodes')

        # Plot edges
        for node in self.nodes:
            if node.parent:
                ax.plot([node.parent.position[0], node.position[0]],
                       [node.parent.position[1], node.position[1]],
                       [node.parent.position[2], node.position[2]],
                       'b-', alpha=0.3, linewidth=0.5)

        # Plot workspace bounds
        self._plot_workspace_bounds(ax)

        # Plot start and goal if provided
        if self.nodes:
            start_pos = self.nodes[0].position
            ax.scatter([start_pos[0]], [start_pos[1]], [start_pos[2]],
                      c='green', s=100, marker='o', label='Start')

        if goal_tcp is not None:
            ax.scatter([goal_tcp[0]], [goal_tcp[1]], [goal_tcp[2]],
                      c='red', s=100, marker='*', label='Goal')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Task-Space RRT* Tree (3D)')
        ax.legend()

        # Set equal aspect ratio
        max_range = np.array([self.workspace_bounds[:, 1] - self.workspace_bounds[:, 0]]).max() / 2.0
        mid_x = (self.workspace_bounds[0, 0] + self.workspace_bounds[0, 1]) * 0.5
        mid_y = (self.workspace_bounds[1, 0] + self.workspace_bounds[1, 1]) * 0.5
        mid_z = (self.workspace_bounds[2, 0] + self.workspace_bounds[2, 1]) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Tree visualization saved to {save_path}")
        else:
            plt.show()

    def _plot_workspace_bounds(self, ax):
        """Plot workspace boundaries"""
        # Get workspace bounds
        x_min, x_max = self.workspace_bounds[0]
        y_min, y_max = self.workspace_bounds[1]
        z_min, z_max = self.workspace_bounds[2]

        # Draw workspace edges (simplified)
        # Bottom face
        xx, yy = np.meshgrid([x_min, x_max], [y_min, y_max])
        ax.plot_surface(xx, yy, np.ones_like(xx) * z_min, alpha=0.1, color='gray')

        # Top face
        ax.plot_surface(xx, yy, np.ones_like(xx) * z_max, alpha=0.1, color='gray')

    def get_statistics(self) -> Dict[str, Any]:
        """Get planning statistics"""
        return self.stats.copy()


def test_task_space_rrt_star():
    """Test Task-Space RRT* planner with example"""
    print("🧪 Testing Task-Space RRT* Planner")

    # Define UR10e workspace bounds (matching config.yaml)
    workspace_bounds = np.array([
        [-1.4, 1.4],    # X-axis range
        [-1.4, 1.4],    # Y-axis range
        [0.1, 1.0]      # Z-axis range (considering floor)
    ])

    # Create planner
    planner = TaskSpaceRRTStar(
        workspace_bounds=workspace_bounds,
        max_iterations=1000,
        step_size=0.05,
        goal_bias=0.1
    )

    # Define start and goal TCP positions
    start_tcp = np.array([0.5, -0.2, 0.3])  # Start position
    goal_tcp = np.array([-0.3, 0.4, 0.7])   # Goal position

    # Plan path
    waypoints = planner.plan(start_tcp, goal_tcp)

    if waypoints:
        print(f"✅ Task-Space planning successful with {len(waypoints)} waypoints")

        # Print first few waypoints
        for i, waypoint in enumerate(waypoints[:5]):
            print(f"   Waypoint {i+1}: {waypoint}")

        # Print statistics
        stats = planner.get_statistics()
        print(f"📊 Planning Statistics: {stats}")

        # Visualize tree (optional)
        # planner.visualize_tree(goal_tcp)
    else:
        print("❌ Task-Space planning failed")

        # Still visualize tree for debugging
        planner.visualize_tree(goal_tcp)


if __name__ == "__main__":
    test_task_space_rrt_star()
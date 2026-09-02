"""
sim_robots.py
Simulation & Pathfinding Support module - SIH 2026, PS 26123
Handles: spawning robots on the grid, maintaining a task queue,
and nearest-robot task assignment.

Role: Simulation & Pathfinding Support
"""

import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional


# ---------- 1. Basic data structures ----------

class RobotStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"


class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    COMPLETED = "completed"


@dataclass
class Task:
    task_id: int
    pickup: tuple        # (row, col)
    dropoff: tuple        # (row, col)
    status: TaskStatus = TaskStatus.PENDING
    assigned_robot: Optional[int] = None


@dataclass
class Robot:
    robot_id: int
    position: tuple
    status: RobotStatus = RobotStatus.IDLE
    current_task: Optional[Task] = None


# ---------- 2. Grid + spawning ----------

class WarehouseGrid:
    def __init__(self, rows, cols, blocked_cells=None):
        self.rows = rows
        self.cols = cols
        self.blocked_cells = set(blocked_cells or [])

    def is_valid(self, cell):
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols and cell not in self.blocked_cells


def spawn_robots(grid: WarehouseGrid, num_robots: int, seed=None) -> list:
    """Places num_robots robots on distinct, valid, unblocked cells."""
    if seed is not None:
        random.seed(seed)

    all_cells = [
        (r, c)
        for r in range(grid.rows)
        for c in range(grid.cols)
        if grid.is_valid((r, c))
    ]

    if num_robots > len(all_cells):
        raise ValueError("Not enough free cells to spawn that many robots")

    chosen_cells = random.sample(all_cells, num_robots)
    return [Robot(robot_id=i, position=chosen_cells[i]) for i in range(num_robots)]


# ---------- 3. Task queue ----------

class TaskQueue:
    def __init__(self):
        self._tasks = []
        self._next_id = 0

    def add_task(self, pickup, dropoff):
        task = Task(task_id=self._next_id, pickup=pickup, dropoff=dropoff)
        self._tasks.append(task)
        self._next_id += 1
        return task

    def pending_tasks(self):
        return [t for t in self._tasks if t.status == TaskStatus.PENDING]

    def all_tasks(self):
        return self._tasks


# ---------- 4. Distance + nearest-robot assignment ----------

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def find_nearest_idle_robot(robots: list, target_cell) -> Optional[Robot]:
    idle_robots = [r for r in robots if r.status == RobotStatus.IDLE]
    if not idle_robots:
        return None
    return min(idle_robots, key=lambda r: manhattan_distance(r.position, target_cell))


def assign_tasks(robots: list, queue: TaskQueue):
    """
    Walks through pending tasks in order and assigns each one to the
    nearest currently-idle robot (measured from the task's pickup point).
    Stops early if idle robots run out.
    """
    assignments = []
    for task in queue.pending_tasks():
        robot = find_nearest_idle_robot(robots, task.pickup)
        if robot is None:
            break

        robot.status = RobotStatus.BUSY
        robot.current_task = task
        task.status = TaskStatus.ASSIGNED
        task.assigned_robot = robot.robot_id

        assignments.append((robot.robot_id, task.task_id))

    return assignments


# ---------- 5. Visualization ----------

def visualize(grid: WarehouseGrid, robots: list, queue: TaskQueue,
              filename: str = "warehouse.png"):
    """
    Draws the current grid state as a PNG:
      - gray cells   = blocked (obstacles/shelves)
      - blue dots    = idle robots
      - orange dots  = busy robots
      - green stars  = task pickup points
      - red squares  = task dropoff points
      - dashed lines = robot -> assigned task's pickup point
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw blocked cells
    for (r, c) in grid.blocked_cells:
        ax.add_patch(patches.Rectangle((c, r), 1, 1, color="#888888"))

    # Draw grid lines
    for r in range(grid.rows + 1):
        ax.axhline(r, color="lightgray", linewidth=0.5)
    for c in range(grid.cols + 1):
        ax.axvline(c, color="lightgray", linewidth=0.5)

    # Draw tasks (pickup = green star, dropoff = red square)
    for task in queue.all_tasks():
        pr, pc = task.pickup
        dr, dc = task.dropoff
        ax.scatter(pc + 0.5, pr + 0.5, marker="*", s=250, color="green", zorder=3)
        ax.scatter(dc + 0.5, dr + 0.5, marker="s", s=80, color="red", zorder=3)

        # Dashed line from assigned robot to its task's pickup point
        if task.assigned_robot is not None:
            robot = next(r for r in robots if r.robot_id == task.assigned_robot)
            rr, rc = robot.position
            ax.plot([rc + 0.5, pc + 0.5], [rr + 0.5, pr + 0.5],
                     linestyle="--", color="black", linewidth=1, zorder=2)

    # Draw robots (blue = idle, orange = busy)
    for robot in robots:
        rr, rc = robot.position
        color = "#1f77b4" if robot.status == RobotStatus.IDLE else "#ff7f0e"
        ax.scatter(rc + 0.5, rr + 0.5, s=300, color=color, zorder=4,
                   edgecolors="black", linewidths=1)
        ax.text(rc + 0.5, rr + 0.5, str(robot.robot_id), color="white",
                ha="center", va="center", fontsize=9, fontweight="bold", zorder=5)

    ax.set_xlim(0, grid.cols)
    ax.set_ylim(0, grid.rows)
    ax.invert_yaxis()   # row 0 at top, like a matrix
    ax.set_aspect("equal")
    ax.set_xticks(range(grid.cols + 1))
    ax.set_yticks(range(grid.rows + 1))
    ax.set_title("Warehouse Grid — Robot Positions & Task Assignments")

    legend_handles = [
        patches.Patch(color="#888888", label="Blocked cell"),
        patches.Patch(color="#1f77b4", label="Idle robot"),
        patches.Patch(color="#ff7f0e", label="Busy robot"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="green",
                   markersize=15, label="Task pickup"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="red",
                   markersize=10, label="Task dropoff"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {filename}")
    plt.show()   # opens a window showing the plot
    plt.close()


# ---------- 6. Demo ----------

if __name__ == "__main__":
    grid = WarehouseGrid(rows=10, cols=10, blocked_cells=[(3, 3), (3, 4), (5, 5)])

    robots = spawn_robots(grid, num_robots=6, seed=42)
    print("Spawned robots:")
    for r in robots:
        print(f"  Robot {r.robot_id} at {r.position} [{r.status.value}]")

    queue = TaskQueue()
    queue.add_task(pickup=(1, 1), dropoff=(8, 8))
    queue.add_task(pickup=(9, 0), dropoff=(0, 9))
    queue.add_task(pickup=(4, 4), dropoff=(2, 7))

    print("\nAssigning tasks...")
    assignments = assign_tasks(robots, queue)
    for robot_id, task_id in assignments:
        print(f"  Task {task_id} -> Robot {robot_id}")

    print("\nRemaining pending tasks:", [t.task_id for t in queue.pending_tasks()])

    visualize(grid, robots, queue, filename="warehouse.png")
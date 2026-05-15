from __future__ import annotations

from typing import Optional

from env import DeliveryEnv, Order


class Solver:
    """
    Base class cho solver.
    """

    def __init__(self, env: DeliveryEnv):
        if not isinstance(env, DeliveryEnv):
            raise TypeError("Solver chỉ hỗ trợ khởi tạo dạng Solver(env: DeliveryEnv).")

        self.env: DeliveryEnv = env
        self.cfg = env.public_cfg if hasattr(env, "public_cfg") else env.cfg
        self.grid = env.grid
        self.orders: list[Order] = []

    def run(self) -> dict:
        raise NotImplementedError


def default_result(method: str, cfg: dict, orders: Optional[list[Order]] = None) -> dict:
    """Kết quả mặc định cho các solver skeleton chưa cài đặt."""
    total_orders = int(cfg.get("G", len(orders) if orders is not None else 0))
    return {
        "method": method,
        "config_name": cfg.get("name", "unknown"),
        "total_orders": total_orders,
        "orders_generated": 0,
        "delivered": 0,
        "on_time": 0,
        "late": 0,
        "missed": total_orders,
        "delivery_rate": 0.0,
        "on_time_rate": 0.0,
        "total_reward": 0.0,
        "total_movecost": 0.0,
        "net_reward": 0.0,
        "elapsed_sec": 0.0,
        "shipper_rewards": [],
        "status": "TODO",
    }
import collections
from typing import List, Tuple

def bfs_path(grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]) -> List[str]:
    """Tìm một đường đi hợp lệ từ start đến goal bằng BFS."""
    if start == goal:
        return []

    n = len(grid)
    q = collections.deque([(start[0], start[1], [])])
    visited = {start}

    for_action = [("U", -1, 0), ("D", 1, 0), ("L", 0, -1), ("R", 0, 1)]

    while q:
        r, c, path = q.popleft()
        for action, dr, dc in for_action:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0 and (nr, nc) not in visited:
                if (nr, nc) == goal:
                    return path + [action]
                visited.add((nr, nc))
                q.append((nr, nc, path + [action]))
    return []
from __future__ import annotations
from typing import List, Optional

from env import DeliveryEnv, Order
from solvers.solver import Solver, default_result


class VRPOrToolsSolver(Solver):
    """Sinh viên cài đặt thuật toán VRP + OR-Tools tại đây."""

    def __init__(self, env_or_cfg, grid: Optional[List[List[int]]] = None, orders: Optional[List[Order]] = None):
        super().__init__(env_or_cfg, grid, orders)

    def run(self) -> dict:
        # TODO: chuyển nghiệm VRP thành mô phỏng từng bước và trả về dict kết quả.
        return default_result("VRP + OR-Tools", self.cfg, self.orders)

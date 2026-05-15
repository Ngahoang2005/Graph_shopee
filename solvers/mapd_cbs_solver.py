from __future__ import annotations
from typing import List, Optional

from env import DeliveryEnv, Order
from solvers.solver import Solver, default_result


class MAPDCBSSolver(Solver):
    """Sinh viên cài đặt MAPD với Conflict-Based Search tại đây."""

    def __init__(self, env_or_cfg, grid: Optional[List[List[int]]] = None, orders: Optional[List[Order]] = None):
        super().__init__(env_or_cfg, grid, orders)

    def run(self) -> dict:
        # TODO: sinh task, chạy CBS, mô phỏng và trả về dict kết quả.
        return default_result("MAPD-CBS", self.cfg, self.orders)

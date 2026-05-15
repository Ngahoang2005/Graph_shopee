from __future__ import annotations

import random
import time
from typing import Dict, List, Optional

from env import DIRS, DeliveryEnv, Order, Shipper, delivery_reward, manhattan, move_cost, SEED
from solvers.solver import Solver
from solvers.routing_utils import compute_path
from solvers.simulation_utils import init_shippers, execute_moves, pickup_phase, delivery_phase, compute_result


class GreedyBFS(Solver):
    """
    Baseline Greedy BFS đơn giản.
    Strategy:
    - idle shipper chọn đơn gần nhất
    - BFS để tìm đường
    - giao đơn theo earliest deadline
    """

    def __init__(self, env_or_cfg, grid: Optional[List[List[int]]] = None, orders: Optional[List[Order]] = None):
        super().__init__(env_or_cfg, grid, orders)
        order_list = self.orders
        self.orders: Dict[int, Order] = {o.id: o for o in order_list}
        self.N = self.cfg["N"]
        self.T = self.cfg["T"]
        self.rng = random.Random(SEED + 10)

    # Init
    def _init_shippers(self) -> List[Shipper]:
        free_cells = [
            (r, c)
            for r in range(self.N)
            for c in range(self.N)
            if self.grid[r][c] == 0
        ]
        self.rng.shuffle(free_cells)

        return init_shippers(
            self.cfg,
            free_cells,
        )

    # Task Assigment
    def _choose_order(self, sh: Shipper, pending: List[Order], t: int) -> Optional[Order]:
        candidates = [o for o in pending if not o.picked and sh.can_pickup(o, self.orders)]
        if not candidates:
            return None

        # Bản baseline chỉ chọn đơn gần nhất, nếu hòa thì ưu tiên deadline sớm hơn.
        return min(
            candidates,
            key=lambda o: (
                manhattan(sh.r, sh.c, o.sx, o.sy),
                o.et,
                -o.p,
                o.id,
            ),
        )

    def _choose_delivery_from_bag(self, sh: Shipper) -> Optional[int]:
        if not sh.bag:
            return None
        # deadline sớm nhất được chọn trước
        return min(sh.bag, key=lambda oid: self.orders[oid].et)

    def _assign_orders(self, shippers: List[Shipper], pending: List[Order], t: int) -> None:
        reserved: set[int] = set()
        for sh in shippers:
            if sh.target_oid >= 0:
                continue

            # nếu đang có hàng thì ưu tiên delivery
            if sh.bag:
                oid = self._choose_delivery_from_bag(sh)
                if oid is not None:
                    sh.target_oid = oid
                    sh.phase = "deliver"
                    sh.path = []
                continue

            # otherwise chọn pickup mới
            order = self._choose_order(sh, [o for o in pending if o.id not in reserved], t)
            if order is not None:
                sh.target_oid = order.id
                sh.phase = "pickup"
                sh.path = []
                reserved.add(order.id)


    # Path Routing
    def _plan_paths(self, shippers: List[Shipper]) -> None:
        for sh in shippers:
            if sh.target_oid < 0:
                continue

            order = self.orders[sh.target_oid]
            if sh.phase == "pickup":
                target = (order.sx, order.sy)
            else:
                target = (order.ex, order.ey)

            if not sh.path:
                sh.path = compute_path(self.grid, (sh.r, sh.c), target)


    # Main simulation loop
    def run(self) -> dict:
        print("GREEDY BFS RUNNING") #  debug
        orders = self.orders
        shippers = self._init_shippers()
        print("NUM SHIPPERS =", len(shippers)) # debug
        for sh in shippers:
            print(sh)

        # group orders by appearance time
        orders_by_t: Dict[int, List[Order]] = {}
        for order in orders.values():
            orders_by_t.setdefault(order.appear_t, []).append(order)

        pending: List[Order] = []
        in_transit: List[Order] = []
        delivered: List[Order] = []
        total_reward = 0.0
        total_movecost = 0.0
        t0 = time.time()

        # Simulation
        for t in range(self.T):
            # observe_state()
            pending.extend(orders_by_t.get(t, []))

            # assign_orders()
            self._assign_orders(shippers, pending, t)

            # plan_paths()
            self._plan_paths(shippers)

            # execute_moves()
            total_movecost += execute_moves(shippers, self.grid, orders)

            # pickup_phase()
            pickup_phase(shippers, pending, in_transit, orders)

            # delivery_phase()
            total_reward += delivery_phase(shippers, orders, in_transit, delivered, t, self.T)

        elapsed = time.time() - t0

        return compute_result(
            method_name="Greedy BFS Simple Baseline",
            cfg=self.cfg,
            orders=orders,
            delivered_orders=delivered,
            shippers=shippers,
            total_reward=total_reward,
            total_movecost=total_movecost,
            elapsed=elapsed,
        )

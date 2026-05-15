from __future__ import annotations
from typing import Dict, List, Tuple
from env import Shipper, Order, DIRS, delivery_reward, move_cost
from solvers.routing_utils import next_position


def init_shippers(cfg: dict, free_cells: List[Tuple[int, int]]) -> List[Shipper]:
    """
    Khởi tạo shipper tại các ô trống đầu tiên
    """
    C = cfg["C"]

    shippers = []
    for i in range(C):
        r, c = free_cells[i]
        shippers.append(
            Shipper(
                id=i,
                r=r,
                c=c,
                W_max=cfg["W_max"][i],
                K_max=cfg["K_max"][i]
            )
        )

    return shippers

def execute_moves(
    shippers: List[Shipper], 
    grid: List[List[int]], 
    orders: Dict[int, Order]
) -> float:
    """
    Di chuyển tất cả shipper 1 bước
    Returns:
        total movement cost
    """
    occupied_next = {}
    total_movecost = 0.0

    for sh in shippers:
        old_pos = (sh.r, sh.c)
        old_weight = sh.w_carried(orders)
        action = "S"

        if sh.path:
            action = sh.path.pop(0)

        nr, nc = next_position(sh.r, sh.c, action)

        n = len(grid)

        valid = (0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0)

        if (action != "S" and valid and (nr, nc) not in occupied_next):
            occupied_next[(nr, nc)] = sh.id
            sh.r = nr
            sh.c = nc
        else:
            occupied_next[(sh.r, sh.c)] = sh.id

            # force replanning later
            sh.path = []

        if (sh.r, sh.c) != old_pos:
            cost = move_cost(old_weight, sh.W_max)
            sh.total_reward += cost
            sh.steps_moved += 1

            total_movecost += cost

    return total_movecost

def pickup_phase(
    shippers: List[Shipper], 
    pending_orders: List[Order],
    in_transit: List[Order], 
    orders: Dict[int, Order]
) -> None:
    """
    Nhặt các đơn ở đúng các vị trí hiện tại
    """
    for sh in shippers:
        here = [
            o for o in pending_orders
            if (o.sx == sh.r and o.sy == sh.c and not o.picked)
        ]

        # priority theo thứ tự giảm dần
        # deadline theo thứ tự tăng dần
        # id theo thứ tự tăng dần
        here.sort(
            key=lambda o: (-o.p, o.et, o.id)
        )

        picked_now = []
        for order in here:
            if not sh.can_pickup(order, orders):
                continue
            order.picked = True
            order.carrier = sh.id
            sh.bag.append(order.id)
            picked_now.append(order)
            in_transit.append(order)

            if sh.target_oid == order.id:
                sh.phase = "deliver"
                sh.path = []

        for order in picked_now:
            pending_orders.remove(order)

def delivery_phase(
    shippers: List[Shipper], 
    orders: Dict[int, Order],
    in_transit: List[Order],
    delivered_orders: List[Order],
    t: int,
    T: int,
) -> float:
    """
    Giao các đơn đã tới destination
    """
    total_reward = 0.0
    for sh in shippers:
        deliver_now = [
            o for o in in_transit 
            if (o.carrier == sh.id and o.ex == sh.r and o.ey == sh.c and not o.delivered)
        ]

        for order in deliver_now:
            order.delivered = True
            order.deliver_t = t
            reward = delivery_reward(order, t, T)
            sh.total_reward += reward
            total_reward += reward
            sh.bag.remove(order.id)
            delivered_orders.append(order)

            if sh.target_oid == order.id:
                sh.target_oid = -1
                sh.phase = "idle"
                sh.path = []

    return total_reward

def update_rewards():
    return

def resolve_basic_collision():
    return

def compute_result(
    method_name: str,
    cfg: dict,
    orders: Dict[int, Order],
    delivered_orders: List[Order],
    shippers: List[Shipper],
    total_reward: float,
    total_movecost: float,
    elapsed: float
) -> dict:
    delivered = len(delivered_orders)
    on_time = sum(1 for o in delivered_orders if o.deliver_t <= o.et)
    late = delivered - on_time
    missed = len(orders) - delivered

    return {
        "method": method_name,
        "config_name": cfg.get("name", ""),
        "total_orders": len(orders),
        "delivered": delivered,
        "on_time": on_time,
        "late": late,
        "missed": missed,
        "delivery_rate": round(delivered / max(len(orders), 1) * 100, 2),
        "on_time_rate": round(on_time / max(len(orders), 1) * 100, 2),
        "total_reward": round(total_reward, 4),
        "total_movecost": round(total_movecost, 4),
        "net_reward": round(total_reward + total_movecost, 4),
        "elapsed_sec": round(elapsed, 2),
        "shipper_rewards": [
            round(sh.total_reward, 4) for sh in shippers
        ]
    }

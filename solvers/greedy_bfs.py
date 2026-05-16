from __future__ import annotations

from shutil import move
import time
from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple

from env import DeliveryEnv, Order, Shipper, is_valid_cell, valid_next_pos
from solvers.solver import Solver


Move = str
Position = Tuple[int, int]
Action = Tuple[Move, object]

INF = 10**9

MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")

MULTI_PICKUP_RADIUS = 4
MULTI_PICKUP_MIN_SCORE = 5.0 # chỉ pickup nếu đơn này không quá lệch route
OPPORTUNISTIC_MIN_SLACK = 5 # chỉ pickup thêm nếu deadline còn dư >= 5 timesteps
MAX_ALLOWED_LATENESS = 5 # chỉ nhận đơn nếu ước lượng thời gian hoàn thành không quá deadline + 5 timesteps


class GreedyBFS(Solver):
    """
    Greedy BFS baseline cho Online MAPD.

    Solver chỉ cài phần policy:
    - chọn đơn cần giao/nhặt;
    - tìm đường bằng BFS trên grid hiện tại.
    """

    method_name = "GreedyBFS"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self._distance_cache: Dict[Tuple[Position, Position], int] = {}
        self._next_move_cache: Dict[Tuple[Position, Position], Move] = {}

    # ------------------------------------------------------------------
    # BFS utilities
    # ------------------------------------------------------------------
    def _neighbors(self, pos: Position) -> Iterable[Tuple[Move, Position]]:
        """Liệt kê các ô kề hợp lệ bằng valid_next_pos() của env."""
        for move in MOVES:
            nxt = valid_next_pos(pos, move, self.grid)
            if nxt != pos:
                yield move, nxt

    def _bfs_parents(
        self,
        start: Position,
        goal: Position,
    ) -> Optional[Dict[Position, Tuple[Optional[Position], Move]]]:
        """Chạy BFS và lưu parent để lấy khoảng cách/next move."""
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return None

        queue: deque[Position] = deque([start])
        parent: Dict[Position, Tuple[Optional[Position], Move]] = {
            start: (None, "S")
        }

        while queue:
            current = queue.popleft()
            if current == goal:
                return parent

            for move, nxt in self._neighbors(current):
                if nxt in parent:
                    continue
                parent[nxt] = (current, move)
                queue.append(nxt)

        return None

    def _distance(self, start: Position, goal: Position) -> int:
        """
        Khoảng cách đường đi ngắn nhất trên grid có vật cản.
        """
        if start == goal:
            return 0

        key = (start, goal)
        if key in self._distance_cache:
            return self._distance_cache[key]

        parent = self._bfs_parents(start, goal)
        if parent is None or goal not in parent:
            self._distance_cache[key] = INF
            return INF

        distance = 0
        current = goal
        while current != start:
            previous, _ = parent[current]
            if previous is None:
                self._distance_cache[key] = INF
                return INF
            current = previous
            distance += 1

        self._distance_cache[key] = distance
        return distance

    def _next_move(self, start: Position, goal: Position) -> Move:
        """Bước đi đầu tiên trên đường BFS từ start tới goal."""
        if start == goal:
            return "S"

        key = (start, goal)
        if key in self._next_move_cache:
            return self._next_move_cache[key]

        parent = self._bfs_parents(start, goal)
        if parent is None or goal not in parent:
            self._next_move_cache[key] = "S"
            return "S"

        current = goal
        while True:
            previous, move = parent[current]
            if previous is None:
                self._next_move_cache[key] = "S"
                return "S"
            if previous == start:
                self._next_move_cache[key] = move
                return move
            current = previous

    # ------------------------------------------------------------------
    # Policy: chọn đơn
    # ------------------------------------------------------------------
    def _select_delivery(self, shipper: Shipper, orders: Dict[int, Order], current_time: int) -> Optional[Order]:
        """
        Nếu shipper đang mang hàng thì chọn đơn đang mang để đi giao.
        Delivery heuristic: ưu tiên đơn có slack nhỏ nhất (thời gian còn lại trước deadline)
        Slack càng nhỏ = đơn sắp trễ, slack âm thì chắc chắn trễ
        --> giảm starvation cho đơn có deadline sớm và tránh chọn đơn có deadline xa trước nếu đang có đơn sắp trễ trong bag.
        """
        carried_orders = [
            orders[oid]
            for oid in shipper.bag
            if oid in orders and not orders[oid].delivered
        ]
        if not carried_orders:
            return None
        
        # ưu tiên minimum slack > gần
        def slack(order):
            """Ước lượng thời gian còn lại trước deadline."""
            d = self._distance(shipper.position, (order.ex, order.ey))
            return order.et - (current_time + d)

        return min(
            carried_orders,
            key=lambda order: (
                slack(order), # urgent hơn
                self._distance(shipper.position, (order.ex, order.ey)), # đích giao gần nhất
                order.et, # deadline sớm hơn
                -order.p, # priority cao hơn
                order.id, # id nhỏ hơn
            ),
        )
    
    def _delivery_urgency(self, shipper: Shipper, order: Order, current_time):
        """
        Độ urgent của delivery hiện tại (cũng là slack). Urgent thấp -> giao ngay
        Dùng để quyết định có nên nhặt thêm đơn opportunistic hay không.
        """
        d = self._distance(shipper.position, (order.ex, order.ey))
        return order.et - (current_time + d)
    
    def _estimate_base_reward(self, order: Order) -> float:
        """
        Ước lượng reward cơ bản của đơn hàng. Reward tăng theo trọng lượng và priority
        Đây không phải reward thực tế vì còn phụ thuộc vào thời gian hoàn thành, nhưng dùng để so sánh giữa các đơn với nhau khi chọn pickup.
        """
        w = order.w
        if w <= 0.2:
            base = 4
        elif w <= 3:
            base = 10
        elif w <= 10:
            base = 15
        elif w <= 30:
            base = 20
        else:
            base = 30

        priority_multiplier = {
            1: 1.0,
            2: 2.0,
            3: 3.0,
        }

        return base * priority_multiplier[order.p]
    
    def _pickup_score(self, shipper: Shipper, order: Order, current_time) -> float:
        """
        Heuristic utility cho pickup candidate.
        Hiện tại: score = estimated_reward - movement_cost - lateness_cost.
        Đây là local greedy utility approximation. Không tối ưu toàn cục, nhưng giúp policy:
        - tránh đơn quá xa
        - tránh đơn gần như chắc chắn trễ
        - ưu tiên đơn có reward cao hơn (nặng hơn, priority cao hơn)
        """
        shipper_pos = shipper.position
        pickup_pos = (order.sx, order.sy)
        delivery_pos = (order.ex, order.ey)

        # khoảng shipper -> pickup
        d_pickup = self._distance(shipper_pos, pickup_pos)
        # khoảng cách pickup -> delivery
        d_delivery = self._distance(pickup_pos, delivery_pos)

        if d_pickup == INF or d_delivery == INF:
            return -INF
        
        # tổng quãng đường cần đi để hoàn thành đơn
        trip_distance = d_pickup + d_delivery
        # ước lượng thời gian hoàn thành đơn 
        eta = current_time + trip_distance
        # ước lượng reward
        reward = self._estimate_base_reward(order)
        # phần trễ deadline (nếu có)
        lateness = max(0, eta - order.et)
        # regularization - xấp xỉ movement cost
        movement_penalty = 0.05 * trip_distance # nên gán đúng move cost trong env, 0.05 là đang tự chọn
        # regularization - deadline penalty, phạt rất mạnh nếu trễ
        lateness_penalty = 2.0 * lateness

        # final utility score
        score = reward - movement_penalty - lateness_penalty

        return score

    def _select_pickup(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        reserved_order_ids: set[int],
        current_time: int
    ) -> Optional[Order]:
        """
        Chọn đơn chưa nhặt có utility tốt và shipper còn khả năng chở.
        Pickup heuristic: utility.
        Pickup policy:
        - feasibility: 
            + chưa nhặt/giao
            + không trùng với đơn đã chọn của shipper khác
            + shipper còn đủ capacity
            + pickup reachable
            + không quá hopeless về deadline.
        - utility ranking: chọn utility cao nhất.
        """

        candidates: List[tuple[float, Order]] = []
        # candidates: List[Order] = []

        for order in orders.values():
            if order.picked or order.delivered:
                continue
            if order.id in reserved_order_ids:
                continue
            if not shipper.can_carry(order, orders):
                continue
            if self._distance(shipper.position, (order.sx, order.sy)) >= INF:
                continue

            d_pickup = self._distance(shipper.position, (order.sx, order.sy))
            d_delivery = self._distance((order.sx, order.sy), (order.ex, order.ey))

            # ETA đơn giản để reject đơn chắc chắn trễ, xem lại
            eta = current_time + d_pickup + d_delivery
            if eta > order.et + MAX_ALLOWED_LATENESS:
                continue

            score = self._pickup_score(shipper, order, current_time)

            candidates.append((score, order))

            # candidates.append(order)

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda order: (
                -order[0], # score cao hơn, tối đa utility
                # self._distance(shipper.position, (order.sx, order.sy)), # pickup gần nhất
                -order[1].p, # priority cao hơn
                order[1].et, # deadline sớm hơn
                order[1].id, # id nhỏ hơn
            ),
        )[1]
    
    def _select_opportunistic_pickup(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        reserved_order_ids: set[int],
        current_time: int,
        primary_delivery: Order,  
    ) -> Optional[Order]:
        """
        Opportunistic pickup: tranh thủ nhặt thêm đơn trong lúc đang đi giao.
        Ý tưởng: chỉ nhặt nếu đơn mới:
        - không lệch route quá nhiều,
        - không làm route tương lai quá xấu,
        - utility đủ tốt.
        Pickup heuristic: hybrid.
        """
        # chỉ nhặt nếu còn chỗ trong bag
        if len(shipper.bag) >= shipper.K_max:
            return None

        delivery_pos = (primary_delivery.ex, primary_delivery.ey)

        candidates = []

        for order in orders.values():
            if order.picked or order.delivered:
                continue
            if order.id in shipper.bag:
                continue
            if order.id in reserved_order_ids:
                continue
            if not shipper.can_carry(order, orders):
                continue
            
            opportunistic_pickup = (order.sx, order.sy)
            opportunistic_delivery = (order.ex, order.ey)

            # future burden estimate: pickup này sẽ còn phải giao về sau, nên ước lượng chi phí giao đơn này từ pickup -> delivery để đánh giá độ phức tạp thêm vào route sau này nếu nhặt nó.
            future_delivery_cost = self._distance(
                opportunistic_pickup,
                opportunistic_delivery
            )
            if future_delivery_cost >= INF:
                continue

            # khoảng cách shipper -> opportunistic pickup, xét xem reachable không
            d_to_pickup = self._distance(shipper.position, opportunistic_pickup)
            if d_to_pickup >= INF:
                continue

            # khoảng cách opportunistic pickup -> delivery chính, xét xem xa quá không
            d_pickup_to_delivery = self._distance(opportunistic_pickup, delivery_pos)
            if d_pickup_to_delivery >= INF:
                continue

            # route trực tiếp shipper -> delivery chính hiện tại
            direct_delivery = self._distance(shipper.position, delivery_pos)
            if direct_delivery >= INF:
                continue

            # insertion detour: xét xem nhặt thêm đơn này có làm lệch route delivery chính quá nhiều không
            detour = max(
                0,
                d_to_pickup + d_pickup_to_delivery - direct_delivery
            ) # triangle inequality slack
            
            # effective detour: cân bằng giữa detour hiện tại và độ phức tạp thêm vào route sau này nếu nhặt đơn này
            # để tránh trường hợp nhặt 1 đơn rất gần nhưng lại làm route giao chính lệch quá nhiều
            # hoặc nhặt 1 đơn có detour chấp nhận được nhưng lại làm route giao chính sau này phức tạp hơn nhiều.
            effective_detour = detour + 0.3 * future_delivery_cost
            
            # nếu route lệch quá nhiều thì không nhặt
            if effective_detour > MULTI_PICKUP_RADIUS:
                continue

            # nếu utility không đủ tốt thì không nhặt
            score = self._pickup_score(shipper, order, current_time)
            if score < MULTI_PICKUP_MIN_SCORE:
                continue

            candidates.append((score, order))

        if not candidates:
            return None
        
        candidates.sort(key=lambda order:(
            -order[0], # score cao hơn, tối đa utility
            # self._distance(shipper.position, (order.sx, order.sy)), # pickup gần nhất
            -order[1].p, # priority cao hơn
            order[1].et, # deadline sớm hơn
            order[1].id, # id nhỏ hơn
        ))

        return candidates[0][1]

    # ------------------------------------------------------------------
    # Policy: tạo action
    # ------------------------------------------------------------------
    def _move_towards(self, shipper: Shipper, goal: Position) -> Tuple[Move, Position]:
        """
        Lấy bước đi kế tiếp và vị trí dự kiến sau bước đó.
        """
        move = self._next_move(shipper.position, goal)
        next_position = valid_next_pos(shipper.position, move, self.grid)
        return move, next_position

    def _delivery_action(self, shipper: Shipper, order: Order) -> Action:
        goal = (order.ex, order.ey)
        move, next_position = self._move_towards(shipper, goal)

        # Với env chuẩn, op=2 nghĩa là giao tất cả đơn trong bag
        # có đích tại ô hiện tại sau khi di chuyển.
        return (move, 2) if next_position == goal else (move, 0)

    def _pickup_action(self, shipper: Shipper, order: Order) -> Action:
        goal = (order.sx, order.sy)
        move, next_position = self._move_towards(shipper, goal)

        # cargo_op = 1: env/Shipper.pickup_best() sẽ nhặt một đơn tốt nhất tại ô hiện tại.
        return (move, 1) if next_position == goal else (move, 0)

    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]

        actions: Dict[int, Action] = {}
        reserved_pickups: set[int] = set()
        current_time = obs["t"]

        for shipper in sorted(shippers, key=lambda s: s.id):
            delivery_order = self._select_delivery(shipper, orders, current_time)
            if delivery_order is not None:
                urgency = self._delivery_urgency(shipper, delivery_order, current_time)

                # chỉ opportunistic pickup nếu còn đủ xa deadline
                if urgency >= OPPORTUNISTIC_MIN_SLACK:
                    opportunistic = self._select_opportunistic_pickup(shipper, orders, reserved_pickups, current_time, delivery_order)
                    if opportunistic is not None:
                        reserved_pickups.add(opportunistic.id)
                        actions[shipper.id] = self._pickup_action(shipper, opportunistic)
                        continue

                actions[shipper.id] = self._delivery_action(shipper, delivery_order)
                continue

            # normal pickup
            pickup_order = self._select_pickup(shipper, orders, reserved_pickups, current_time)
            # pickup_order = self._select_pickup(shipper, orders, reserved_pickups)
            if pickup_order is not None:
                reserved_pickups.add(pickup_order.id)
                actions[shipper.id] = self._pickup_action(shipper, pickup_order)
                continue

            # idle
            actions[shipper.id] = ("S", 0)

        return actions

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> dict:
        """
        Solver nhìn observation hiện tại.
        Với từng shipper:
        - nếu đang mang đơn → chọn đơn để giao.
        - nếu chưa có đơn → chọn đơn để đi pickup.
        Tạo action (move, cargo_op).
        Env thực thi:
        - move
        - pickup
        - deliver
        - sinh order mới
        """
        start_time = time.time()
        obs = self.env.reset()

        while not obs.get("done", False):
            actions = self._decide_actions(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        return self.env.result(
            self.method_name,
            elapsed_sec=time.time() - start_time,
        )
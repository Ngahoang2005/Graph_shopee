from __future__ import annotations

from shutil import move
import time
import math
from collections import deque, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from env import DeliveryEnv, Order, Shipper, is_valid_cell, valid_next_pos, manhattan, r_base
from solvers.solver import Solver


Move = str
Position = Tuple[int, int]
Action = Tuple[Move, object]

INF = 10**9

MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")
ALPHA = {1: 1.0, 2: 2.0, 3: 3.0}
BETA  = {1: 0.1, 2: 0.3, 3: 0.5}
GAMMA = 1.0

MULTI_PICKUP_MIN_SCORE = 10.0 # chỉ pickup nếu đơn này có score >=10
OPPORTUNISTIC_MIN_SLACK = 5 # chỉ pickup thêm nếu deadline còn dư >= 5 timesteps
MAX_ALLOWED_LATENESS = 5 # chỉ nhận đơn nếu ước lượng thời gian hoàn thành không quá deadline + 5 timesteps
DELIVERY_CLUSTER_BONUS = 2 # hệ số nhân reward cho mỗi delivery nearby

AVG_REWARD = 15.0
HOTSPOT_RADIUS = 3.0 # đề cho

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

        self.adaptive_radius = max(2, min(10, self.env.N // 3)) # bán kính để coi là "gần", dùng trong opportunistic pickup và delivery cluster, có thể điều chỉnh dựa trên kích cỡ map
        
        self.hotspot_memory: Dict[Position, float] = defaultdict(float) # lưu nhiệt độ hotspot để ổn định điểm hotspot theo thời gian, tránh nhảy lung tung
        self.hotspot_sigma = max(1.0, self.env.N / 2)
        self.hotspot_decay = max(1.0, self.adaptive_radius) # map càng lớn decay càng chậm, vì travel time dài hơn
        self.last_seen_orders: set[int] = set() # để theo dõi đơn hàng mới xuất hiện, có thể dùng để cập nhật hotspot

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
    def _delivery_urgency(self, shipper: Shipper, order: Order, current_time):
        """
        Độ urgent của delivery hiện tại (cũng là slack). Urgent thấp -> giao ngay
        Dùng để quyết định có nên nhặt thêm đơn opportunistic hay không.
        """
        d = self._distance(shipper.position, (order.ex, order.ey))
        return order.et - (current_time + d)

    def _delivery_cluster_bonus(self, target_order: Order, carried_orders: List[Order]) -> float:
        """
        Tính bonus reward nếu destination của target_order nằm gần nhiều destination khác trong bag (cùng cluster)
        Ý tưởng: nếu có nhiều đơn giao gần nhau thì sẽ tiết kiệm được nhiều thời gian di chuyển -> tăng net reward
        Heuristic: local cluster density.
        """
        tx, ty = target_order.ex, target_order.ey
        nearby = 0

        for other in carried_orders:
            if other.id == target_order.id:
                continue
            
            d = self._distance((tx, ty), (other.ex, other.ey))

            if d <= self.adaptive_radius:
                nearby += 1

        return nearby * DELIVERY_CLUSTER_BONUS
    
    def _select_delivery(self, shipper: Shipper, orders: Dict[int, Order], current_time: int) -> Optional[Order]:
        """
        Nếu shipper đang mang hàng thì chọn đơn đang mang để đi giao.
        Delivery heuristic: ưu tiên đơn có slack nhỏ nhất (thời gian còn lại trước deadline)
        Slack càng nhỏ = đơn sắp trễ, slack âm thì chắc chắn trễ
        --> giảm starvation cho đơn có deadline sớm và tránh chọn đơn có deadline xa trước nếu đang có đơn sắp trễ trong bag.

        Cải tiến bước tiếp theo (delivery clustering): Delivery heuristic: 
        - slack nhỏ nhất
        - trong số các đơn có slack gần nhau thì ưu tiên đơn có nhiều đơn giao gần đó trong bag (cùng cluster) để tiết kiệm thời gian di chuyển.
        """
        carried_orders = [
            orders[oid]
            for oid in shipper.bag
            if oid in orders and not orders[oid].delivered
        ]
        if not carried_orders:
            return None
        
        candidates = []
        
        for order in carried_orders:
            d = self._distance(shipper.position, (order.ex, order.ey))
            if d >= INF: 
                continue

            # ưu tiên minimum slack > gần
            slack = order.et - (current_time + d)

            # delivery clustering bonus
            cluster_bonus = self._delivery_cluster_bonus(order, carried_orders)

            # heuristic deliver score: kết hợp giữa slack, khoảng cách đến destination, priority và cluster bonus
            delivery_score = (
                -2.0 * slack # urgent hơn
                - 1.0 * d # đích giao gần hơn thì tốt hơn
                + cluster_bonus # nhiều đơn giao gần đó trong bag thì tốt hơn
                + 0.5 * order.p # priority cao hơn thì tốt hơn, bonus nhẹ
            )

            candidates.append((delivery_score, order))

        if not candidates:
            return None
        
        candidates.sort(
            key=lambda x: (
                -x[0],      # maximize score
                x[1].et,    # deadline sớm hơn
                x[1].id,    # id nhỏ hơn
            )
        )

        return candidates[0][1]
    
    def _estimate_order_score(self, sh: Shipper, order: Order, orders: Dict[int, Order], current_time: int, total_time: int) -> float:
        """
        Ước lượng score của đơn hàng để so sánh giữa các đơn khi chọn pickup, tính theo đúng công thức đề bài.
        Score cao hơn nghĩa là đơn hàng tốt hơn để pickup.
        Hiện tại: score = estimated_reward - movement_cost.
        Đây là local greedy utility approximation. Không tối ưu toàn cục, nhưng giúp policy:
        - tránh đơn quá xa
        - tránh đơn gần như chắc chắn trễ
        - ưu tiên đơn có reward cao hơn (nặng hơn, priority cao hơn)
        """
        # ước tính khoảng cách
        d1 = manhattan(sh.r, sh.c, order.sx, order.sy)
        d2 = manhattan(order.sx, order.sy, order.ex, order.ey)

        travel_time = d1 + d2
        est_delivery_time = current_time + travel_time

        # ước tính reward cơ bản
        rb = r_base(order.w)
        if est_delivery_time <= order.et:
            bonus = max(
                0.0,
                (order.et - est_delivery_time) / max(order.et, 1)
            )
            reward = ALPHA[order.p] * rb * (1.0 + bonus)

        else:
            factor = max(
                0.0,
                1.0 - (est_delivery_time - order.et) / max(total_time, 1)
            )
            reward = BETA[order.p] * rb * factor

        # movement cost estimate
        carried = sum(orders[oid].w for oid in sh.bag if oid in orders)

        move_penalty = (
            travel_time * 0.01 
            * (1.0 + GAMMA * carried / max(sh.W_max, 1.0))
        )

        return reward - move_penalty

    def _select_pickup(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        reserved_order_ids: set[int],
        current_time: int,
        total_time: int,
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
            if eta > order.et + self.adaptive_radius:
                continue

            score = self._estimate_order_score(shipper, order, orders, current_time, total_time)

            candidates.append((score, order))

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
        total_time: int,
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
            if effective_detour > self.adaptive_radius:
                continue

            # nếu score không đủ tốt thì không nhặt
            score = self._estimate_order_score(shipper, order, orders, current_time, total_time)
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
    # Ý tưởng: xử lý hotspot
    # ------------------------------------------------------------------
    def _update_hotspots(
        self,
        orders: Dict[int, Order]
    ) -> None:
        """
        Online hotspot estimation.
        Hotspot = vùng có mật độ pickup cao gần đây.
        Mỗi order mới xuất hiện sẽ tăng heat quanh pickup region.
        Heat decay dần theo thời gian để thích nghi với surge động.
        """
        for pos in list(self.hotspot_memory.keys()):
            self.hotspot_memory[pos] *= 0.8 # số chọn bừa, đây là heat giảm dần theo thời gian

            if self.hotspot_memory[pos] < 0.01:
                del self.hotspot_memory[pos] # heat nhỏ quá thì coi như đây kcon là hotspot nữa

        for order in orders.values():
            if order.picked or order.delivered:
                continue

            center = (order.sx, order.sy)

            self.hotspot_memory[center] += 1.0 # cách nghĩ bên dưới phức tạp quá, tăng heat tại pickup center là đủ

            # for dr in range(-self.adaptive_radius, self.adaptive_radius +1):
            #     for dc in range(-self.adaptive_radius, self.adaptive_radius +1):
            #         # để tránh update Gaussian influence lên toàn bộ map, chỉ update trong khoảng cách ảnh hưởng mà đề cho là manhattan <= 3
            #         r = center[0] + dr
            #         c = center[1] + dc
            #         if not is_valid_cell((r, c), self.grid):
            #             continue
            #         d = abs(dr) + abs(dc)
            #         if d > self.adaptive_radius:
            #             continue
            #         influence = math.exp(
            #             -(d ** 2) / (2 * self.hotspot_sigma ** 2)
            #         )
            #         self.hotspot_memory[(r, c)] = (
            #             self.hotspot_memory.get((r, c), 0.0)
            #             + influence
            #         )

    def _select_hotspot(self, shipper: Shipper, orders: Dict[int, Order]) -> Optional[Position]:
        """
        Chọn hotspot có expected future net reward tốt nhất.
        Ý tưởng:
        - hotspot value = expected future reward density
        - penalty theo movement cost để tránh chase hotspot quá xa
        """ 
        if not self.hotspot_memory:
            return None
        
        best_score = -INF
        best_pos = None

        carried_weight = sum(
            orders[oid].w
            for oid in shipper.bag
            if oid in orders
        )

        for pos, hotspot_value in self.hotspot_memory.items():
            d = self._distance(shipper.position, pos)
            if d >= INF:
                continue

            # expected future reward
            expected_reward = hotspot_value * AVG_REWARD # vì đơn tương lai chưa được sinh nên chưa rõ reward của đơn

            # estimated movement cost
            move_penalty = (
                d * 0.01
                * (1.0 + GAMMA * carried_weight / max(shipper.W_max, 1.0))
            )

            score = expected_reward - move_penalty
            if score > best_score:
                best_score = score
                best_pos = pos

        return best_pos 

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
        total_time = obs["T"]

        self._update_hotspots(orders) # cập nhật hotspot trước khi quyết định action

        for shipper in sorted(shippers, key=lambda s: s.id):
            delivery_order = self._select_delivery(shipper, orders, current_time)
            if delivery_order is not None:
                urgency = self._delivery_urgency(shipper, delivery_order, current_time)

                # chỉ opportunistic pickup nếu còn đủ xa deadline
                if urgency >= OPPORTUNISTIC_MIN_SLACK:
                    opportunistic = self._select_opportunistic_pickup(shipper, orders, reserved_pickups, current_time, total_time, delivery_order)
                    if opportunistic is not None:
                        reserved_pickups.add(opportunistic.id)
                        actions[shipper.id] = self._pickup_action(shipper, opportunistic)
                        continue

                actions[shipper.id] = self._delivery_action(shipper, delivery_order)
                continue

            # normal pickup
            pickup_order = self._select_pickup(shipper, orders, reserved_pickups, current_time, total_time)
            # pickup_order = self._select_pickup(shipper, orders, reserved_pickups)
            if pickup_order is not None:
                reserved_pickups.add(pickup_order.id)
                actions[shipper.id] = self._pickup_action(shipper, pickup_order)
                continue

            # idle
            # actions[shipper.id] = ("S", 0)

            # hotspot-aware idle repositioning:
            hotspot = self._select_hotspot(shipper, orders)
            if hotspot is not None:
                move, _ = self._move_towards(shipper, hotspot)
                actions[shipper.id] = (move, 0)
            else:
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
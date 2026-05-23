from __future__ import annotations

import time
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple
import heapq

from env import DeliveryEnv, Order, Shipper, is_valid_cell, valid_next_pos, r_base
from solvers.solver import Solver

Move = str
Position = Tuple[int, int]
SpaceTimeState = Tuple[int, int, int] # (r, c, t), toạ độ tại thời điểm t
Action = Tuple[Move, object]

INF = 10**9
MOVES: Tuple[Move, ...] = ("U", "D", "L", "R", "S")
ALPHA = {1: 1.0, 2: 2.0, 3: 3.0}
BETA  = {1: 0.1, 2: 0.3, 3: 0.5}
GAMMA = 1.0

class Conflict:
    def __init__(
        self,
        kind: str,
        a1: int,
        a2: int,
        time: int,
        cell: Optional[Position] = None,
        edge: Optional[Tuple[Position, Position]] = None,
    ):
        # loại conflict: "vertex" hoặc "edge"
        self.kind = kind
        # agent ids
        self.a1 = a1
        self.a2 = a2
        # timestep xảy ra conflict
        self.time = time
        # vertex conflict: 2 shipper đứng cùng vị trí tại timestep t
        self.cell = cell
        # edge conflict: a1: A -> B, a2: B -> A
        self.edge = edge

class Constraint:
    """
    CBS constraint.
    Vertex:
        agent cannot be at cell at time t.
    Edge:
        agent cannot traverse
        (from_cell -> to_cell)
        at time t.
    """

    def __init__(
        self,
        agent: int,
        time: int,
        cell: Optional[Position] = None,
        edge: Optional[Tuple[Position, Position]] = None,
    ):
        # agent bị áp constraint
        self.agent = agent
        # timestep constraint có hiệu lực
        self.time = time
        # vertex constraint
        self.cell = cell
        # edge constraint
        self.edge = edge

    # để có thể dùng trong set/dict giống frozen dataclass
    def __hash__(self):
        return hash((
            self.agent,
            self.time,
            self.cell,
            self.edge
        ))

    def __eq__(self, other):
        if not isinstance(other, Constraint):
            return False
        return (
            self.agent == other.agent
            and self.time == other.time
            and self.cell == other.cell
            and self.edge == other.edge
        )

class CBSNode:
    def __init__(
        self,
        constraints: List[Constraint],
        paths: Dict[int, List[Position]],
        cost: int,
    ):
        # tập constraints của node hiện tại
        self.constraints = constraints
        # solution = tập paths hiện tại của từng agent
        self.paths = paths
        # objective value - thường là SOC = sum of costs của tất cả các paths trong solution
        self.cost = cost

class PlannedPath:
    def __init__(
        self,
        shipper_id: int,
        path: List[Position],
        target: Optional[Position]
    ):
        self.shipper_id = shipper_id
        self.path = path
        self.target = target

class MAPDCBSSolver(Solver):
    """
    Lightweight Online MAPD-CBS (baseline)
    Ý tưởng:
    - Online MAPD
    - Task allocation: greedy đơn giản
    - Path planning: space-time BFS
    - Xử lý conflict: CBS branching

    Bản chất: rolling-horizon MAPD-CBS phù hợp với env online realtime.
    (vì đơn chưa biết trước, được spawn theo thời gian,... vì vậy không thể plan toàn bộ T timestep)
    """

    method_name = "MAPDCBSSolver"

    def __init__(self, env: DeliveryEnv):
        if not hasattr(env, "cfg"):
            env.cfg = {"N": env.N, "T": env.T, "C": env.C, "G": env.G}

        super().__init__(env)
        self.N = env.N
        self.T = env.T
        self.C = env.C
        self.grid = env.grid
        self.t = env.t

        self.max_plan_depth = max(10, self.N) # planning horizon, planner chỉ nhìn max_plan_depth timestep tương lai
        self._distance_cache: Dict[Tuple[Position, Position], int] = {}
        self._cbs_counter = 0

        # ------------------------------------------------------------------
        # Persistent execution state
        # ------------------------------------------------------------------
        # current committed target của mỗi shipper
        self.current_targets: Dict[int, Position] = {}
        # loại target: "pickup" hoặc "delivery"
        self.current_target_kind: Dict[int, str] = {}
        # persistent planned path
        self.current_paths: Dict[int, PlannedPath] = {}
        # order hiện tại đang phục vụ
        self.current_orders: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # BFS và grid utilities
    # ------------------------------------------------------------------
    def _neighbors(self, pos: Position) -> Iterable[Tuple[Move, Position]]:
        for move in MOVES:
            nxt = valid_next_pos(pos, move, self.grid)
            if nxt != pos:
                yield move, nxt

    def _bfs_distance(self, start: Position, goal: Position) -> int:
        if start == goal:
            return 0
        
        key = (start, goal)
        if key in self._distance_cache:
            return self._distance_cache[key]

        q = deque([(start, 0)])
        visited = {start}
        while q:
            pos, d = q.popleft()
            if pos == goal:
                self._distance_cache[key] = d
                return d

            for _, nxt in self._neighbors(pos):
                if nxt in visited:
                    continue
                visited.add(nxt)
                q.append((nxt, d + 1))

        self._distance_cache[key] = INF
        return INF

    # ------------------------------------------------------------------
    # Policy: chọn đơn, greedy task allocation
    # ------------------------------------------------------------------
    def _estimate_order_score(self, sh: Shipper, order: Order, orders: Dict[int, Order], current_time: int, total_time: int) -> float:
        """
        Ước lượng score của đơn hàng để so sánh giữa các đơn khi chọn pickup, tính theo đúng công thức đề bài.
        Score cao hơn nghĩa là đơn hàng tốt hơn để pickup.
        Hiện tại: score = estimated_reward - movement_cost.
        """
        # ước tính khoảng cách
        d1 = self._bfs_distance(sh.position, (order.sx, order.sy))
        d2 = self._bfs_distance((order.sx, order.sy), (order.ex, order.ey))
        if d1 >= INF or d2 >= INF:
            return -INF

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

        future_carried = carried + order.w

        move_penalty = (
            travel_time * 0.01 
            * (1.0 + GAMMA * future_carried / max(sh.W_max, 1.0))
        )

        return reward - move_penalty
    
    def _select_delivery(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        current_time: int,
    ) -> Optional[Order]:
        carried = [orders[oid] for oid in shipper.bag
                   if oid in orders and not orders[oid].delivered]
        if not carried: 
            return None

        best = None
        best_key = None

        for order in carried:
            d = self._bfs_distance(shipper.position, (order.ex, order.ey))
            slack = order.et - (current_time + d)

            # feasibility: slack -> khoảng cách -> priority -> id nhỏ
            key = (slack, d, -order.p, order.id)
            if best is None or key < best_key:
                best = order
                best_key = key

        return best
    
    def _select_pickup(
        self, 
        shipper: Shipper,
        orders: Dict[int, Order],
        reserved_orders: Set[int],
        current_time: int,
        total_time: int,
    ) -> Optional[Order]:
        best = None
        best_score = -INF
        for order in orders.values():
            if order.picked or order.delivered:
                continue
            if order.id in reserved_orders:
                continue
            if not shipper.can_carry(order, orders):
                continue

            # khoảng cách vị trí hiện tại -> địa điểm nhặt đơn
            d1 = self._bfs_distance(
                shipper.position, (order.sx, order.sy)
            )

            # khoảng cách địa điểm nhặt đơn -> địa điểm giao đơn
            d2 = self._bfs_distance(
                (order.sx, order.sy), (order.ex, order.ey)
            )

            if d1 >= INF or d2 >= INF:
                continue

            # nhặt đơn theo cơ hội net reward cao nhất
            score = self._estimate_order_score(shipper, order, orders, current_time, total_time)
            if score > best_score:
                best_score = score
                best = order

        return best
    
    def _assign_task(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        reserved_orders: Set[int],
        current_time: int,
        total_time: int
    ) -> bool:
        """Chỉ assign task mới khi agent/shipper đang free."""
        sid = shipper.id

        # ưu tiên assign delivery trước
        delivery = self._select_delivery(shipper, orders, current_time)
        if delivery is not None:
            self.current_targets[sid] = (delivery.ex, delivery.ey)
            self.current_target_kind[sid] = "delivery"
            self.current_orders[sid] = delivery.id
            return True
        
        # assign pickup
        pickup = self._select_pickup(shipper, orders, reserved_orders, current_time, total_time)
        if pickup is not None:
            reserved_orders.add(pickup.id)
            self.current_targets[sid] = (pickup.sx, pickup.sy)
            self.current_target_kind[sid] = "pickup"
            self.current_orders[sid] = pickup.id
            return True
        
        return False
    
    # ------------------------------------------------------------------
    # Xử lý conflict
    # ------------------------------------------------------------------
    def _violates_constraint(
        self,
        agent: int,
        current: Position,
        nxt: Position,
        t: int,
        constraints: List[Constraint]
    ) -> bool:
        """
        CBS high-level search sinh ra các constraints.
        Low-level planner (space-time BFS) phải tìm path thỏa toàn bộ constraints của riêng agent đó.
        Hàm này KHÔNG kiểm tra collision trực tiếp với agent khác (collision được xử lý bởi CBS high-level branching)
        """
        for c in constraints:
            if c.agent != agent:
                continue
            # constraint đỉnh: agents không được đứng chung 1 vị trí tại timestep t
            if c.cell is not None and c.time == t and c.cell == nxt:
                return True
            # constraint cạnh: agents không được cùng đi qua 1 cạnh tại timestep t
            if c.edge is not None and c.time == t and c.edge == (current, nxt):
                return True
            
        return False

    def _detect_conflict(self, paths: Dict[int, List[Position]]) -> Optional[Conflict]:
        """
        High-level conflict detector.
        Sau khi tất cả agents được plan độc lập, CBS sẽ kiểm tra conflicts giữa các paths:
        1. Vertex conflict: 2 agents cùng ở 1 ô trong cùng timestep
        2. Edge conflict: 2 agents hoán đổi vị trí cho nhau trong cùng timestep
        Nếu phát hiện conflict:   
        -> CBS branching: tạo 2 child nodes với constraints khác nhau.
        """
        agents = list(paths.keys())
        max_len = max(len(p) for p in paths.values())

        for t in range(max_len):
            # kiểm tra sự tồn tại của vertex conflicts
            occupied = {}
            for a in agents:
                path = paths[a] # lấy path của agent a
                if t >= len(path):
                    continue
                pos = path[t] # lấy vị trí trong path tại timestep t

                if pos in occupied: # nếu vị trí này đã occupied thì thêm conflict đỉnh
                    return Conflict(
                        kind="vertex",
                        a1=occupied[pos],
                        a2=a,
                        time=t,
                        cell=pos
                    )
                occupied[pos] = a # nếu vị trí này chưa occupied thì bây giờ occupied

            # kiểm tra sự tồn tại edge conflicts
            for i in range(len(agents)):
                for j in range(i+1, len(agents)):
                    a1 = agents[i]
                    a2 = agents[j]
                    p1 = paths[a1]
                    p2 = paths[a2]
                    prev1 = (p1[t-1] if t-1 < len(p1) and t > 0 else p1[0])
                    prev2 = (p2[t-1] if t-1 < len(p2) and t > 0 else p2[0])

                    if t >= len(p1) or t >= len(p2):
                        continue
                    curr1 = p1[t]
                    curr2 = p2[t]

                    if prev1==curr2 and prev2 == curr1:
                        return Conflict(
                            kind="edge",
                            a1=a1,
                            a2=a2,
                            time=t,
                            edge=((prev1), (curr1))
                        )
        return None
    
    # ------------------------------------------------------------------
    # Thuật toán path planning ở low-level: Space-time BFS
    # ------------------------------------------------------------------
    def _space_time_bfs(
        self,
        agent_id: int,
        start: Position,
        goal: Position,
        constraints: List[Constraint]
    ) -> List[Position]:
        """
        CBS low-level planner.
        BFS phải tìm path:
        - tránh vật cản
        - tránh vi phạm constraints
        Wait action (S) cực kỳ quan trọng: nhiều conflicts có thể resolve chỉ bằng việc đợi vài timestep.
        Path được padding tới planning horizon giúp detect future conflicts chính xác hơn.
        """
        start_state: SpaceTimeState = (start[0], start[1], 0)
        q = deque([start_state])

        parent: Dict[
            SpaceTimeState, 
            Optional[SpaceTimeState]
        ] = {start_state: None}
        found = None

        while q:
            r, c, t, = q.popleft()
            current_pos = (r, c)
            if current_pos == goal and t>0:
                found = (r, c, t)
                break
            if t >= self.max_plan_depth:
                continue

            candidates = list(self._neighbors(current_pos))
            # WAIT action
            candidates.append(("S", current_pos))

            for _, nxt in candidates:
                nr, nc = nxt
                nt = t + 1
                # chỉ tuân theo constraints của node hiện tại
                if self._violates_constraint(agent_id, current_pos, nxt, nt, constraints):
                    continue

                state = (nr, nc, nt)
                if state in parent:
                    continue

                parent[state] = (r, c, t)
                q.append(state)

        if found is None:
            # không tìm được path conflict-free thì đứng yên
            return [start]
        
        path = []
        cur = found
        while cur is not None:
            rr, cc, _ = cur
            path.append((rr, cc))
            cur = parent[cur]
        
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Path planning - CBS
    # ------------------------------------------------------------------
    def _plan_paths(
        self,
        shippers: List[Shipper],
        targets: Dict[int, Position],
        existing_paths: Optional[Dict[int, List[Position]]] = None,
        replanning_agents: Optional[Set[int]] = None
    ) -> Dict[int, PlannedPath]:
        """
        Plan đường đi cho tất cả shipper:
        1. Detect conflict
        2. Nếu conflict-free: return solution
        3. Nếu phát hiện conflict:
        - Branching thành 2 child node
        - Mỗi child thêm constraints cho 1 agent
        4. Replan chỉ cho agent bị ảnh hưởng
        Planner này chỉ dùng cho timestep hiện tại vì đang làm rolling-horizon.
        """
        # khởi tạo, root node không có constraints, agents plan độc lập
        root_constraints = []
        root_paths = {}
        total_cost = 0
        max_cbs_nodes = 100 # giới hạn mở rộng node cho conflict tree
        expanded = 0

        for shipper in shippers:
            sid = shipper.id
            # giữ nguyên path cũ nếu không cần replan
            if (
                existing_paths is not None
                and replanning_agents is not None
                and sid not in replanning_agents
                and sid in existing_paths
            ):
                path = existing_paths[sid]
            else:  
                target = targets.get(shipper.id) # lấy các địa điểm giao đơn
                if target is None:
                    path = [shipper.position] # đứng yên
                else:
                    path = self._space_time_bfs(
                        shipper.id,
                        shipper.position,
                        target,
                        root_constraints
                    )
            
            root_paths[shipper.id] = path
            total_cost += (len(path) - 1)

        root = CBSNode(
            constraints=root_constraints,
            paths=root_paths,
            cost=total_cost
        )

        # high-level CBS search
        open_list = []
        # độ phức tạp heappush: O(log n), heappop: O(log n)
        heapq.heappush(open_list, (root.cost, self._cbs_counter, root))
        self._cbs_counter += 1
        while open_list:
            expanded += 1
            if expanded > max_cbs_nodes:
                break
            # best-first search, 2 dòng dưới đang có độ phức tạp O(nlogn), có thể cải tiến = dùng heap
            _, _, node = heapq.heappop(open_list)
            conflict = self._detect_conflict(node.paths)

            # nếu tìm được conflict-free solution
            if conflict is None:
                result = {}
                for shipper in shippers:
                    result[shipper.id] = PlannedPath(
                        shipper_id=shipper.id,
                        path=node.paths[shipper.id],
                        target=targets.get(shipper.id)
                    )
                return result
            
            # CBS branching
            for constrained_agent in [conflict.a1, conflict.a2]: # binary branching: CBS chia thành 2 child node
                child_constraints = list(node.constraints)

                # thêm constraint
                if conflict.kind  == "vertex": # conflict đỉnh
                    child_constraints.append(
                        Constraint(
                            agent=constrained_agent,
                            time=conflict.time,
                            cell=conflict.cell
                        )
                    )
                else: # conflict cạnh
                    u, v = conflict.edge
                    # đảo đầu mút để đúng cạnh
                    if constrained_agent == conflict.a1:
                        edge = (u, v)
                    else:
                        edge = (v, u)

                    child_constraints.append(
                        Constraint(
                            agent=constrained_agent,
                            time=conflict.time,
                            edge=edge
                        )
                    )

                # các agent còn lại không bị ảnh hưởng thì giữ nguyên path
                child_paths = dict(node.paths)

                # replan chỉ cho các agent bị ảnh hưởng
                shipper = next(s for s in shippers if s.id == constrained_agent)

                target = targets.get(shipper.id)
                if target is None:
                    new_path = [shipper.position]
                else:
                    new_path = self._space_time_bfs(
                        shipper.id, shipper.position, target, child_constraints
                    )
                if new_path is None:
                    continue
                
                child_paths[shipper.id] = new_path
                child_cost = sum(len(p) for p in child_paths.values())

                child = CBSNode(
                    constraints=child_constraints,
                    paths=child_paths,
                    cost=child_cost
                )

                heapq.heappush(open_list, (child.cost, self._cbs_counter, child))
                self._cbs_counter += 1

        # fallback
        return existing_paths
    
    def _path_to_move(self, current: Position, nxt: Position) -> Move:
        """Chuyển đổi path thành move từng bước."""
        r1, c1 = current
        r2, c2 = nxt
        if r2 == r1 - 1:
            return "U"
        if r2 == r1 + 1:
            return "D"
        if c2 == c1 - 1:
            return "L"
        if c2 == c1 + 1:
            return "R"
        return "S"
    
    def _advance_path(
        self,
        shipper_id: int,
        actual_position: Position
    ):
        """
        Sau khi agent thực hiện 1 move, remove bước đầu tiên khỏi persistent path.
        Đồng bộ persistent path với vị trí thực tế.
        """
        if shipper_id not in self.current_paths:
            return

        plan = self.current_paths[shipper_id]
        while len(plan.path) >= 2 and plan.path[0] != actual_position:
            plan.path.pop(0)
    
    def _needs_new_task(self, shipper: Shipper, orders: Dict[int, Order]) -> bool:
        """
        Agent cần task mới nếu:
        - chưa có target location
        - chưa có path/path đã đi hết
        - order hiện tại không còn hợp lệ.
        """
        sid = shipper.id
        # trường hợp chưa có path
        if sid not in self.current_paths:
            return True
        # trường hợp chưa có target location
        if sid not in self.current_targets:
            return True
        # trường hợp đã đi hết path:
        plan = self.current_paths[sid]
        if len(plan.path) <= 1:
            return True
        # trường hợp order hiện tại không còn tồn tại
        oid = self.current_orders.get(sid)
        if oid is None:
            return True
        if oid not in orders:
            return True
        
        order = orders[oid]
        # nếu order này đã được hoàn thành
        if order.delivered:
            return True
        
        # nếu pickup target nhưng order đã bị shipper khác nhặt
        kind = self.current_target_kind.get(sid)
        if kind == "pickup" and order.picked:
            return True
        
        return False
    
    def _needs_replan(self, shipper: Shipper, orders: Dict[int, Order]) -> bool:
        """CBS-style replanning conditions."""
        sid = shipper.id

        # nếu shipper chưa có path
        if sid not in self.current_paths:
            return True
        plan = self.current_paths[sid]

        # nếu đã đi hết path của mình
        if len(plan.path) <= 1:
            return True
        
        # nếu target mất
        if sid not in self.current_targets:
            return True
        
        oid = self.current_orders.get(sid)
        if oid is None:
            return True
        if oid not in orders:
            return True
        
        order = orders[oid]
        # nếu order đã hoàn thành
        if order.delivered:
            return True
        # nếu target kind hiện tại là pickup nhưng thực chất đơn đã được pickup rồi
        kind = self.current_target_kind[sid]
        if kind == "pickup" and order.picked:
            return True
        
        return False
    
    def _replan_agents(
        self,
        shippers: List[Shipper],
        replanning_agents: List[Shipper],
    ):
        """
        Replan chỉ cho affected agents.
        Các agent khác giữ nguyên persistent paths.
        """
        # giữ nguyên path cũ
        existing_paths = {}
        for sid, plan in self.current_paths.items():
            existing_paths[sid] = plan.path

        # targets cần replan
        targets = {}
        for shipper in replanning_agents:
            sid = shipper.id
            if sid not in self.current_targets:
                continue

            targets[sid] = self.current_targets[sid]

        # CBS replanning
        replanned = self._plan_paths(
            shippers,
            targets,
            existing_paths=existing_paths,
            replanning_agents={
                s.id for s in replanning_agents
            }
        )

        # update persistent paths
        for sid, plan in replanned.items():
            self.current_paths[sid] = plan

    # ------------------------------------------------------------------
    # Policy: tạo action
    # ------------------------------------------------------------------
    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]

        actions: Dict[int, Action] = {}
        reserved_orders: Set[int] = set()
        targets: Dict[int, Position] = {}
        target_kind: Dict[int, str] = {}

        current_time = obs["t"]
        total_time = obs["T"]

        # --------------------------------------------------------
        # Phase 1: assign task mới chỉ cho agents đang free
        # --------------------------------------------------------
        replanning_agents = []
        for shipper in sorted(shippers, key=lambda s: s.id):
            if self._needs_new_task(shipper, orders):
                assigned = self._assign_task(
                    shipper,
                    orders,
                    reserved_orders,
                    current_time,
                    total_time,
                )

                if assigned:
                    replanning_agents.append(shipper)

        # --------------------------------------------------------
        # Phase 2: MAPD-CBS path planning, chỉ thực hiện replan cho agents bị ảnh hưởng
        # --------------------------------------------------------
        if replanning_agents:
            self._replan_agents(shippers, replanning_agents)

        # --------------------------------------------------------
        # Phase 3: execute persistent paths
        # --------------------------------------------------------
        for shipper in shippers:
            sid = shipper.id
            # default idle action
            move = "S"
            cargo_op = 0
            if sid in self.current_paths:
                plan = self.current_paths[sid]

                if len(plan.path) >= 2:
                    current_pos = plan.path[0]
                    next_pos = plan.path[1]
                    move = self._path_to_move(current_pos, next_pos)

            # nhặt/giao
            if sid in self.current_targets:
                target = self.current_targets[sid]
                kind = self.current_target_kind[sid]

                # predicted position sau khi move
                predicted_pos = valid_next_pos(shipper.position, move, self.grid)
                # nhặt
                if kind=="pickup" and predicted_pos == target:
                    cargo_op = 1 # cập nhật trạng thái đơn
                # giao
                if kind=="delivery" and predicted_pos == target:
                    cargo_op = 2 # cập nhật trạng thái đơn
        
            # final action
            actions[sid] = (move, cargo_op)

        return actions

    def run(self) -> dict:
        # python run_test.py --config test_config.txt --out results/ --method MAPDCBSSolver

        start_time = time.time()
        obs = self.env.reset()

        while not obs.get("done", False):
            # lưu vị trí trước step
            old_positions = {s.id: s.position for s in obs["shippers"]}
            # decide actions
            actions = self._decide_actions(obs)
            # env step
            obs, _, done, _ = self.env.step(actions)
            # chỉ consume persistent paths nếu thành công
            for shipper in obs["shippers"]:
                sid = shipper.id
                old_pos = old_positions[sid]
                new_pos = shipper.position

                # agent thực sự đã di chuyển
                if old_pos != new_pos:
                    self._advance_path(sid, new_pos)

            if done:
                break

        return self.env.result(
            self.method_name,
            elapsed_sec=time.time() - start_time,
        )
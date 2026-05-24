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

        self.max_plan_depth = max(20, 2 * self.N) # planning horizon, planner chỉ nhìn max_plan_depth timestep tương lai
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
    
    def _heuristic(self, pos: Position, goal: Position) -> int:
        """Manhattan heuristic."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

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

    def _detect_all_conflict(self, paths: Dict[int, List[Position]], window: int = 12) -> List[Conflict]:
        """
        High-level conflict detector.
        Sau khi tất cả agents được plan path tương lai giả lập 1 cách độc lập, CBS sẽ kiểm tra conflicts trong tương lai giữa các paths:
        1. Vertex conflict: 2 agents cùng ở 1 ô trong cùng timestep
        2. Edge conflict: 2 agents hoán đổi vị trí cho nhau trong cùng timestep
        Nếu phát hiện conflict:   
        -> CBS branching: tạo 2 child nodes với constraints khác nhau.

        --> ver 2 cải tiến: tạo danh sách tất cả conflict đang tồn tại 
        thay vì trả về ngay conflict đầu tiên tìm được.
        -> Phục vụ local conflict component detection.
        """
        # danh sách chứa tất cả conflict
        conflicts = []
        agents = list(paths.keys())
        if not agents:
            return conflicts
        
        max_future_timestep = max(window, max(len(p) for p in paths.values()))

        for t in range(max_future_timestep): # scan từng timestep
            # kiểm tra sự tồn tại của vertex conflicts
            occupied = {}
            for a in agents:
                path = paths[a] # lấy path của agent a
                pos = path[t] if t < len(path) else path[-1] # lấy vị trí trong path tại timestep t

                if pos in occupied: # nếu vị trí này đã occupied thì thêm conflict đỉnh
                    conflicts.append(
                        Conflict(
                            kind="vertex",
                            a1=occupied[pos],
                            a2=a,
                            time=t,
                            cell=pos
                        )
                    )
                else:
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
                        conflicts.append(
                            Conflict(
                                kind="edge",
                                a1=a1,
                                a2=a2,
                                time=t,
                                edge=((prev1), (curr1))
                            )
                        )
        return conflicts
    
    def _detect_incremental_conflict(
        self,
        paths: Dict[int, List[Position]],
        dirty_agents: Set[int],
        window: int = 12
    ) -> List[Conflict]:
        """
        Incremental conflict detection: Chỉ detect conflicts liên quan dirty agents.
        Ý tưởng:
        - Agents khác giữ nguyên path
        - Chỉ recompute conflicts của agents vừa replan

        Complexity:
            O(W * d * k)
        với:
            d = số dirty agents
            k = tổng agents
        """

        conflicts = []
        all_agents = list(paths.keys())

        if not dirty_agents:
            return conflicts

        max_t = min(
            window,
            max(len(p) for p in paths.values())
        )

        for t in range(max_t): # scan theo từng timestep
            # kiểm tra vertex conflict
            # position -> list agents
            occupied = defaultdict(list)

            for a in all_agents:
                path = paths[a]
                curr = path[t] if t < len(path) else path[-1]
                occupied[curr].append(a)

            # chỉ check dirty agents
            for pos, agents_here in occupied.items():
                if len(agents_here) < 2:
                    continue

                # chỉ giữ conflicts liên quan dirty agents
                dirty_here = [
                    a for a in agents_here
                    if a in dirty_agents
                ]

                if not dirty_here:
                    continue

                # tạo conflicts
                for da in dirty_here:
                    for other in agents_here:
                        if da == other:
                            continue

                        conflicts.append(
                            Conflict(
                                kind="vertex",
                                a1=da,
                                a2=other,
                                time=t,
                                cell=pos
                            )
                        )

            # kiểm tra edge conflicts
            edge_table = {}
            for a in all_agents:
                if t == 0:
                    continue

                path = paths[a]

                prev = path[t - 1] if t - 1 < len(path) else path[-1]
                curr = path[t] if t < len(path) else path[-1]
                edge = (prev, curr)
                reverse = (curr, prev)

                # Edge swap conflict
                if reverse in edge_table:
                    other_agent = edge_table[reverse]

                    # chỉ giữ conflict nếu liên quan dirty agent
                    if (
                        a in dirty_agents
                        or other_agent in dirty_agents
                    ):
                        conflicts.append(
                            Conflict(
                                kind="edge",
                                a1=other_agent,
                                a2=a,
                                time=t,
                                edge=edge
                            )
                        )
                edge_table[edge] = a
        return conflicts
    
    def _build_conflict_components(self, conflicts: List[Conflict]) -> List[Set[int]]:
        """
        Xây conflict graph, thể hiện rõ từng thành phần local conflict components.
        E.g.: Conflicts A-B, B-C, D-E => [{A, B, C}, {D, E}]
        """
        graph = defaultdict(set)
        # vẽ cạnh tương ứng với mối quan hệ conflict
        for c in conflicts:
            graph[c.a1].add(c.a2)
            graph[c.a2].add(c.a1)

        # duyệt từng đỉnh/agent = DFS
        visited = set()
        components = []
        for agent in graph:
            if agent in visited:
                continue
            comp = set()

            q = deque([agent])
            visited.add(agent)
            while q:
                u = q.popleft()
                comp.add(u)
                for v in graph[u]:
                    if v not in visited:
                        visited.add(v)
                        q.append(v)

            components.append(comp)
        return components
    
    # ------------------------------------------------------------------
    # Thuật toán path planning ở low-level: Space-time (weighted) A* 
    # --> tạo các candidate path của tương lai giả lập
    # ------------------------------------------------------------------
    def _space_time_astar(
        self,
        agent_id: int,
        start: Position,
        goal: Position,
        constraints: List[Constraint]
    ) -> Optional[List[Position]]:
        """
        Space-Time A* low-level planner cho CBS
        State: (row, col, timestep)
        Cost: g = timestep
        Heuristic: Manhattan distance giữa start và goal.
        """
        start_state: SpaceTimeState = (start[0], start[1], 0)

        # priority queue: (f, g, state)
        open_heap = []

        start_h = self._heuristic(start, goal)

        heapq.heappush(open_heap, (start_h, 0, start_state))

        parent: Dict[
            SpaceTimeState, Optional[SpaceTimeState]
        ] = {start_state: None}

        g_score: Dict[SpaceTimeState, int] = {start_state: None}
        found = None
        while open_heap:
            _, g, current = heapq.heappop(open_heap)
            r, c, t = current
            current_pos = (r, c)

            # goal reached
            if current_pos == goal and t > 0:
                found = current
                break

            # planning horizon
            if t >= self.max_plan_depth:
                continue

            candidates = list(self._neighbors(current_pos))

            # WAIT action
            candidates.append(("S", current_pos))
            for _, nxt in candidates:
                nr, nc = nxt
                nt = t+1
                # constraint check
                if self._violates_constraint(agent_id, current_pos, nxt, nt, constraints):
                    continue

                neighbor_state = (nr, nc, nt)

                # movement cost
                ng = g + 1

                # IMPORTANT: chỉ update nếu tìm được path tốt hơn
                if ng >= g_score.get(neighbor_state, float("inf")):
                    continue

                g_score[neighbor_state] = ng
                h = self._heuristic(nxt, goal)
                f = ng + 1.2 * h # weighted A*, w > 1 để tăng speed
                parent[neighbor_state] = current
                heapq.heappush(open_heap, (f, ng, neighbor_state))

        # no solution
        if found is None:
            return None
        
        # reconstruct path
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
        Plan đường đi tập shipper:
        1. Detect conflict
        2. Nếu conflict-free: return solution
        3. Nếu phát hiện conflict:
        - Branching thành 2 child node
        - Mỗi child thêm constraints cho 1 agent
        4. Replan chỉ cho agent bị ảnh hưởng
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
                    path = self._space_time_astar(
                        shipper.id,
                        shipper.position,
                        target,
                        root_constraints
                    )
                    if path is None:
                        path = [shipper.position]
            
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
            conflicts = self._detect_all_conflict(node.paths)

            # nếu tìm được conflict-free solution -> terminate, trả về kết quả hiện tại đã conflict-free
            if not conflicts:
                result = {}
                for shipper in shippers:
                    result[shipper.id] = PlannedPath(
                        shipper_id=shipper.id,
                        path=node.paths[shipper.id],
                        target=targets.get(shipper.id)
                    )
                return result
            
            conflict = conflicts[0] # chỉ chọn conflict đầu tiên để resolve
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
                    new_path = self._space_time_astar(
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

                heapq.heappush(open_list, (child.cost, self._cbs_counter, child)) # expand node có cost nhỏ nhất
                self._cbs_counter += 1

        # fallback
        # nếu CBS fail thì giữ nguyên path cũ nếu có, còn không thì đứng yên
        result = {}
        for shipper in shippers:
            sid = shipper.id

            if (
                existing_paths is not None
                and sid in existing_paths
            ):
                path = existing_paths[sid]
            else:
                path = [shipper.position]

            result[sid] = PlannedPath(
                shipper_id=sid,
                path=path,
                target=targets.get(sid)
            )
        return result
    
    def _solve_component_cbs(
        self,
        component_agents: List[Shipper],
        targets: Dict[int, Position],
        existing_paths: Dict[int, List[Position]]
    ) -> Dict[int, List[Position]]:
        """Chạy CBS local cho 1 conflict component."""
        local_result = self._plan_paths(
            shippers=component_agents,
            targets=targets,
            existing_paths=existing_paths,
            replanning_agents=set(s.id for s in component_agents)
        )

        result = {}
        for sid, plan in local_result.items():
            result[sid] = plan.path
        return result
    
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
    
    def _advance_path(self, shipper_id: int):
        """
        Sau khi agent thực hiện 1 move, remove bước đầu tiên khỏi persistent path.
        Đồng bộ persistent path với vị trí thực tế.
        """
        if shipper_id not in self.current_paths:
            return

        plan = self.current_paths[shipper_id]
        if len(plan.path) >= 2:
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
    
    def _replan_agents(
        self,
        shippers: List[Shipper],
        replanning_agents: List[Shipper],
    ):
        """
        Replan chỉ cho conflict components có chứa replanning agents (dirty agents).
        Các agent khác giữ nguyên persistent paths.
        """
        dirty_ids = {s.id for s in replanning_agents}

        # lấy toàn bộ persistent paths đang có
        existing_paths = {}
        for sid, plan in self.current_paths.items():
            existing_paths[sid] = plan.path

        # dirty agents LUÔN phải được replan trước
        if dirty_ids:
            # lấy target mới
            targets = {
                sid: self.current_targets[sid]
                for sid in dirty_ids
                if sid in self.current_targets
            }
            # lấy đúng dirty agents
            dirty_shippers = [
                s for s in shippers
                if s.id in dirty_ids
            ]
            # replan chỉ dirty agents --> dirty agents tự plan đường mới trước
            replanned = self._plan_paths(
                shippers=dirty_shippers,
                targets=targets,
            )
            # ghi đè lên persistent paths
            for sid, plan in replanned.items():
                self.current_paths[sid] = plan
                existing_paths[sid] = plan.path

        # detect conflicts mới sinh ra giữa dirty agents và các agents còn lại
        # sau khi dirty agents vừa replan
        # conflicts = self._detect_all_conflict(existing_paths)
        conflicts = self._detect_incremental_conflict(existing_paths, dirty_agents=dirty_ids, window=8)
        if not conflicts:
            return
        
        # build local conflict components
        components = self._build_conflict_components(conflicts)

        for comp in components:
            comp_shippers = [s for s in shippers if s.id in comp]
            local_paths = {
                sid: self.current_paths[sid].path
                for sid in comp if sid in self.current_paths
            }

            # local targets cần replan
            targets = {}
            for shipper in comp_shippers:
                sid = shipper.id
                if sid not in self.current_targets:
                    continue
                targets[sid] = self.current_targets[sid]

            # local CBS replanning
            replanned = self._solve_component_cbs(
                component_agents=comp_shippers,
                targets=targets,
                existing_paths=local_paths
            )

            # merge back
            for sid, path in replanned.items():
                self.current_paths[sid] = PlannedPath(
                    shipper_id=sid,
                    path=path,
                    target=targets.get(sid)
                )

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
        # Phase 2: detect future conflicts trên persistent paths
        # --------------------------------------------------------
        existing_paths = {
            sid: p.path for sid, p in self.current_paths.items()
        }
        future_conflicts = self._detect_all_conflict(existing_paths)

        # --------------------------------------------------------
        # Phase 3: MAPD-CBS path planning, 
        # chỉ thực hiện replan cho agents đổi task hoặc persistent paths sắp conflict
        # --------------------------------------------------------
        if replanning_agents or future_conflicts:
            self._replan_agents(shippers, replanning_agents)

        # --------------------------------------------------------
        # Phase 4: execute persistent paths
        # --------------------------------------------------------
        for shipper in shippers:
            sid = shipper.id
            # default idle action
            move = "S"
            cargo_op = 0
            if sid in self.current_paths:
                plan = self.current_paths[sid]
                if len(plan.path) >= 2:
                    # current_pos = plan.path[0]
                    next_pos = plan.path[1] # chỉ thực hiện future timestep đầu tiên
                    move = self._path_to_move(shipper.position, next_pos)

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
            # decide actions
            actions = self._decide_actions(obs)
            # env step
            obs, _, done, _ = self.env.step(actions)
            # consume persistent paths
            for sid, (move, _) in actions.items():
                if sid not in self.current_paths:
                    continue
                plan = self.current_paths[sid]
                # consume exactly 1 timestep
                if len(plan.path) >= 2:
                    self._advance_path(sid)

            if done:
                break

        return self.env.result(
            self.method_name,
            elapsed_sec=time.time() - start_time,
        )
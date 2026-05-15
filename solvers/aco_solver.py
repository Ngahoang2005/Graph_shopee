from __future__ import annotations

import collections
import heapq
import random
import time
from typing import Dict, List, Tuple

from env import DeliveryEnv, Order, Shipper, manhattan, DIRS, SEED
from solvers.solver import Solver

class AgentState:
    """Quản lý trạng thái cá nhân của từng Shipper trong môi trường Online"""
    def __init__(self):
        self.target_oid = -1
        self.phase = "idle"

class ACOSolver(Solver):
    """
    Thuật toán Ant Colony Optimization (ACO) tương thích với 
    Môi trường Online Stateful (self.env.step).
    """

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self.N = self.cfg["N"]
        self.T = self.cfg["T"]
        self.C = self.cfg["C"]
        
        # Bắt buộc dùng self.rng với SEED cố định để hệ thống chấm điểm ổn định
        self.rng = random.Random(SEED + 20)
        
        # --- Tham số ACO ---
        self.num_ants = 10         
        self.num_iterations = 20
        self.evaporation_rate = 0.2
        self.alpha = 1.0           
        self.beta = 5.0       
        self.window_size = 15 
        
        self.pheromones: Dict[Tuple[int, int], float] = collections.defaultdict(lambda: 1.0)
        self.agents = {i: AgentState() for i in range(self.C)}
        
        # --- Radar Surge/Hotspot ---
        self.recent_orders_history: List[Order] = []
        self.surge_detected = False
        self.estimated_hotspot: Tuple[int, int] = (-1, -1)

    def _update_radar(self, t: int, new_orders: List[Order]):
        self.recent_orders_history.extend(new_orders)
        self.recent_orders_history = [o for o in self.recent_orders_history if t - o.appear_t <= 30]
        
        expected_orders = (self.cfg.get("G", 100) / max(1, self.T)) * 30
        if len(self.recent_orders_history) > expected_orders * 1.5:
            self.surge_detected = True
            sum_r = sum(o.sx for o in self.recent_orders_history)
            sum_c = sum(o.sy for o in self.recent_orders_history)
            count = max(1, len(self.recent_orders_history))
            self.estimated_hotspot = (sum_r // count, sum_c // count)
        else:
            self.surge_detected = False

    def _heuristic(self, sh: Shipper, order: Order, t: int) -> float:
        dist = manhattan(sh.r, sh.c, order.sx, order.sy)
        r_base = 10.0 * (0.4 if order.w <= 0.2 else 1.0 if order.w <= 3.0 else 1.5 if order.w <= 10.0 else 2.0 if order.w <= 30.0 else 3.0)
        priority_multiplier = {1: 1.0, 2: 2.0, 3: 3.0}[order.p]
        weight_penalty = 1.0 + (order.w / max(1.0, sh.W_max))
        time_left = max(1, order.et - t - dist)
        urgency = 100.0 / time_left if time_left > 0 else 0.1 
        
        eta = (r_base * priority_multiplier * urgency) / ((dist + 1) * weight_penalty)
        
        if self.surge_detected and self.estimated_hotspot != (-1, -1):
            dist_to_hotspot = manhattan(order.sx, order.sy, self.estimated_hotspot[0], self.estimated_hotspot[1])
            if dist_to_hotspot <= 3:
                eta *= 2.0
        return eta

    def _find_path(self, start: Tuple[int, int], goal: Tuple[int, int], obstacles: set) -> List[str]:
        """Thuật toán A* né chướng ngại vật động (các xe khác)"""
        if start == goal:
            return []
        open_set = []
        heapq.heappush(open_set, (0, 0, start, []))
        visited = set()

        while open_set:
            _, g, (r, c), path = heapq.heappop(open_set)
            if (r, c) == goal:
                return path
            if (r, c) in visited:
                continue
            visited.add((r, c))

            for move, (dr, dc) in DIRS.items():
                if move == "S": continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.N and 0 <= nc < self.N and self.grid[nr][nc] == 0:
                    if (nr, nc) not in visited and (nr, nc) not in obstacles:
                        new_g = g + 1
                        h = manhattan(nr, nc, goal[0], goal[1])
                        heapq.heappush(open_set, (new_g + h, new_g, (nr, nc), path + [move]))
        return []

    def _assign_orders_aco(self, obs: dict):
        """Bầy kiến tìm nhiệm vụ cho các xe rảnh rỗi"""
        t = obs["t"]
        pending = [o for o in obs["orders"].values() if not o.picked]
        free_shippers = [sh for sh in obs["shippers"] if self.agents[sh.id].phase == "idle"]

        if not free_shippers or not pending:
            return

        best_assignment: Dict[int, int] = {} 
        best_epoch_reward = -float('inf')

        for _ in range(self.num_iterations):
            for ant in range(self.num_ants):
                ant_assign = {}
                ant_reward = 0.0
                
                avail = pending.copy()
                self.rng.shuffle(avail) 
                
                for sh in free_shippers:
                    w_carried = sum(obs["orders"][oid].w for oid in sh.bag if oid in obs["orders"])
                    valid = [o for o in avail if len(sh.bag) < sh.K_max and w_carried + o.w <= sh.W_max]
                    if not valid: continue
                        
                    probs = []
                    for o in valid:
                        tau = self.pheromones[(sh.r * self.N + sh.c, o.id)]
                        eta = self._heuristic(sh, o, t)
                        probs.append((tau ** self.alpha) * (eta ** self.beta))
                        
                    s = sum(probs)
                    if s == 0:
                        chosen = self.rng.choice(valid)
                    else:
                        r = self.rng.uniform(0, s)
                        acc = 0.0
                        for idx, p in enumerate(probs):
                            acc += p
                            if acc >= r:
                                chosen = valid[idx]
                                break
                        else:
                            chosen = valid[-1]
                            
                    ant_assign[sh.id] = chosen.id
                    avail.remove(chosen)
                    
                    dist = manhattan(sh.r, sh.c, chosen.sx, chosen.sy)
                    ant_reward += self._heuristic(sh, chosen, t) - (dist * 0.01)

                if ant_reward > best_epoch_reward:
                    best_epoch_reward = ant_reward
                    best_assignment = ant_assign.copy()
        
        for key in list(self.pheromones.keys()):
            self.pheromones[key] *= (1.0 - self.evaporation_rate)
            if self.pheromones[key] < 0.01:
                del self.pheromones[key]
                
        if best_epoch_reward > 0:
            for sh_id, o_id in best_assignment.items():
                sh = next(s for s in obs["shippers"] if s.id == sh_id)
                self.pheromones[(sh.r * self.N + sh.c, o_id)] += 10.0 / self.window_size

        for sid, oid in best_assignment.items():
            self.agents[sid].target_oid = oid
            self.agents[sid].phase = "pickup"

    def _get_action(self, sid: int, sh: Shipper, obs: dict) -> Tuple[str, int]:
        """Quyết định hành động (Move + Op) cho từng xe tại một bước t"""
        agent = self.agents[sid]
        
        # 1. Cập nhật trạng thái
        if agent.target_oid != -1:
            order = obs["orders"].get(agent.target_oid)
            if not order or order.delivered:
                agent.target_oid = -1
                agent.phase = "idle"
            elif agent.phase == "pickup" and order.picked:
                if order.carrier == sid:
                    agent.phase = "deliver"
                else:
                    agent.target_oid = -1
                    agent.phase = "idle"
        
        # 2. Xử lý xe không có mục tiêu
        if agent.target_oid == -1:
            if sh.bag:
                # Giao đơn gấp nhất trong balo
                urgent_oid = min(sh.bag, key=lambda oid: obs["orders"][oid].et)
                agent.target_oid = urgent_oid
                agent.phase = "deliver"
            else:
                return ("S", 0)

        # 3. Tìm mục tiêu & Check Abort
        order = obs["orders"][agent.target_oid]
        goal = (order.sx, order.sy) if agent.phase == "pickup" else (order.ex, order.ey)
        
        if (sh.r, sh.c) == goal:
            if agent.phase == "pickup":
                w_carried = sum(obs["orders"][oid].w for oid in sh.bag)
                if len(sh.bag) >= sh.K_max or w_carried + order.w > sh.W_max:
                    # Balo đầy, hủy đi lấy, ép đi giao
                    agent.target_oid = -1
                    agent.phase = "idle"
                    return ("S", 0)
            return ("S", 1 if agent.phase == "pickup" else 2)

        # 4. Tìm đường A* né xe khác
        obstacles = { (other.r, other.c) for other in obs["shippers"] if other.id != sid }
        path = self._find_path((sh.r, sh.c), goal, obstacles)
        
        if not path:
            # Fallback nếu bị vây: cho phép lập đường đâm xuyên để kỳ vọng xe kia nhường
            path = self._find_path((sh.r, sh.c), goal, set())
            if not path:
                return ("S", 0)

        move = path[0]
        nr, nc = sh.r + DIRS[move][0], sh.c + DIRS[move][1]
        
        # Tính năng Tối ưu: Nếu bước tới là chạm đích, thực hiện (Move, Op) luôn 1 nhịp
        if (nr, nc) == goal:
            op = 1 if agent.phase == "pickup" else 2
            return (move, op)
            
        return (move, 0)

    def run(self) -> dict:
        """Vòng lặp tương tác với Môi trường RL"""
        obs = self.env.observe()
        t0 = time.time()
        
        while not obs["done"]:
            # Cập nhật Radar ngầm
            if obs["new_order_ids"]:
                new_orders = [obs["orders"][oid] for oid in obs["new_order_ids"]]
                self._update_radar(obs["t"], new_orders)
            
            # Re-plan mỗi window
            if obs["t"] % self.window_size == 0 or obs["t"] == 0:
                self._assign_orders_aco(obs)
                
            # Agent tạo action dict
            actions = {}
            for sh in obs["shippers"]:
                actions[sh.id] = self._get_action(sh.id, sh, obs)
                
            # Đẩy vào môi trường và nhận lại Observation mới
            obs, reward, done, info = self.env.step(actions)
            
        elapsed = time.time() - t0
        return self.env.result("Ant Colony Optimization", elapsed)
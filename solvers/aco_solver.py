from __future__ import annotations
import collections
import random
from typing import Dict, List, Optional, Tuple

from env import DeliveryEnv, Order, Shipper, manhattan, SEED
from solvers.solver import Solver, bfs_path

class AgentState:
    def __init__(self):
        self.target_oid: int = -1
        self.phase: str = "idle"
        self.path: List[str] = []

class ACOSolver(Solver):
    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self.N = self.cfg["N"]
        self.T = self.cfg["T"]
        self.rng = random.Random(SEED + 20)
        self.agent_states: Dict[int, AgentState] = {i: AgentState() for i in range(self.cfg["C"])}
        
        # ACO Params
        self.num_ants = 10
        self.num_iterations = 15
        self.evaporation_rate = 0.2
        self.alpha = 1.0
        self.beta = 2.5
        self.window_size = 20
        self.pheromones: Dict[Tuple[int, int], float] = collections.defaultdict(lambda: 1.0)
        
        # Radar Surge
        self.recent_orders_history: List[Order] = []
        self.surge_detected = False
        self.estimated_hotspot: Tuple[int, int] = (-1, -1)

    def _update_radar(self, t: int, new_orders: List[Order]):
        self.recent_orders_history.extend(new_orders)
        self.recent_orders_history = [o for o in self.recent_orders_history if t - o.appear_t <= 30]
        expected_orders = (self.cfg["G"] / max(self.T, 1)) * 30
        if len(self.recent_orders_history) > expected_orders * 1.5:
            self.surge_detected = True
            sum_r = sum(o.sx for o in self.recent_orders_history)
            sum_c = sum(o.sy for o in self.recent_orders_history)
            count = len(self.recent_orders_history)
            self.estimated_hotspot = (sum_r // count, sum_c // count)
        else:
            self.surge_detected = False

    def _heuristic(self, sh: Shipper, order: Order, t: int) -> float:
        dist = manhattan(sh.r, sh.c, order.sx, order.sy)
        base_reward = 10.0 * (0.4 if order.w <= 0.2 else 1.0 if order.w <= 3.0 else 1.5 if order.w <= 10.0 else 2.0 if order.w <= 30.0 else 3.0)
        priority_multiplier = {1: 1.0, 2: 2.0, 3: 3.0}[order.p]
        weight_penalty = 1.0 + (order.w / sh.W_max)
        time_left = max(1, order.et - t - dist)
        urgency = 100.0 / time_left if time_left > 0 else 0.1
        
        eta = (base_reward * priority_multiplier * urgency) / ((dist + 1) * weight_penalty)
        
        if self.surge_detected and self.estimated_hotspot != (-1, -1):
            if manhattan(order.sx, order.sy, self.estimated_hotspot[0], self.estimated_hotspot[1]) <= 3:
                eta *= 2.0
        return eta

    def _run_aco_epoch(self, shippers: List[Shipper], orders: Dict[int, Order], t: int) -> Dict[int, int]:
        best_assignment: Dict[int, int] = {}
        best_epoch_reward = -float('inf')

        # Tìm shipper rảnh (trên xe không có hàng và đang không có target)
        free_shippers = [sh for sh in shippers if not sh.bag and self.agent_states[sh.id].target_oid == -1]
        pending = [o for o in orders.values() if not o.picked and not o.delivered]
        
        if not free_shippers or not pending:
            return {}

        for _ in range(self.num_iterations):
            for _ant in range(self.num_ants):
                ant_assignment = {}
                ant_reward = 0.0
                available_orders = pending.copy()
                self.rng.shuffle(available_orders)
                
                for sh in free_shippers:
                    valid_orders = [o for o in available_orders if len(sh.bag) < sh.K_max and sh.W_max >= sum(orders[i].w for i in sh.bag) + o.w]
                    if not valid_orders: continue
                        
                    probs = [(self.pheromones[(sh.r * self.N + sh.c, o.id)] ** self.alpha) * (self._heuristic(sh, o, t) ** self.beta) for o in valid_orders]
                    sum_prob = sum(probs)
                    
                    if sum_prob == 0:
                        chosen = self.rng.choice(valid_orders)
                    else:
                        r = self.rng.uniform(0, sum_prob)
                        acc = 0.0
                        for idx, p in enumerate(probs):
                            acc += p
                            if acc >= r:
                                chosen = valid_orders[idx]
                                break
                        else:
                            chosen = valid_orders[-1]
                            
                    ant_assignment[sh.id] = chosen.id
                    available_orders.remove(chosen)
                    dist = manhattan(sh.r, sh.c, chosen.sx, chosen.sy)
                    ant_reward += self._heuristic(sh, chosen, t) - (dist * 0.01)

                if ant_reward > best_epoch_reward:
                    best_epoch_reward = ant_reward
                    best_assignment = ant_assignment.copy()
        
        for key in list(self.pheromones.keys()):
            self.pheromones[key] *= (1.0 - self.evaporation_rate)
            if self.pheromones[key] < 0.01: del self.pheromones[key]
                
        if best_epoch_reward > 0:
            for sh_id, o_id in best_assignment.items():
                sh = next(s for s in shippers if s.id == sh_id)
                self.pheromones[(sh.r * self.N + sh.c, o_id)] += 10.0 / self.window_size

        return best_assignment

    def run(self) -> dict:
        obs = self.env.observe()
        
        while not obs["done"]:
            t = obs["t"]
            orders = obs["orders"]
            shippers = obs["shippers"]
            actions = {}

            # Dò Radar xem đơn mới xuất hiện ở đâu
            new_orders = [orders[oid] for oid in obs["new_order_ids"]]
            if new_orders:
                self._update_radar(t, new_orders)

            # Cứ mỗi window_size, gọi não bộ ACO ra phân chia công việc
            if t % self.window_size == 0 or t == 0:
                best_assignments = self._run_aco_epoch(shippers, orders, t)
                for sid, oid in best_assignments.items():
                    self.agent_states[sid].target_oid = oid
                    self.agent_states[sid].phase = "pickup"
                    self.agent_states[sid].path = []

            # Gán Task giao hàng cho ai đang có hàng trên xe
            for sh in shippers:
                state = self.agent_states[sh.id]
                if sh.bag and state.phase != "deliver":
                    urgent_oid = min(sh.bag, key=lambda oid: orders[oid].et if oid in orders else 9999)
                    state.target_oid = urgent_oid
                    state.phase = "deliver"
                    state.path = []

            # Dịch bước đi (1 trong 3 hành động)
            for sh in shippers:
                state = self.agent_states[sh.id]
                if state.target_oid != -1 and state.target_oid in orders:
                    target = orders[state.target_oid]
                    goal = (target.sx, target.sy) if state.phase == "pickup" else (target.ex, target.ey)
                    
                    if (sh.r, sh.c) == goal:
                        if state.phase == "pickup":
                            actions[sh.id] = ("S", 1) # Pick
                            state.target_oid = -1
                        else:
                            actions[sh.id] = ("S", 2) # Deliver
                            state.target_oid = -1
                    else:
                        if not state.path:
                            state.path = bfs_path(obs["grid"], (sh.r, sh.c), goal)
                        
                        if state.path:
                            actions[sh.id] = (state.path.pop(0), 0)
                        else:
                            actions[sh.id] = ("S", 0)
                else:
                    actions[sh.id] = ("S", 0)

            # Step Môi trường
            obs, step_reward, done, info = self.env.step(actions)

            # Trị kẹt xe cục bộ
            for sh in obs["shippers"]:
                state = self.agent_states[sh.id]
                if sh.id in actions and actions[sh.id][0] in ["U", "D", "L", "R"]:
                    old_sh = shippers[sh.id]
                    if (old_sh.r, old_sh.c) == (sh.r, sh.c):
                        state.path = []

        return self.env.result(method="Ant Colony Optimization")
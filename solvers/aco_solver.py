from __future__ import annotations

import collections
import math
import random
import time
from typing import Dict, List, Optional, Tuple

from env import DeliveryEnv, Order, Shipper, manhattan, SEED
from solvers.solver import Solver
from solvers.routing_utils import compute_path
from solvers.simulation_utils import (
    init_shippers,
    execute_moves,
    pickup_phase,
    delivery_phase,
    compute_result
)

class ACOSolver(Solver):
    """
    Thuật toán Ant Colony Optimization (ACO) kết hợp Rolling Horizon 
    và cơ chế dò tìm Hotspot/Surge thời gian thực (Online Routing).
    """

    def __init__(self, env_or_cfg, grid: Optional[List[List[int]]] = None, orders: Optional[List[Order]] = None):
        super().__init__(env_or_cfg, grid, orders)
        # Chuyển list orders thành dict để truy xuất O(1)
        self.orders: Dict[int, Order] = {o.id: o for o in self.orders}
        self.N = self.cfg["N"]
        self.T = self.cfg["T"]
        self.rng = random.Random(SEED + 20)
        
        # --- Tham số ACO ---
        self.num_ants = 20         # Giảm số lượng kiến để chạy kịp trong 60 phút
        self.num_iterations = 15
        self.evaporation_rate = 0.2
        self.alpha = 1.0           # Trọng số Pheromone
        self.beta = 2.5        # Trọng số Heuristic (Đề cao tầm nhìn ngắn hạn)
        self.window_size = 20   # Rolling Horizon: Re-plan mỗi 15 bước thời gian
        
        # Ma trận Pheromone: dict lưu vết mùi giữa 2 đơn hàng (id1 -> id2)
        self.pheromones: Dict[Tuple[int, int], float] = collections.defaultdict(lambda: 1.0)
        
        # --- Radar Surge/Hotspot ---
        self.recent_orders_history: List[Order] = []
        self.surge_detected = False
        self.estimated_hotspot: Tuple[int, int] = (-1, -1)

    def _update_radar(self, t: int, new_orders: List[Order]):
        """Dò tìm Surge và Hotspot ẩn trong Phase 1."""
        self.recent_orders_history.extend(new_orders)
        # Giữ lại lịch sử trong 30 bước gần nhất
        self.recent_orders_history = [o for o in self.recent_orders_history if t - o.appear_t <= 30]
        
        # Nếu có quá nhiều đơn xuất hiện trong thời gian ngắn -> Có Surge
        expected_orders = (self.cfg["G"] / self.T) * 30
        if len(self.recent_orders_history) > expected_orders * 1.5:
            self.surge_detected = True
            # Ước lượng Hotspot bằng cách tìm trung bình cộng tọa độ của các đơn gần đây
            sum_r = sum(o.sx for o in self.recent_orders_history)
            sum_c = sum(o.sy for o in self.recent_orders_history)
            count = len(self.recent_orders_history)
            self.estimated_hotspot = (sum_r // count, sum_c // count)
        else:
            self.surge_detected = False

    def _heuristic(self, sh: Shipper, order: Order, t: int) -> float:
        """Hàm đánh giá độ hấp dẫn của đơn hàng (Thị lực của kiến)."""
        dist = manhattan(sh.r, sh.c, order.sx, order.sy)
        
        # Tiền gốc ước tính và Hệ số ưu tiên (Hỏa tốc x3)
        base_reward = 10.0 * (0.4 if order.w <= 0.2 else 1.0 if order.w <= 3.0 else 1.5 if order.w <= 10.0 else 2.0 if order.w <= 30.0 else 3.0)
        priority_multiplier = {1: 1.0, 2: 2.0, 3: 3.0}[order.p]
        
        # Điểm trừ nếu đơn quá nặng (gây tốn chi phí di chuyển rc)
        weight_penalty = 1.0 + (order.w / sh.W_max)
        
        # Độ gấp gáp (Càng gần deadline et, càng phải ưu tiên)
        time_left = max(1, order.et - t - dist)
        urgency = 100.0 / time_left if time_left > 0 else 0.1 # Trễ rồi thì bỏ qua
        
        eta = (base_reward * priority_multiplier * urgency) / ((dist + 1) * weight_penalty)
        
        # Cực kỳ quan trọng: Nếu Radar báo có Surge, hút kiến về phía Hotspot
        if self.surge_detected and self.estimated_hotspot != (-1, -1):
            dist_to_hotspot = manhattan(order.sx, order.sy, self.estimated_hotspot[0], self.estimated_hotspot[1])
            if dist_to_hotspot <= 3:
                eta *= 2.0  # Tăng gấp đôi lực hút
                
        return eta

    def _run_aco_epoch(self, shippers: List[Shipper], pending: List[Order], t: int) -> Dict[int, int]:
        """Chạy một kỷ nguyên ACO để tìm target_oid tốt nhất cho các shipper đang rảnh."""
        best_assignment: Dict[int, int] = {} # shipper_id -> order_id
        best_epoch_reward = -float('inf')

        # Lọc ra các shipper đang rảnh rỗi (idle)
        free_shippers = [sh for sh in shippers if sh.phase == "idle" and sh.target_oid == -1]
        if not free_shippers or not pending:
            return {}

        for _ in range(self.num_iterations):
            for ant in range(self.num_ants):
                ant_assignment = {}
                ant_reward = 0.0
                
                # Clone trạng thái ảo để kiến bò mà không ảnh hưởng thực tế
                available_orders = pending.copy()
                # random
                self.rng.shuffle(available_orders)
                
                for sh in free_shippers:
                    valid_orders = [o for o in available_orders if sh.can_pickup(o, self.orders)]
                    if not valid_orders:
                        continue
                        
                    # Tính xác suất chọn dựa trên Pheromone và Heuristic
                    probabilities = []
                    for o in valid_orders:
                        # Vết mùi từ vị trí hiện tại đến đơn hàng
                        tau = self.pheromones[(sh.r * self.N + sh.c, o.id)]
                        eta = self._heuristic(sh, o, t)
                        prob = (tau ** self.alpha) * (eta ** self.beta)
                        probabilities.append(prob)
                        
                    sum_prob = sum(probabilities)
                    if sum_prob == 0:
                        chosen_order = self.rng.choice(valid_orders)
                    else:
                        # Roulette Wheel Selection
                        r = self.rng.uniform(0, sum_prob)
                        acc = 0.0
                        for idx, prob in enumerate(probabilities):
                            acc += prob
                            if acc >= r:
                                chosen_order = valid_orders[idx]
                                break
                        else:
                            chosen_order = valid_orders[-1]
                            
                    ant_assignment[sh.id] = chosen_order.id
                    available_orders.remove(chosen_order)
                    
                    # Ước lượng Net Reward ảo mà con kiến này đạt được
                    dist = manhattan(sh.r, sh.c, chosen_order.sx, chosen_order.sy)
                    ant_reward += self._heuristic(sh, chosen_order, t) - (dist * 0.01)

                # Cập nhật Best Global Solution
                if ant_reward > best_epoch_reward:
                    best_epoch_reward = ant_reward
                    best_assignment = ant_assignment.copy()
        
        # --- Cập nhật Pheromone (Global Update) ---
        # 1. Bay hơi (Evaporation)
        for key in list(self.pheromones.keys()):
            self.pheromones[key] *= (1.0 - self.evaporation_rate)
            if self.pheromones[key] < 0.01:
                del self.pheromones[key] # Dọn dẹp bộ nhớ
                
        # 2. Rắc mùi (Deposit) lên lộ trình tốt nhất
        if best_epoch_reward > 0:
            deposit_amount = 10.0 / self.window_size
            for sh_id, o_id in best_assignment.items():
                sh = next(s for s in shippers if s.id == sh_id)
                self.pheromones[(sh.r * self.N + sh.c, o_id)] += deposit_amount

        return best_assignment

    def run(self) -> dict:
        t0 = time.time()
        
        # Phân loại đơn hàng theo thời gian xuất hiện để mô phỏng Real-time
        orders_by_t: Dict[int, List[Order]] = collections.defaultdict(list)
        for order in self.orders.values():
            orders_by_t[order.appear_t].append(order)

        # Khởi tạo mô phỏng sử dụng simulation_utils.py
        free_cells = [(r, c) for r in range(self.N) for c in range(self.N) if self.grid[r][c] == 0]
        self.rng.shuffle(free_cells)
        shippers = init_shippers(self.cfg, free_cells)

        pending: List[Order] = []
        in_transit: List[Order] = []
        delivered: List[Order] = []
        
        total_reward = 0.0
        total_movecost = 0.0

        # Vòng lặp thời gian chính
        for t in range(self.T):
            # 1. Đơn hàng mới rơi xuống
            new_orders = orders_by_t.get(t, [])
            if new_orders:
                pending.extend(new_orders)
                self._update_radar(t, new_orders)

            # 2. Chiến lược Rolling Horizon: Chạy bộ não ACO mỗi `window_size` bước
            if t % self.window_size == 0 or t == 0:
                best_assignments = self._run_aco_epoch(shippers, pending, t)
                
                # Gán mục tiêu (Target) cho shipper dựa trên giải pháp của kiến
                for sh in shippers:
                    if sh.id in best_assignments:
                        sh.target_oid = best_assignments[sh.id]
                        sh.phase = "pickup"
                        sh.path = [] # Xóa đường cũ để tính lại bằng BFS/A*

            # 3. Chuyển Phase cho các Shipper đang có hàng trong túi nhưng chưa có mục tiêu
            for sh in shippers:
                if sh.target_oid == -1 and sh.bag:
                    # Ưu tiên giao đơn sắp hết hạn trước
                    urgent_oid = min(sh.bag, key=lambda oid: self.orders[oid].et)
                    sh.target_oid = urgent_oid
                    sh.phase = "deliver"
                    sh.path = []

            # 4. Tìm đường di chuyển vật lý trên lưới (Routing)
            for sh in shippers:
                if sh.target_oid != -1 and not sh.path:
                    target_order = self.orders[sh.target_oid]
                    goal = (target_order.sx, target_order.sy) if sh.phase == "pickup" else (target_order.ex, target_order.ey)
                    
                    if (sh.r, sh.c) != goal:
                        # Sử dụng routing_utils đã cung cấp
                        sh.path = compute_path(self.grid, (sh.r, sh.c), goal)

            # 5. Thực thi di chuyển (Simulation Utils)
            total_movecost += execute_moves(shippers, self.grid, self.orders)

            # 6. Nhặt hàng (Pickup Phase)
            pickup_phase(shippers, pending, in_transit, self.orders)

            # 7. Giao hàng (Delivery Phase)
            total_reward += delivery_phase(shippers, self.orders, in_transit, delivered, t, self.T)

        elapsed = time.time() - t0
        
        # Đóng gói kết quả
        return compute_result(
            method_name="Ant Colony Optimization",
            cfg=self.cfg,
            orders=self.orders,
            delivered_orders=delivered,
            shippers=shippers,
            total_reward=total_reward,
            total_movecost=total_movecost,
            elapsed=elapsed
        )
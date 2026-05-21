"""
vrp_ortools.py — VRP solver tự implement lại thuật toán cốt lõi của OR-Tools
==============================================================================

Không dùng bất kỳ thư viện ngoài nào. Toàn bộ thuật toán được viết từ đầu,
mô phỏng đúng pipeline mà Google OR-Tools Routing Library thực hiện nội bộ:

  Layer 1 — BFSGraph:
      BFS trên grid, cache khoảng cách & path, tái sử dụng xuyên suốt episode.
      Tương đương OR-Tools: RoutingIndexManager + distance callback.

  Layer 2 — Construction (First Solution Strategy):
      ClarkeWright : Clarke-Wright Savings + Nearest-Neighbor ordering.
      OR-Tools tương đương: FirstSolutionStrategy.SAVINGS.

  Layer 3 — Local Search Operators:
      relocate : di chuyển 1 node sang route/vị trí khác.   (OR-Tools: Relocate)
      two_opt  : đảo đoạn trong cùng route.                  (OR-Tools: TwoOpt)
      cross    : hoán đổi node giữa 2 route.                 (OR-Tools: Cross)
      or_opt   : di chuyển chuỗi 2-3 node.                   (OR-Tools: OrOpt)

  Layer 4 — Meta-heuristic:
      GuidedLocalSearch : penalize arc hay gặp ở local optima.
                          OR-Tools: GUIDED_LOCAL_SEARCH.
      LargeNeighborhoodSearch : Ruin (worst/random removal) + Recreate (regret-2).
                          OR-Tools: ruin-and-recreate / lns_time_limit.

  Layer 5 — Online controller:
      VRPOrToolsSolver : vòng lặp online với DeliveryEnv.
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from env import DeliveryEnv, Order, Shipper


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — BFSGraph
# ══════════════════════════════════════════════════════════════════════════════

class BFSGraph:
    """
    BFS trên grid với full cache.
    Gọi _bfs(src) một lần → cache dist & path từ src đến mọi cell có thể đến.
    """

    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    INF  = 10 ** 7

    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows else 0
        self._dist: Dict[Tuple, int]                   = {}
        self._path: Dict[Tuple, List[Tuple[int, int]]] = {}

    def _valid(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] == 0

    def _bfs(self, src: Tuple[int, int]) -> None:
        dist:   Dict[Tuple, int]            = {src: 0}
        parent: Dict[Tuple, Optional[Tuple]]= {src: None}
        q = deque([src])
        while q:
            r, c = q.popleft()
            for dr, dc in self.DIRS:
                nb = (r + dr, c + dc)
                if nb not in dist and self._valid(nb[0], nb[1]):
                    dist[nb]   = dist[(r, c)] + 1
                    parent[nb] = (r, c)
                    q.append(nb)

        for dst, d in dist.items():
            self._dist[(src, dst)] = d
            if (dst, src) not in self._dist:
                self._dist[(dst, src)] = d

        for dst in dist:
            if (src, dst) in self._path:
                continue
            if dst == src:
                self._path[(src, dst)] = []
                continue
            p: List[Tuple[int, int]] = []
            cur = dst
            while cur != src:
                p.append(cur)
                cur = parent[cur]  # type: ignore[arg-type]
            p.reverse()
            self._path[(src, dst)] = p

    def dist(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        if a == b:
            return 0
        if (a, b) not in self._dist:
            self._bfs(a)
        return self._dist.get((a, b), self.INF)

    def path(self, a: Tuple[int, int], b: Tuple[int, int]) -> List[Tuple[int, int]]:
        if a == b:
            return []
        if (a, b) not in self._path:
            self._bfs(a)
        return list(self._path.get((a, b), []))


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — Cấu trúc dữ liệu VRP
# ══════════════════════════════════════════════════════════════════════════════

class VRPNode:
    """Node VRP = điểm pickup của 1 đơn hàng."""
    __slots__ = ("node_id", "pos", "order_id", "demand")

    def __init__(self, node_id: int, pos: Tuple[int, int], order_id: int, demand: float):
        self.node_id  = node_id
        self.pos      = pos
        self.order_id = order_id
        self.demand   = demand


class VRPRoute:
    """Route của 1 vehicle (shipper)."""

    def __init__(self, vid: int, depot: Tuple[int, int], cap_w: float, cap_k: int):
        self.vid    = vid
        self.depot  = depot
        self.cap_w  = cap_w
        self.cap_k  = cap_k
        self.nodes: List[VRPNode] = []

    def load_w(self) -> float:
        return sum(n.demand for n in self.nodes)

    def feasible_add(self, node: VRPNode) -> bool:
        return (self.load_w() + node.demand <= self.cap_w and
                len(self.nodes) + 1 <= self.cap_k)

    def cost(self, g: BFSGraph) -> int:
        if not self.nodes:
            return 0
        pts = [self.depot] + [n.pos for n in self.nodes] + [self.depot]
        return sum(g.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))

    def copy(self) -> "VRPRoute":
        r = VRPRoute(self.vid, self.depot, self.cap_w, self.cap_k)
        r.nodes = list(self.nodes)
        return r


class VRPSolution:
    def __init__(self, routes: List[VRPRoute]):
        self.routes = routes

    def total_cost(self, g: BFSGraph) -> int:
        return sum(r.cost(g) for r in self.routes)

    def copy(self) -> "VRPSolution":
        return VRPSolution([r.copy() for r in self.routes])


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — Clarke-Wright Savings (First Solution Strategy)
# ══════════════════════════════════════════════════════════════════════════════

class ClarkeWright:
    """
    Clarke-Wright Savings + Nearest-Neighbor ordering.
    OR-Tools tương đương: FirstSolutionStrategy.SAVINGS.
    """

    def build(
        self,
        depots: List[Tuple[int, int]],
        nodes : List[VRPNode],
        cap_w : List[float],
        cap_k : List[int],
        g     : BFSGraph,
    ) -> VRPSolution:
        C = len(depots)
        routes = [VRPRoute(v, depots[v], cap_w[v], cap_k[v]) for v in range(C)]
        if not nodes:
            return VRPSolution(routes)

        # Gán node cho vehicle gần nhất có capacity
        for node in nodes:
            order = sorted(range(C), key=lambda v: g.dist(depots[v], node.pos))
            for v in order:
                if routes[v].feasible_add(node):
                    routes[v].nodes.append(node)
                    break
            # Nếu không vehicle nào nhận được → bỏ qua (sẽ re-plan sau)

        # Sắp xếp thứ tự node trong route bằng Nearest-Neighbor từ depot
        for v in range(C):
            r = routes[v]
            if len(r.nodes) <= 1:
                continue
            unvisited = list(r.nodes)
            ordered   = []
            cur       = r.depot
            while unvisited:
                nxt = min(unvisited, key=lambda n: g.dist(cur, n.pos))
                ordered.append(nxt)
                cur = nxt.pos
                unvisited.remove(nxt)
            r.nodes = ordered

        return VRPSolution(routes)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — Local Search Operators
# ══════════════════════════════════════════════════════════════════════════════

class LocalSearch:
    """
    Các LS operator.  Mỗi operator trả (new_sol, delta) hoặc None.
    delta < 0 → cải thiện.  Dùng GLS penalty để augmented cost.
    """

    # ── Relocate ─────────────────────────────────────────────────────────────
    @staticmethod
    def relocate(
        sol: VRPSolution, g: BFSGraph,
        pen: Dict[Tuple, float], lam: float = 0.15,
    ) -> Optional[Tuple[VRPSolution, int]]:
        best_delta = 0
        best_move  = None   # (sv, si, dv, di)

        for sv, r_src in enumerate(sol.routes):
            if not r_src.nodes:
                continue
            sp = [r_src.depot] + [n.pos for n in r_src.nodes] + [r_src.depot]
            for si, node in enumerate(r_src.nodes):
                gain = (g.dist(sp[si], sp[si + 1]) +
                        g.dist(sp[si + 1], sp[si + 2]) -
                        g.dist(sp[si], sp[si + 2]))
                for dv, r_dst in enumerate(sol.routes):
                    if dv == sv and len(r_src.nodes) == 1:
                        continue
                    if dv != sv and not r_dst.feasible_add(node):
                        continue
                    dp = [r_dst.depot] + [n.pos for n in r_dst.nodes] + [r_dst.depot]
                    for di in range(len(dp) - 1):
                        if dv == sv and (di == si or di == si - 1):
                            continue
                        ins = (g.dist(dp[di], node.pos) +
                               g.dist(node.pos, dp[di + 1]) -
                               g.dist(dp[di], dp[di + 1]))
                        p = lam * (pen.get((dp[di], node.pos), 0) +
                                   pen.get((node.pos, dp[di + 1]), 0))
                        delta = ins - gain + p
                        if delta < best_delta:
                            best_delta = delta
                            best_move  = (sv, si, dv, di)

        if best_move is None:
            return None
        sv, si, dv, di = best_move
        ns = sol.copy()
        node = ns.routes[sv].nodes.pop(si)
        real_di = (di if di < si else di - 1) if dv == sv else di
        ns.routes[dv].nodes.insert(real_di, node)
        return ns, best_delta

    # ── 2-Opt ────────────────────────────────────────────────────────────────
    @staticmethod
    def two_opt(
        sol: VRPSolution, g: BFSGraph,
        pen: Dict[Tuple, float], lam: float = 0.15,
    ) -> Optional[Tuple[VRPSolution, int]]:
        best_delta = 0
        best_move  = None

        for v, route in enumerate(sol.routes):
            if len(route.nodes) < 2:
                continue
            pts = [route.depot] + [n.pos for n in route.nodes] + [route.depot]
            for i in range(len(pts) - 2):
                for j in range(i + 2, len(pts) - 1):
                    d_old = g.dist(pts[i], pts[i + 1]) + g.dist(pts[j], pts[j + 1])
                    d_new = g.dist(pts[i], pts[j])     + g.dist(pts[i + 1], pts[j + 1])
                    p = lam * (pen.get((pts[i], pts[j]), 0) +
                               pen.get((pts[i + 1], pts[j + 1]), 0) -
                               pen.get((pts[i], pts[i + 1]), 0) -
                               pen.get((pts[j], pts[j + 1]), 0))
                    delta = d_new - d_old + p
                    if delta < best_delta:
                        best_delta = delta
                        best_move  = (v, i, j)

        if best_move is None:
            return None
        v, i, j = best_move
        ns = sol.copy()
        seg = ns.routes[v].nodes[i:j]
        seg.reverse()
        ns.routes[v].nodes[i:j] = seg
        return ns, best_delta

    # ── Cross ────────────────────────────────────────────────────────────────
    @staticmethod
    def cross(
        sol: VRPSolution, g: BFSGraph,
        pen: Dict[Tuple, float], lam: float = 0.15,
    ) -> Optional[Tuple[VRPSolution, int]]:
        best_delta = 0
        best_move  = None
        C = len(sol.routes)

        for v1 in range(C):
            for v2 in range(v1 + 1, C):
                r1, r2  = sol.routes[v1], sol.routes[v2]
                pts1 = [r1.depot] + [n.pos for n in r1.nodes] + [r1.depot]
                pts2 = [r2.depot] + [n.pos for n in r2.nodes] + [r2.depot]
                for i1, n1 in enumerate(r1.nodes):
                    for i2, n2 in enumerate(r2.nodes):
                        # weight feasibility
                        if (r1.load_w() - n1.demand + n2.demand > r1.cap_w or
                                r2.load_w() - n2.demand + n1.demand > r2.cap_w):
                            continue
                        old = (g.dist(pts1[i1], pts1[i1+1]) + g.dist(pts1[i1+1], pts1[i1+2]) +
                               g.dist(pts2[i2], pts2[i2+1]) + g.dist(pts2[i2+1], pts2[i2+2]))
                        new = (g.dist(pts1[i1], n2.pos) + g.dist(n2.pos, pts1[i1+2]) +
                               g.dist(pts2[i2], n1.pos) + g.dist(n1.pos, pts2[i2+2]))
                        if new - old < best_delta:
                            best_delta = new - old
                            best_move  = (v1, i1, v2, i2)

        if best_move is None:
            return None
        v1, i1, v2, i2 = best_move
        ns = sol.copy()
        ns.routes[v1].nodes[i1], ns.routes[v2].nodes[i2] = (
            ns.routes[v2].nodes[i2], ns.routes[v1].nodes[i1])
        return ns, best_delta

    # ── Or-Opt ───────────────────────────────────────────────────────────────
    @staticmethod
    def or_opt(
        sol: VRPSolution, g: BFSGraph,
        pen: Dict[Tuple, float], lam: float = 0.15,
        seg_len: int = 2,
    ) -> Optional[Tuple[VRPSolution, int]]:
        best_delta = 0
        best_move  = None

        for sv, r_src in enumerate(sol.routes):
            if len(r_src.nodes) < seg_len:
                continue
            sp = [r_src.depot] + [n.pos for n in r_src.nodes] + [r_src.depot]
            for si in range(len(r_src.nodes) - seg_len + 1):
                seg = r_src.nodes[si: si + seg_len]
                seg_int = sum(g.dist(seg[k].pos, seg[k+1].pos) for k in range(seg_len - 1))
                gain = (g.dist(sp[si], sp[si+1]) +
                        g.dist(sp[si+seg_len], sp[si+seg_len+1]) -
                        g.dist(sp[si], sp[si+seg_len+1]) - seg_int)
                extra_w = sum(n.demand for n in seg)
                for dv, r_dst in enumerate(sol.routes):
                    if dv == sv and len(r_src.nodes) == seg_len:
                        continue
                    if dv != sv:
                        if r_dst.load_w() + extra_w > r_dst.cap_w:
                            continue
                        if len(r_dst.nodes) + seg_len > r_dst.cap_k:
                            continue
                    dp = [r_dst.depot] + [n.pos for n in r_dst.nodes] + [r_dst.depot]
                    for di in range(len(dp) - 1):
                        if dv == sv and si <= di < si + seg_len:
                            continue
                        ins = (g.dist(dp[di], seg[0].pos) + seg_int +
                               g.dist(seg[-1].pos, dp[di+1]) -
                               g.dist(dp[di], dp[di+1]))
                        delta = ins - gain
                        if delta < best_delta:
                            best_delta = delta
                            best_move  = (sv, si, dv, di)

        if best_move is None:
            return None
        sv, si, dv, di = best_move
        ns  = sol.copy()
        seg = [ns.routes[sv].nodes.pop(si) for _ in range(seg_len)]
        real_di = (di if di < si else di - seg_len) if dv == sv else di
        for k, node in enumerate(seg):
            ns.routes[dv].nodes.insert(real_di + k, node)
        return ns, best_delta


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4A — Guided Local Search (GLS)
# ══════════════════════════════════════════════════════════════════════════════

class GuidedLocalSearch:
    """
    Guided Local Search.
    OR-Tools: LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH.

    Sau mỗi local optimum:
        utility(arc) = cost(arc) / (1 + penalty(arc))
        → tăng penalty cho arc có utility cao nhất
    """

    def __init__(self, lam: float = 0.15):
        self.lam = lam
        self.pen: Dict[Tuple, float] = {}

    def _update(self, sol: VRPSolution, g: BFSGraph) -> None:
        arcs = []
        for r in sol.routes:
            pts = [r.depot] + [n.pos for n in r.nodes] + [r.depot]
            for i in range(len(pts) - 1):
                c = g.dist(pts[i], pts[i+1])
                if c > 0:
                    arcs.append((pts[i], pts[i+1], c))
        if not arcs:
            return
        max_u = max(c / (1 + self.pen.get((a, b), 0)) for a, b, c in arcs)
        for a, b, c in arcs:
            if c / (1 + self.pen.get((a, b), 0)) >= max_u - 1e-9:
                self.pen[(a, b)] = self.pen.get((a, b), 0) + 1

    def optimize(
        self,
        sol       : VRPSolution,
        g         : BFSGraph,
        time_limit: float = 0.4,
        max_iter  : int   = 300,
    ) -> VRPSolution:
        ops = [
            LocalSearch.relocate,
            LocalSearch.two_opt,
            LocalSearch.cross,
            lambda s, gr, p, l: LocalSearch.or_opt(s, gr, p, l, 2),
            lambda s, gr, p, l: LocalSearch.or_opt(s, gr, p, l, 3),
        ]
        current   = sol.copy()
        best      = sol.copy()
        best_cost = best.total_cost(g)
        t0        = time.time()

        for _ in range(max_iter):
            if time.time() - t0 > time_limit:
                break
            improved = False
            for op in ops:
                res = op(current, g, self.pen, self.lam)
                if res is not None:
                    current, _ = res
                    improved   = True
                    c = current.total_cost(g)
                    if c < best_cost:
                        best      = current.copy()
                        best_cost = c
            if not improved:
                self._update(current, g)

        return best


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4B — Large Neighborhood Search (LNS)
# ══════════════════════════════════════════════════════════════════════════════

class LargeNeighborhoodSearch:
    """
    LNS: Ruin + Recreate.
    OR-Tools tương đương: ruin-and-recreate.

    Ruin    : worst-removal (alternating với random-removal).
    Recreate: regret-2 insertion.
    Accept  : greedy (chỉ nhận nghiệm tốt hơn best).
    """

    def __init__(self, rng: random.Random, q_frac: float = 0.35):
        self.rng    = rng
        self.q_frac = q_frac

    # ── Worst removal (fixed: không dùng index shift) ─────────────────────────
    def _worst_removal(
        self, sol: VRPSolution, g: BFSGraph, q: int
    ) -> Tuple[VRPSolution, List[VRPNode]]:
        # Tính contribution của từng node
        contribs: List[Tuple[int, int, int]] = []   # (cost, vi, ni)
        for vi, route in enumerate(sol.routes):
            pts = [route.depot] + [n.pos for n in route.nodes] + [route.depot]
            for ni in range(len(route.nodes)):
                c = (g.dist(pts[ni], pts[ni+1]) +
                     g.dist(pts[ni+1], pts[ni+2]) -
                     g.dist(pts[ni], pts[ni+2]))
                contribs.append((c, vi, ni))
        contribs.sort(reverse=True)

        # Đánh dấu node cần xóa theo (vi, ni) — dùng set để tránh trùng
        to_remove: Dict[int, Set[int]] = {}   # vi → set of ni
        removed_nodes: List[VRPNode]   = []
        for _, vi, ni in contribs:
            if sum(len(s) for s in to_remove.values()) >= q:
                break
            to_remove.setdefault(vi, set()).add(ni)

        # Copy và xóa theo thứ tự ngược (ni lớn trước để không lệch index)
        ns = sol.copy()
        for vi, ni_set in to_remove.items():
            for ni in sorted(ni_set, reverse=True):
                removed_nodes.append(ns.routes[vi].nodes[ni])
                ns.routes[vi].nodes.pop(ni)
        return ns, removed_nodes

    # ── Random removal ────────────────────────────────────────────────────────
    def _random_removal(
        self, sol: VRPSolution, q: int
    ) -> Tuple[VRPSolution, List[VRPNode]]:
        all_pos = [(vi, ni)
                   for vi, r in enumerate(sol.routes)
                   for ni in range(len(r.nodes))]
        self.rng.shuffle(all_pos)
        chosen = all_pos[:q]
        # Group và xóa ngược index
        by_v: Dict[int, List[int]] = {}
        for vi, ni in chosen:
            by_v.setdefault(vi, []).append(ni)

        ns = sol.copy()
        removed: List[VRPNode] = []
        for vi, ni_list in by_v.items():
            for ni in sorted(ni_list, reverse=True):
                removed.append(ns.routes[vi].nodes[ni])
                ns.routes[vi].nodes.pop(ni)
        return ns, removed

    # ── Regret-2 insertion ───────────────────────────────────────────────────
    def _regret_insert(
        self, sol: VRPSolution, removed: List[VRPNode], g: BFSGraph,
    ) -> VRPSolution:
        remaining = list(removed)
        cur = sol.copy()

        while remaining:
            best_regret = -math.inf
            chosen_node = None
            chosen_vi   = -1
            chosen_pos  = 0

            for node in remaining:
                opts = []
                for vi, route in enumerate(cur.routes):
                    if not route.feasible_add(node):
                        continue
                    pts = [route.depot] + [n.pos for n in route.nodes] + [route.depot]
                    bc, bp = math.inf, 0
                    for di in range(len(pts) - 1):
                        c = (g.dist(pts[di], node.pos) +
                             g.dist(node.pos, pts[di+1]) -
                             g.dist(pts[di], pts[di+1]))
                        if c < bc:
                            bc, bp = c, di
                    if bc < math.inf:
                        opts.append((bc, vi, bp))

                if not opts:
                    continue
                opts.sort()
                c1      = opts[0][0]
                c2      = opts[1][0] if len(opts) > 1 else c1 + 1e9
                regret  = c2 - c1
                if regret > best_regret:
                    best_regret = regret
                    chosen_node = node
                    chosen_vi   = opts[0][1]
                    chosen_pos  = opts[0][2]

            if chosen_node is None:
                break
            cur.routes[chosen_vi].nodes.insert(chosen_pos, chosen_node)
            remaining.remove(chosen_node)

        return cur

    # ── Main ─────────────────────────────────────────────────────────────────
    def optimize(
        self,
        sol       : VRPSolution,
        g         : BFSGraph,
        time_limit: float = 0.3,
        max_iter  : int   = 60,
    ) -> VRPSolution:
        best      = sol.copy()
        best_cost = best.total_cost(g)
        current   = sol.copy()
        t0        = time.time()
        total_n   = sum(len(r.nodes) for r in sol.routes)
        q         = max(1, int(self.q_frac * total_n))

        for it in range(max_iter):
            if time.time() - t0 > time_limit:
                break
            if it % 2 == 0:
                ruined, removed = self._worst_removal(current, g, q)
            else:
                ruined, removed = self._random_removal(current, q)

            repaired = self._regret_insert(ruined, removed, g)
            cost     = repaired.total_cost(g)
            if cost < best_cost:
                best      = repaired.copy()
                best_cost = cost
            current = repaired

        return best


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5A — ORToolsVRPPipeline
# ══════════════════════════════════════════════════════════════════════════════

class ORToolsVRPPipeline:
    """
    Pipeline thuần Python mô phỏng OR-Tools Routing solver:
      1. ClarkeWright  (First Solution)
      2. GLS + LS      (Improve)
      3. LNS           (Large neighborhood)
    """

    def __init__(self, g: BFSGraph, rng: Optional[random.Random] = None):
        self.g   = g
        self.rng = rng or random.Random(42)
        self.gls = GuidedLocalSearch(lam=0.15)
        self.lns = LargeNeighborhoodSearch(self.rng, q_frac=0.35)

    def solve(
        self,
        depots    : List[Tuple[int, int]],
        nodes     : List[VRPNode],
        cap_w     : List[float],
        cap_k     : List[int],
        time_limit: float = 0.5,
    ) -> List[List[VRPNode]]:
        C = len(depots)
        if not nodes:
            return [[] for _ in range(C)]

        # Step 1: Clarke-Wright
        sol = ClarkeWright().build(depots, nodes, cap_w, cap_k, self.g)

        t0        = time.time()
        remaining = time_limit

        # Step 2: GLS (60% budget)
        sol = self.gls.optimize(sol, self.g,
                                time_limit=remaining * 0.60,
                                max_iter=500)
        remaining -= (time.time() - t0)

        # Step 3: LNS (còn lại)
        total_n = sum(len(r.nodes) for r in sol.routes)
        if remaining > 0.04 and total_n > 1:
            sol = self.lns.optimize(sol, self.g,
                                    time_limit=remaining * 0.88,
                                    max_iter=80)

        return [r.nodes for r in sol.routes]


# ══════════════════════════════════════════════════════════════════════════════
# ShipperPlan — waypoint queue
# ══════════════════════════════════════════════════════════════════════════════

class ShipperPlan:
    __slots__ = ("vid", "waypoints", "path_buf", "assigned")

    def __init__(self, vid: int):
        self.vid      = vid
        self.waypoints: deque                 = deque()
        self.path_buf : List[Tuple[int, int]] = []
        self.assigned : List[int]             = []

    def clear(self):
        self.waypoints.clear()
        self.path_buf.clear()
        self.assigned.clear()

    def add(self, pos: Tuple[int, int], kind: str, oid: Optional[int] = None):
        self.waypoints.append((pos, kind, oid))

    def has_work(self) -> bool:
        return bool(self.waypoints) or bool(self.path_buf)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5B — VRPOrToolsSolver (online controller)
# ══════════════════════════════════════════════════════════════════════════════

class VRPOrToolsSolver:
    """
    Online VRP solver tích hợp DeliveryEnv.
    Pipeline: ClarkeWright → GLS (Relocate/2Opt/Cross/OrOpt) → LNS (Regret-2).
    Không dùng bất kỳ thư viện ngoài nào.
    """

    REPLAN_INTERVAL = 5   # re-plan tối đa mỗi 5 bước

    def __init__(self, env: DeliveryEnv):
        self.env = env
        self.cfg = env.public_cfg if hasattr(env, "public_cfg") else env.cfg
        self.C   = self.cfg["C"]
        self.T   = self.cfg["T"]

        self.g        = BFSGraph(env.grid)
        self.rng      = random.Random(7)
        self.pipeline = ORToolsVRPPipeline(self.g, self.rng)

        self.plans   : Dict[int, ShipperPlan] = {i: ShipperPlan(i) for i in range(self.C)}
        self.planned : Set[int]               = set()
        self._last_replan = -self.REPLAN_INTERVAL

    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> dict:
        obs    = self.env.reset()
        t_wall = time.time()

        while not obs["done"]:
            t        = obs["t"]
            shippers = obs["shippers"]
            orders   = obs["orders"]

            unassigned = [
                o for o in orders.values()
                if not o.picked and not o.delivered and o.id not in self.planned
            ]

            # Re-plan nếu:
            #   a) Có đơn mới VÀ không có shipper nào đang làm việc (idle)
            #   b) Có đơn mới VÀ đến kỳ replan
            idle_shippers = [v for v in range(self.C) if not self.plans[v].has_work()]
            should_replan = bool(unassigned) and (
                idle_shippers or t - self._last_replan >= self.REPLAN_INTERVAL
            )

            if should_replan:
                self._replan(shippers, orders, t)
                self._last_replan = t

            actions = self._build_actions(shippers, orders)
            obs, _, _, _ = self.env.step(actions)

        return self.env.result("VRPOrToolsSolver", time.time() - t_wall)

    # ─────────────────────────────────────────────────────────────────────────

    def _replan(
        self,
        shippers: List[Shipper],
        orders  : Dict[int, Order],
        t       : int,
    ) -> None:
        candidates = [o for o in orders.values() if not o.picked and not o.delivered]
        if not candidates:
            return

        # Ưu tiên deadline sớm và priority cao
        candidates.sort(key=lambda o: (o.et - t, -o.p))

        depots = [(s.r, s.c) for s in shippers]
        cap_w  = [s.W_max    for s in shippers]
        cap_k  = [s.K_max    for s in shippers]
        w_used = [sum(orders[oid].w for oid in s.bag if oid in orders) for s in shippers]
        k_used = [len(s.bag) for s in shippers]
        avail_w = [max(0.0, cap_w[i] - w_used[i]) for i in range(self.C)]
        avail_k = [max(0,   cap_k[i] - k_used[i]) for i in range(self.C)]

        node_list = [
            VRPNode(idx, (o.sx, o.sy), o.id, o.w)
            for idx, o in enumerate(candidates)
        ]

        # Time budget: tỷ lệ số đơn, giới hạn 0.6s
        budget = min(0.6, max(0.10, 0.03 * len(node_list)))

        routes_nodes = self.pipeline.solve(depots, node_list, avail_w, avail_k, budget)

        for v, vnodes in enumerate(routes_nodes):
            plan    = self.plans[v]
            shipper = shippers[v]
            plan.clear()

            carrying = set(shipper.bag)
            # Giữ waypoint deliver cho đơn đang mang
            for oid in list(carrying):
                if oid in orders:
                    o = orders[oid]
                    plan.add((o.ex, o.ey), "deliver", oid)
                    plan.assigned.append(oid)

            # Thêm pickup → deliver theo route VRP
            used_w = w_used[v]
            used_k = k_used[v]
            for vn in vnodes:
                o = orders.get(vn.order_id)
                if o is None or o.picked or o.delivered or vn.order_id in carrying:
                    continue
                if used_w + o.w > cap_w[v] or used_k + 1 > cap_k[v]:
                    continue
                plan.add((o.sx, o.sy), "pickup",  o.id)
                plan.add((o.ex, o.ey), "deliver", o.id)
                plan.assigned.append(o.id)
                self.planned.add(o.id)
                used_w += o.w
                used_k += 1

    # ─────────────────────────────────────────────────────────────────────────

    def _build_actions(
        self,
        shippers: List[Shipper],
        orders  : Dict[int, Order],
    ) -> Dict[int, Tuple[str, int]]:
        return {s.id: self._action_for(s, self.plans[s.id], orders) for s in shippers}

    def _action_for(
        self,
        shipper: Shipper,
        plan   : ShipperPlan,
        orders : Dict[int, Order],
    ) -> Tuple[str, int]:
        pos = (shipper.r, shipper.c)

        # Giao tại chỗ
        for oid in shipper.bag:
            if oid in orders and (orders[oid].ex, orders[oid].ey) == pos:
                return ("S", 2)

        # Theo path buffer
        if plan.path_buf:
            return (self._dir(pos, plan.path_buf.pop(0)), 0)

        # Lấy waypoint tiếp theo
        while plan.waypoints:
            tgt, kind, oid = plan.waypoints[0]

            if oid is not None:
                o = orders.get(oid)
                if o is None:
                    plan.waypoints.popleft(); continue
                if kind == "pickup"  and (o.picked or o.delivered):
                    plan.waypoints.popleft(); continue
                if kind == "deliver" and o.delivered:
                    plan.waypoints.popleft(); continue

            if pos == tgt:
                plan.waypoints.popleft()
                return ("S", 1) if kind == "pickup" else ("S", 2)

            path = self.g.path(pos, tgt)
            if not path:
                plan.waypoints.popleft(); continue
            plan.path_buf = path[1:]
            return (self._dir(pos, path[0]), 0)

        return ("S", 0)

    @staticmethod
    def _dir(src: Tuple[int, int], dst: Tuple[int, int]) -> str:
        dr, dc = dst[0] - src[0], dst[1] - src[1]
        if dr == -1: return "U"
        if dr ==  1: return "D"
        if dc == -1: return "L"
        if dc ==  1: return "R"
        return "S"
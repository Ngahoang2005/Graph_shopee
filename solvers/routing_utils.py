from __future__ import annotations
from typing import List, Tuple
from solvers.solver import bfs_path

def compute_path(
    grid: List[List[int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> List[str]:
    """
    Wrapper routing abstraction.
    Hiện tại dùng BFS (sau có thể đổi thành A*, CBS, OR-Tools generated route, etc.)
    """
    return bfs_path(grid, start, goal)

def next_position(r: int, c: int, action: str):
    if action == "U":
        return r-1, c
    if action == "D":
        return r+1, c
    if action == "L":
        return r, c-1
    if action == "R":
        return r, c+1
    
    return r, c

def path_distance(path: List[str]) -> int:
    return len(path)
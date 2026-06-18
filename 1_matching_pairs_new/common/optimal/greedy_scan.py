"""
Greedy Scan — optimal baseline for single-player Matching Pairs.

Strategy: perfect memory + greedy scan
- flip cells left-to-right, top-to-bottom
- remember every flipped card perfectly
- if a card's matching position is known, prioritize matching it
- otherwise keep exploring the next un-flipped card

Purpose: a lower bound on model performance (theoretical optimum); used to
compute resp_times / optimal to measure model efficiency.

Ported directly from legacy_code/1_matching_pairs/matching_game/optimal_single/greedy_scan.py.
"""

from typing import List, Set, Tuple, Dict


def compute_optimal_resp_times(board: List[List[str]]) -> int:
    """Compute the optimal strategy's resp_times.

    Strategy (perfect memory + greedy):
    - maintain a pending_matches queue: cards with a known pair position not yet matched
    - if pending_matches is non-empty: Step1 flip one of them, Step2 flip its pair -> match (2 resp)
    - otherwise: Step1 flip the next un-flipped new card
      - if its pair is known: Step2 flip the pair -> match (2 resp)
      - otherwise: Step2 flip the next un-flipped new card -> learn two (2 resp), possibly a lucky match
    """
    rows = len(board)
    cols = len(board[0])
    cells = [(r, c, board[r][c]) for r in range(rows) for c in range(cols)]

    known: Dict[str, Tuple[int, int]] = {}
    matched: Set[Tuple[int, int]] = set()
    pending_matches: List[str] = []
    resp_times = 0
    scan_idx = 0

    def next_unvisited():
        nonlocal scan_idx
        while scan_idx < len(cells) and (cells[scan_idx][0], cells[scan_idx][1]) in matched:
            scan_idx += 1
        if scan_idx < len(cells):
            return scan_idx
        return None

    while len(matched) < len(cells):
        # Priority 1: match a pending pair
        if pending_matches:
            face = pending_matches.pop(0)
            other_positions = [(r, c, f) for r, c, f in cells if f == face and (r, c) not in matched]
            if len(other_positions) >= 2:
                (r1, c1, _), (r2, c2, _) = other_positions[0], other_positions[1]
                resp_times += 2
                matched.add((r1, c1))
                matched.add((r2, c2))
                if face in known:
                    del known[face]
                continue

        # Priority 2: explore new cells
        idx1 = next_unvisited()
        if idx1 is None:
            break

        r1, c1, face1 = cells[idx1]
        resp_times += 1
        scan_idx = idx1 + 1

        if face1 in known and known[face1] != (r1, c1) and known[face1] not in matched:
            r2, c2 = known[face1]
            resp_times += 1
            matched.add((r1, c1))
            matched.add((r2, c2))
            del known[face1]
        else:
            known[face1] = (r1, c1)

            idx2 = next_unvisited()
            if idx2 is None:
                break

            r2, c2, face2 = cells[idx2]
            resp_times += 1
            scan_idx = idx2 + 1

            if face1 == face2:
                matched.add((r1, c1))
                matched.add((r2, c2))
                if face1 in known:
                    del known[face1]
            else:
                if face2 in known and known[face2] != (r2, c2) and known[face2] not in matched:
                    pending_matches.append(face2)
                known[face2] = (r2, c2)

    return resp_times

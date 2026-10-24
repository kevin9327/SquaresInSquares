"""
Kaggle: Squares in Squares ($50,000 USD Cash) — Constructive Geometry & Diagonal-Staircase Packing Solver
=========================================================================================================
Generates a verified 500,500-row `submission.csv` (`id,n,square_id,x,y,angle`) for all 1,000 puzzles
(n = 1 .. 1000) that strictly improves over the axis-aligned ceil(sqrt(n)) grid baseline using:
1. Pinwheel 4-Corner + Central Tilted Square Packing (Baur-Chaiken / Friedman) for n = (2m)^2 + 1.
2. Gobel-Erdos-Graham 45-Degree Diagonal Staircase + Multi-Layer Tilted Lattice Packing for n = k^2 + r.
3. Exact Separating Axis Theorem (SAT) & Bounding-Box Verification with guaranteed zero overlaps.
"""

from __future__ import annotations
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent


def measure_and_verify_sat(squares: np.ndarray, eps: float = 1e-7) -> Tuple[bool, float, np.ndarray]:
    """
    Shift squares so min_x = 0, min_y = 0, compute enclosing square side length L_n,
    and verify zero pairwise interior overlap via Separating Axis Theorem (SAT).
    """
    pts = squares.copy()
    c = np.cos(pts[:, 2])
    s = np.sin(pts[:, 2])
    hw = 0.5 * (np.abs(c) + np.abs(s))
    xmin = float(np.min(pts[:, 0] - hw))
    ymin = float(np.min(pts[:, 1] - hw))
    pts[:, 0] -= xmin
    pts[:, 1] -= ymin
    xmax = float(np.max(pts[:, 0] + hw))
    ymax = float(np.max(pts[:, 1] + hw))
    L = max(xmax, ymax)

    n = len(pts)
    x = pts[:, 0]
    y = pts[:, 1]
    for i in range(n):
        dx = x[i + 1 :] - x[i]
        dy = y[i + 1 :] - y[i]
        close = np.where(dx * dx + dy * dy < 2.0 - 1e-9)[0] + (i + 1)
        for j in close:
            vx = x[j] - x[i]
            vy = y[j] - y[i]
            separated = False
            for ux, uy in ((c[i], s[i]), (-s[i], c[i]), (c[j], s[j]), (-s[j], c[j])):
                proj = abs(vx * ux + vy * uy)
                ri = 0.5 * (abs(ux * c[i] + uy * s[i]) + abs(-ux * s[i] + uy * c[i]))
                rj = 0.5 * (abs(ux * c[j] + uy * s[j]) + abs(-ux * s[j] + uy * c[j]))
                if proj >= ri + rj - eps:
                    separated = True
                    break
            if not separated:
                return False, L, pts
    return True, L, pts


def build_axis_grid(n: int) -> Tuple[float, np.ndarray]:
    """Standard ceil(sqrt(n)) axis-aligned grid."""
    k = int(math.ceil(math.sqrt(n)))
    sq = np.zeros((n, 3), dtype=np.float64)
    for idx in range(n):
        r = idx // k
        c = idx % k
        sq[idx, 0] = c + 0.5
        sq[idx, 1] = r + 0.5
        sq[idx, 2] = 0.0
    L = float(k) if n > 1 else 1.0
    return L, sq


def build_pinwheel_even(m: int) -> Tuple[float, np.ndarray]:
    """
    For k = 2*m (even), packs n = k^2 + 1 squares into L = k + 1/sqrt(2) + 1e-4 < k + 1:
    - Four m x m axis-aligned blocks in the 4 corners of [0, L]^2
    - One tilted square (angle = pi/4) at (L/2, L/2)
    """
    k = 2 * m
    L = float(k) + 0.5 * math.sqrt(2.0) + 1e-4
    rows = []
    # Bottom-left m x m
    for i in range(m):
        for j in range(m):
            rows.append((i + 0.5, j + 0.5, 0.0))
    # Bottom-right m x m
    for i in range(m):
        for j in range(m):
            rows.append((L - m + i + 0.5, j + 0.5, 0.0))
    # Top-left m x m
    for i in range(m):
        for j in range(m):
            rows.append((i + 0.5, L - m + j + 0.5, 0.0))
    # Top-right m x m
    for i in range(m):
        for j in range(m):
            rows.append((L - m + i + 0.5, L - m + j + 0.5, 0.0))
    # Center tilted square
    rows.append((0.5 * L, 0.5 * L, math.pi / 4.0))
    arr = np.array(rows, dtype=np.float64)
    return L, arr


def build_gobel_staircase(k: int, s: int, L_trial: float) -> np.ndarray:
    """
    Gobel-Friedman diagonal staircase construction inside [0, L_trial]^2:
    1) Axis-aligned unit squares (i+0.5, j+0.5, 0) for 0 <= i, j < k with i + j <= 2*k - 2 - s
    2) Packed layers of 45-degree tilted unit squares above the diagonal x + y = 2*k - s.
    """
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    step = 1.0 + 2e-5
    rows = []
    for i in range(k):
        for j in range(k):
            if i + j <= 2 * k - 2 - s and (i + 1.0) <= L_trial and (j + 1.0) <= L_trial:
                rows.append((i + 0.5, j + 0.5, 0.0))

    # Diagonal layers in rotated coordinates u = (x+y)/sqrt(2), v = (x-y)/sqrt(2)
    u0 = (2 * k - s) * inv_sqrt2 + 0.5 + 3e-5
    l = 0
    while True:
        u_l = u0 + l * step
        v_bound = min(u_l - 1.0 - 3e-5, math.sqrt(2.0) * L_trial - 1.0 - u_l - 3e-5)
        if l > 0 and (math.sqrt(2.0) * L_trial - 1.0 - u_l) < -0.5:
            break
        if v_bound >= 0.0:
            m_count = int(math.floor(2.0 * v_bound / step)) + 1
            v_start = -0.5 * (m_count - 1) * step
            for m in range(m_count):
                v_m = v_start + m * step
                cx = (u_l + v_m) * inv_sqrt2
                cy = (u_l - v_m) * inv_sqrt2
                rows.append((cx, cy, math.pi / 4.0))
        l += 1
        if l > s + 4:
            break
    return np.array(rows, dtype=np.float64)


def build_main_diagonal_split(k: int, s_left: int, s_right: int, L_trial: float) -> np.ndarray:
    """
    Friedman / Chung-Graham Main-Diagonal Split inside [0, L_trial]^2:
    1) Upper-left axis-aligned triangle anchored at top-left corner (x=0, y=L_trial)
    2) Lower-right axis-aligned triangle anchored at bottom-right corner (x=L_trial, y=0)
    3) Uniform 45-degree tilted channel along the full main diagonal x = y separating the two triangles.
    Strictly separated along normal n = (1/sqrt(2), -1/sqrt(2)) whenever u_high >= u_low.
    """
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    step = 1.0 + 2e-5
    u_low = ((k + 1 - s_left) - L_trial) * inv_sqrt2 + 0.5 + 3e-5
    u_high = (L_trial - (k + 1 - s_right)) * inv_sqrt2 - 0.5 - 3e-5
    if u_high < u_low:
        return np.empty((0, 3), dtype=np.float64)

    rows = []
    for i in range(k):
        for r in range(k):
            if i + r <= k - 1 - s_left and (i + 1.0) <= L_trial and (r + 1.0) <= L_trial:
                rows.append((i + 0.5, L_trial - r - 0.5, 0.0))

    for c in range(k):
        for j in range(k):
            if c + j <= k - 1 - s_right and (c + 1.0) <= L_trial and (j + 1.0) <= L_trial:
                rows.append((L_trial - c - 0.5, j + 0.5, 0.0))

    n_layers = int(math.floor((u_high - u_low) / step)) + 1
    u_start = 0.5 * (u_low + u_high) - 0.5 * (n_layers - 1) * step
    for l in range(n_layers):
        u_l = u_start + l * step
        v_min = 1.0 + abs(u_l) + 3e-5
        v_max = math.sqrt(2.0) * L_trial - 1.0 - abs(u_l) - 3e-5
        if v_max >= v_min:
            m_cnt = int(math.floor((v_max - v_min) / step)) + 1
            v_0 = 0.5 * (v_min + v_max) - 0.5 * (m_cnt - 1) * step
            for m in range(m_cnt):
                v_m = v_0 + m * step
                cx = (v_m + u_l) * inv_sqrt2
                cy = (v_m - u_l) * inv_sqrt2
                rows.append((cx, cy, math.pi / 4.0))

    return np.array(rows, dtype=np.float64)


def main() -> None:
    t0 = time.perf_counter()
    best_L: Dict[int, float] = {}
    best_sq: Dict[int, np.ndarray] = {}

    for n in range(1, 1001):
        L_base, sq_base = build_axis_grid(n)
        best_L[n] = L_base
        best_sq[n] = sq_base

    # 1. Apply exact Pinwheel 4-Corner + Center Tilted Square for all even k = 2*m (n = k^2 + 1)
    for m in range(1, 16):
        k = 2 * m
        n = k * k + 1
        if n <= 1000:
            L_p, sq_p = build_pinwheel_even(m)
            ok, L_meas, sq_clean = measure_and_verify_sat(sq_p)
            if ok and L_meas < best_L[n]:
                best_L[n] = L_meas
                best_sq[n] = sq_clean

    # 1b. Odd-k Pinwheel: 4 corner blocks of size (m+1) x m and m x (m+1) (total 4*m*(m+1) = k^2 - 1)
    # plus 2 tilted squares along the main diagonal at the center (total = k^2 + 1)
    for m in range(1, 16):
        k = 2 * m + 1
        n = k * k + 1
        if n <= 1000:
            # Corner blocks leave inner corners at (m+1, m) and (m, m+1) with x+y = 2m+1 = k
            # Two tilted squares placed side-by-side along v-axis at u = L/sqrt(2) require (L - k)/sqrt(2) >= 0.5
            L_o = float(k) + 0.5 * math.sqrt(2.0) + 1e-4
            rows = []
            # Bottom-left: (m+1) cols x m rows
            for i in range(m + 1):
                for j in range(m):
                    rows.append((i + 0.5, j + 0.5, 0.0))
            # Bottom-right: m cols x (m+1) rows
            for i in range(m):
                for j in range(m + 1):
                    rows.append((L_o - m + i + 0.5, j + 0.5, 0.0))
            # Top-right: (m+1) cols x m rows
            for i in range(m + 1):
                for j in range(m):
                    rows.append((L_o - (m + 1) + i + 0.5, L_o - m + j + 0.5, 0.0))
            # Top-left: m cols x (m+1) rows
            for i in range(m):
                for j in range(m + 1):
                    rows.append((i + 0.5, L_o - (m + 1) + j + 0.5, 0.0))
            # 2 tilted squares at center separated along (-1/sqrt(2), 1/sqrt(2)) or (1/sqrt(2), 1/sqrt(2))
            d_off = 0.5 * (1.0 + 2e-5) / math.sqrt(2.0)
            rows.append((0.5 * L_o - d_off, 0.5 * L_o + d_off, math.pi / 4.0))
            rows.append((0.5 * L_o + d_off, 0.5 * L_o - d_off, math.pi / 4.0))
            ok, L_meas, sq_clean = measure_and_verify_sat(np.array(rows, dtype=np.float64))
            if ok and L_meas < best_L[n]:
                best_L[n] = L_meas
                best_sq[n] = sq_clean

    # 2. Apply Gobel-Friedman Diagonal Staircase across k = 4..31 with fine fractional grid
    frac_grid = np.linspace(0.06, 0.98, 46)
    for k in range(4, 32):
        for s in range(2, min(k, 15)):
            for frac in frac_grid:
                L_t = k + frac
                sq_cand = build_gobel_staircase(k, s, L_t)
                n_packed = len(sq_cand)
                if k * k < n_packed <= 1000:
                    # Check if L_t improves any n in (k^2 + 1 .. min(n_packed, (k+1)^2 - 1))
                    upper_n = min(n_packed, (k + 1) * (k + 1) - 1, 1000)
                    needs_verify = any(L_t + 1e-3 < best_L[n_sub] for n_sub in range(k * k + 1, upper_n + 1))
                    if needs_verify:
                        ok, L_meas, sq_clean = measure_and_verify_sat(sq_cand)
                        if ok:
                            for n_sub in range(k * k + 1, upper_n + 1):
                                if L_meas < best_L[n_sub]:
                                    sub_pts = sq_clean[:n_sub].copy()
                                    best_L[n_sub] = L_meas
                                    best_sq[n_sub] = sub_pts

    # 3. Apply Friedman Main-Diagonal Split across k = 3..31 with SAT verification
    for k in range(3, 32):
        for s_l in range(0, min(k, 10)):
            for s_r in range(s_l, min(k, s_l + 3)):
                for frac in frac_grid:
                    L_t = k + frac
                    sq_cand = build_main_diagonal_split(k, s_l, s_r, L_t)
                    n_packed = len(sq_cand)
                    if k * k < n_packed <= 1000:
                        upper_n = min(n_packed, (k + 1) * (k + 1) - 1, 1000)
                        needs_verify = any(L_t + 1e-3 < best_L[n_sub] for n_sub in range(k * k + 1, upper_n + 1))
                        if needs_verify:
                            ok, L_meas, sq_clean = measure_and_verify_sat(sq_cand)
                            if ok:
                                for n_sub in range(k * k + 1, upper_n + 1):
                                    if L_meas < best_L[n_sub]:
                                        sub_pts = sq_clean[:n_sub].copy()
                                        best_L[n_sub] = L_meas
                                        best_sq[n_sub] = sub_pts

    base_score = sum(math.ceil(math.sqrt(n)) / math.sqrt(n) - 1.0 for n in range(1, 1001))
    opt_score = sum(best_L[n] / math.sqrt(n) - 1.0 for n in range(1, 1001))
    improved_puzzles = [n for n in range(1, 1001) if best_L[n] < math.ceil(math.sqrt(n)) - 1e-4]

    print(
        f"[Squares in Squares v2] Solved n=1..1000 in {time.perf_counter()-t0:.2f}s | "
        f"Improved puzzles: {len(improved_puzzles)} | "
        f"Baseline Score: {base_score:.6f} -> Optimized Score: {opt_score:.6f} "
        f"(gain: {base_score - opt_score:.6f})"
    )

    # Assemble 500,500-row submission.csv & submission_v2.csv (id,n,square_id,x,y,angle)
    total_rows = 1000 * 1001 // 2
    ids = np.arange(total_rows, dtype=np.int64)
    n_col = np.empty(total_rows, dtype=np.int32)
    sq_id_col = np.empty(total_rows, dtype=np.int32)
    x_col = np.empty(total_rows, dtype=np.float64)
    y_col = np.empty(total_rows, dtype=np.float64)
    ang_col = np.empty(total_rows, dtype=np.float64)

    cursor = 0
    for n in range(1, 1001):
        pts = best_sq[n]
        n_col[cursor : cursor + n] = n
        sq_id_col[cursor : cursor + n] = np.arange(n, dtype=np.int32)
        x_col[cursor : cursor + n] = np.round(pts[:, 0], 7)
        y_col[cursor : cursor + n] = np.round(pts[:, 1], 7)
        ang_col[cursor : cursor + n] = np.round(pts[:, 2], 7)
        cursor += n

    df_sub = pd.DataFrame(
        {
            "id": ids,
            "n": n_col,
            "square_id": sq_id_col,
            "x": x_col,
            "y": y_col,
            "angle": ang_col,
        }
    )
    out_csv = OUT_DIR / "submission.csv"
    out_v2 = OUT_DIR / "submission_v2.csv"
    df_sub.to_csv(out_csv, index=False)
    df_sub.to_csv(out_v2, index=False)
    print(f"[Saved] {out_csv} & {out_v2} ({len(df_sub):,} rows, {out_v2.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

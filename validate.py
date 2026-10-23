"""Validate the bigraded action-matrix solver on an instance without a full reference set."""
import sys, pickle, time
import numpy as np
from solve import random_instance, solve
from bigraded_action2 import solve_instance
from monodromy import F_J_Ft, cayley_from_R

seed = int(sys.argv[1])
A, B, C = random_instance(seed)
P0 = (A.astype(complex), B.astype(complex), C.astype(complex))
t0 = time.time()
pts, lam = solve_instance(A, B, C, verbose=False)
t_solve = time.time() - t0
res = np.array([np.linalg.norm(F_J_Ft(s, P0, P0)[0]) for s in pts])
pol = []
for s in pts:
    v = s.copy()
    for _ in range(6):
        F, J, _ = F_J_Ft(v, P0, P0)
        v = v - np.linalg.solve(J, F)
    pol.append(v)
pol = np.array(pol)
res_pol = np.array([np.linalg.norm(F_J_Ft(s, P0, P0)[0]) for s in pol])
uniq = []
for s in pol:
    if not any(np.linalg.norm(s - u) < 1e-6 for u in uniq):
        uniq.append(s)
n_real_eig = int(np.sum(np.abs(lam.imag) < 1e-6))
real_pts = [s.real for s in pol if np.linalg.norm(s.imag) < 1e-6]
# reference real solutions from random restarts
ref_real = solve(A, B, C, n_starts=2500, seed=seed)
ref_v = [np.concatenate([cayley_from_R(R2), cayley_from_R(R3)]) for R2, R3 in ref_real]
matched = sum(any(np.linalg.norm(r - p) < 1e-6 for p in real_pts) for r in ref_v)
print(f"seed {seed}: solver time {t_solve:.1f}s; eigenvalues {len(lam)}, real {n_real_eig}")
print(f"  raw residuals: median {np.median(res):.1e}, max {res.max():.1e}; after Newton: max {res_pol.max():.1e}")
print(f"  distinct solutions after polish: {len(uniq)}; real solutions: {len(real_pts)}")
print(f"  random-restart real solutions: {len(ref_real)}, of which found by action matrix: {matched}")

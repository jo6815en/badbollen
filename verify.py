"""Compare a solution set against the monodromy reference set (sols_seed<k>.pkl)."""
import pickle
import numpy as np

def verify(sols, A, B, C, ref_path='sols_seed0.pkl'):
    from monodromy import F_J_Ft
    P0 = (A.astype(complex), B.astype(complex), C.astype(complex))
    ref = np.array(pickle.load(open(ref_path, 'rb'))['sols'])
    res = np.array([np.linalg.norm(F_J_Ft(s, P0, P0)[0]) for s in sols])
    dist = np.array([np.min(np.linalg.norm(ref - s, axis=1)) for s in sols])
    print(f"residuals: median {np.median(res):.1e}, max {res.max():.1e}")
    for thr in (1e-2, 1e-4, 1e-6, 1e-8):
        print(f"  matched to reference within {thr:.0e}: {np.sum(dist < thr)}/600")
    # polish with Newton and count distinct
    pol = []
    for s in sols:
        v = s.copy()
        for _ in range(6):
            F, J, _ = F_J_Ft(v, P0, P0)
            try:
                v = v - np.linalg.solve(J, F)
            except np.linalg.LinAlgError:
                break
        pol.append(v)
    pol = np.array(pol)
    dist2 = np.array([np.min(np.linalg.norm(ref - s, axis=1)) for s in pol])
    uniq = []
    for s in pol[dist2 < 1e-8]:
        if not any(np.linalg.norm(s - u) < 1e-6 for u in uniq):
            uniq.append(s)
    print(f"after Newton polish: {np.sum(dist2 < 1e-8)} converge to reference solutions, {len(uniq)} distinct")

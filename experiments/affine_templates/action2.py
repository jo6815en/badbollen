"""Stage 2: action matrix from numerically saturated ideal elements (degree <= 7).

N = list of polys spanning (hopefully) I_sat_{<=7}. Basis B = 600 standard monomials (deg <= 6).
Non-basis monomials of degree <= 7: 1716 - 600 = 1116. Need dim N = 1116 and the non-basis block invertible.
"""
import sys, pickle, itertools, time
import numpy as np
from numsat6 import monos_upto, to_matrix, input_polys, NV
from solve import random_instance

def parse_kbase(path):
    names = 'bcdfgh'
    mons = []
    for tok in open(path).read().strip().split(','):
        if tok == '1':
            mons.append((0,)*NV); continue
        e = [0]*NV; i = 0
        while i < len(tok):
            v = names.index(tok[i]); i += 1
            k = 0
            while i < len(tok) and tok[i].isdigit():
                k = 10*k + int(tok[i]); i += 1
            e[v] += k if k else 1
        mons.append(tuple(e))
    return mons

def action_matrix(N, basis, act=0, verbose=True):
    t0 = time.time()
    Bset = set(basis)
    cols_all = monos_upto(7)
    nonb = [m for m in cols_all if m not in Bset]
    order = nonb + list(basis)
    M = to_matrix(N, order)
    nN, nB = len(nonb), len(basis)
    if verbose:
        print(f"N: {M.shape}, non-basis {nN}, basis {nB}", flush=True)
    # reduce: want X with [I | X] = rows spanning N (after elimination on the non-basis block)
    Mn, Mb = M[:, :nN], M[:, nN:]
    U, s, Vt = np.linalg.svd(Mn, full_matrices=False)
    r = int(np.sum(s > 1e-10 * s[0]))
    if verbose:
        print(f"rank of non-basis block: {r} / {nN}; sv range {s[0]:.2e} .. {s[r-1]:.2e}, next {s[r] if r < len(s) else 0:.2e}", flush=True)
    if r < nN:
        return None
    # X = Mn^+ Mb  (normal form of non-basis monomial i = -X[i] . B)
    X = np.linalg.lstsq(Mn, Mb, rcond=None)[0]
    # consistency check: residual of the lstsq (should be ~0 if N's rows are all in the ideal & B independent)
    res = np.linalg.norm(Mn @ X - Mb) / np.linalg.norm(Mb)
    nf = {m: -X[i] for i, m in enumerate(nonb)}
    bidx = {m: i for i, m in enumerate(basis)}
    T = np.zeros((nB, nB))
    for j, m in enumerate(basis):
        mm = tuple(m[i] + (1 if i == act else 0) for i in range(NV))
        if mm in bidx:
            T[j, bidx[mm]] = 1.0
        else:
            T[j, :] = nf[mm]
    w, V = np.linalg.eig(T.T)
    one = bidx[(0,)*NV]
    var_rows = [bidx[tuple(1 if i == v else 0 for i in range(NV))] for v in range(NV)]
    sols = (V[var_rows, :] / V[one, :]).T
    if verbose:
        print(f"lstsq residual {res:.1e}; action matrix built in {time.time()-t0:.1f}s", flush=True)
    return sols

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

if __name__ == '__main__':
    satfile = sys.argv[1]
    N = pickle.load(open(satfile, 'rb'))
    basis = parse_kbase('kbase6_gfp.txt')
    assert len(basis) == 600
    A, B, C = random_instance(0)
    sols = action_matrix(N, basis)
    if sols is not None:
        verify(sols, A, B, C)

"""Numerical saturation in Cayley variables (b,c,d,f,g,h), s2 = 1+b^2+c^2+d^2.

N_k(D) = { g : deg g <= D, s2^k g in span{ m E_c : deg(m E_c) <= D+2k } }.
"""
import itertools, sys, time, pickle
import numpy as np
import sympy as sp
from solve import random_instance

VARS = sp.symbols('b c d f g h')
NV = 6

def cayley(x, y, z):
    return sp.Matrix([[1+x*x-y*y-z*z, 2*(x*y-z), 2*(x*z+y)],
                      [2*(x*y+z), 1-x*x+y*y-z*z, 2*(y*z-x)],
                      [2*(x*z-y), 2*(y*z+x), 1-x*x-y*y+z*z]])

def input_polys(A, B, C):
    b, c, d, f, g, h = VARS
    R2, R3 = cayley(b, c, d), cayley(f, g, h)
    out = []
    for k in range(6):
        M = sp.Matrix.hstack(sp.Matrix(A[k]), R2 * sp.Matrix(B[k]), R3 * sp.Matrix(C[k]))
        out.append({e_: float(v) for e_, v in sp.Poly(sp.expand(M.det()), *VARS).terms()})
    return out

def s2_pow(k):
    b, c, d = VARS[:3]
    return {e_: float(v) for e_, v in sp.Poly((1+b*b+c*c+d*d)**k, *VARS).terms()}

def monos_upto(D):
    out = []
    for deg in range(D+1):
        for comb in itertools.combinations_with_replacement(range(NV), deg):
            e_ = [0]*NV
            for v in comb: e_[v] += 1
            out.append(tuple(e_))
    return out

def mul(p, q):
    out = {}
    for e1, c1 in p.items():
        for e2, c2 in q.items():
            e_ = tuple(x+y for x, y in zip(e1, e2))
            out[e_] = out.get(e_, 0.0) + c1*c2
    return out

def mono_mul(m, p):
    return {tuple(x+y for x, y in zip(m, e_)): c for e_, c in p.items()}

def to_matrix(polys, cols):
    idx = {m: i for i, m in enumerate(cols)}
    M = np.zeros((len(polys), len(cols)))
    for r, p in enumerate(polys):
        for e_, c in p.items():
            M[r, idx[e_]] = c
    return M

def rowbasis(M, rtol=1e-11):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    r = int(np.sum(s > rtol * s[0]))
    return Vt[:r], s

def saturate(E, D, k, tol=1e-7, verbose=True):
    """Return (list of polys g with deg<=D and s2^k g in I_raw_{<=D+2k}, principal cosines)."""
    t0 = time.time()
    cols = monos_upto(D + 2*k)
    rows_I = [mono_mul(m, Ec) for Ec in E for m in monos_upto(D + 2*k - 4)]
    s2k = s2_pow(k)
    target = monos_upto(D)
    rows_S = [mul({m: 1.0}, s2k) for m in target]
    A = to_matrix(rows_I, cols)
    Bm = to_matrix(rows_S, cols)
    QA, sA = rowbasis(A)
    QB, sB = rowbasis(Bm)
    U, s, Vt = np.linalg.svd(QA @ QB.T)
    n_int = int(np.sum(s > 1 - tol))
    gs = []
    Y = np.linalg.lstsq(Bm.T, (Vt[:n_int] @ QB).T, rcond=None)[0]   # columns: coefficient vectors y
    for l in range(n_int):
        gs.append({m: Y[t, l] for t, m in enumerate(target) if abs(Y[t, l]) > 1e-13})
    if verbose:
        print(f"D={D} k={k}: A {A.shape} rank {QA.shape[0]}, S dim {QB.shape[0]}, intersection {n_int} "
              f"(cos gap: {s[n_int-1]:.8f} -> {s[n_int] if n_int < len(s) else 0:.6f}), {time.time()-t0:.1f}s", flush=True)
    return gs, s

if __name__ == '__main__':
    A, B, C = random_instance(0)
    E = input_polys(A, B, C)
    for D, k in [(6, 1), (6, 2), (7, 1), (7, 2)]:
        gs, s = saturate(E, D, k)
        with open(f'sat_D{D}_k{k}.pkl', 'wb') as fh:
            pickle.dump(gs, fh)

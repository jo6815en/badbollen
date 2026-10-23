"""Numerical saturation of the bihomogeneous ideal by s2 = q2.q2.

Variables q2=(a,b,c,d), q3=(e,f,g,h). Polynomials are dicts {exponent tuple (8,): coeff}.
For target bidegree (i,j) and exponent k, find all g of bidegree (i,j) with s2^k * g in the
raw ideal piece I_{(i+2k, j)} = span{ m * E_c : m of bidegree (i+2k-2, j-2) }.
"""
import itertools, sys, time
import numpy as np
import sympy as sp
from solve import random_instance

VARS = sp.symbols('a b c d e f g h')

def rotq(w, x, y, z):
    return sp.Matrix([[w*w+x*x-y*y-z*z, 2*(x*y-w*z), 2*(x*z+w*y)],
                      [2*(x*y+w*z), w*w-x*x+y*y-z*z, 2*(y*z-w*x)],
                      [2*(x*z-w*y), 2*(y*z+w*x), w*w-x*x-y*y+z*z]])

def input_polys(A, B, C):
    a, b, c, d, e, f, g, h = VARS
    R2, R3 = rotq(a, b, c, d), rotq(e, f, g, h)
    out = []
    for k in range(6):
        M = sp.Matrix.hstack(sp.Matrix(A[k]), R2 * sp.Matrix(B[k]), R3 * sp.Matrix(C[k]))
        out.append({e_: float(v) for e_, v in sp.Poly(sp.expand(M.det()), *VARS).terms()})
    return out

def s2_pow(k):
    a, b, c, d = VARS[:4]
    return {e_: float(v) for e_, v in sp.Poly((a*a+b*b+c*c+d*d)**k, *VARS).terms()}

def monos(i, j):
    """All monomials of bidegree (i, j) as exponent tuples."""
    def part(deg, n):
        for comb in itertools.combinations_with_replacement(range(n), deg):
            e_ = [0]*n
            for v in comb: e_[v] += 1
            yield tuple(e_)
    return [p + q for p in part(i, 4) for q in part(j, 4)]

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

def saturate_piece(E, i, j, k, tol=1e-8, verbose=True):
    """Return list of polys g (bidegree (i,j)) with s2^k g in I_{(i+2k, j)}; also the singular values."""
    t0 = time.time()
    cols = monos(i + 2*k, j)
    mult = monos(i + 2*k - 2, j - 2) if j >= 2 and i + 2*k >= 2 else []
    rows_I = [mono_mul(m, Ec) for m in mult for Ec in E]
    s2k = s2_pow(k)
    target = monos(i, j)
    rows_S = [mul({m: 1.0}, s2k) for m in target]
    A = to_matrix(rows_I, cols)
    Bm = to_matrix(rows_S, cols)
    # orthonormal bases of the two row spaces
    def rowbasis(M):
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        r = int(np.sum(s > 1e-10 * s[0]))
        return Vt[:r]
    QA, QB = rowbasis(A), rowbasis(Bm)
    # principal angles: singular values of QA QB^T near 1 -> intersection
    U, s, Vt = np.linalg.svd(QA @ QB.T)
    n_int = int(np.sum(s > 1 - tol))
    # intersection vectors in row space of B: (Vt[:n_int] @ QB); express as g by solving Bm^T y = v
    gs = []
    for l in range(n_int):
        v = Vt[l] @ QB
        y, *_ = np.linalg.lstsq(Bm.T, v, rcond=None)
        gs.append({m: y[t] for t, m in enumerate(target) if abs(y[t]) > 1e-14})
    if verbose:
        print(f"bidegree ({i},{j}) k={k}: I-piece rank {QA.shape[0]} (rows {A.shape[0]}), S dim {QB.shape[0]}, "
              f"ambient {len(cols)}, intersection dim {n_int}, gap sv[{n_int-1 if n_int else 0}]={s[max(n_int-1,0)]:.6f} "
              f"next={s[n_int] if n_int < len(s) else 0:.6f}, {time.time()-t0:.1f}s", flush=True)
    return gs, s

if __name__ == '__main__':
    A, B, C = random_instance(0)
    E = input_polys(A, B, C)
    for (i, j, k) in [(2, 2, 1), (4, 2, 1), (2, 4, 1), (4, 4, 1), (4, 2, 2), (2, 4, 2), (4, 4, 2), (6, 2, 1), (2, 6, 1), (6, 4, 1), (4, 6, 1)]:
        try:
            saturate_piece(E, i, j, k)
        except MemoryError:
            print(f"({i},{j},{k}) too big")

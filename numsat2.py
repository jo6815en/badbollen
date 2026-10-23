"""Cleaner numerical quotient (I : h) at a bidegree, via nullspace of the Macaulay block.

Z = orthonormal basis of the orthogonal complement of rowspace(A)  (A = Macaulay rows at bidegree (i,j)+deg h)
g in (I:h)_{(i,j)}  <=>  (h g) . Z = 0  <=>  y in null(Bm Z),  Bm rows = h * monomials of bidegree (i,j).
The singular values of Bm Z show the gap directly.
"""
import sys, time, pickle
import numpy as np
import scipy.linalg as sla
import sympy as sp
from numsat import input_polys, monos, mul, mono_mul, to_matrix, VARS, s2_pow

def poly_of(expr):
    return {e_: float(v) for e_, v in sp.Poly(sp.expand(expr), *VARS).terms()}

def nullspace_rows(A, rtol=1e-10):
    """Orthonormal basis (columns) of the nullspace of A, via pivoted QR of A^T."""
    Q, R, piv = sla.qr(A.T, mode='full', pivoting=True)
    diag = np.abs(np.diag(R))
    r = int(np.sum(diag > rtol * diag[0]))
    return Q[:, r:], r, diag

def quotient_piece(E, h, hdeg, i, j, verbose=True, show=None):
    """(I_raw : h)_{(i,j)} where h has bidegree hdeg. Returns (list of g dicts, singular values)."""
    t0 = time.time()
    ti, tj = i + hdeg[0], j + hdeg[1]
    cols = monos(ti, tj)
    mult = monos(ti - 2, tj - 2)
    rows_I = [mono_mul(m, Ec) for m in mult for Ec in E]
    A = to_matrix(rows_I, cols)
    Z, rankA, diag = nullspace_rows(A)
    target = monos(i, j)
    Bm = to_matrix([mul({m: 1.0}, h) for m in target], cols)
    Cm = Bm @ Z
    U, s, Vt = sla.svd(Cm, full_matrices=True)
    n = len(target)
    # singular values sorted descending; nullspace = trailing ones. find the gap
    s_full = np.concatenate([s, np.zeros(n - len(s))]) if len(s) < n else s
    rank = int(np.sum(s_full > 1e-8 * s_full[0]))
    nullity = n - rank
    gs = []
    for l in range(rank, n):                 # left null vectors: y^T Cm = 0
        y = U[:, l]
        gs.append({m: y[t] for t, m in enumerate(target) if abs(y[t]) > 1e-14})
    if verbose:
        print(f"(I:h)_({i},{j}) with h of bidegree {hdeg}: A {A.shape} rank {rankA} (A-gap {diag[rankA-1]:.1e}/{diag[rankA] if rankA < len(diag) else 0:.1e}), "
              f"BmZ {Cm.shape}: rank {rank}, nullity {nullity} -> quotient dim {rank}; "
              f"sv around gap: {s_full[max(rank-3,0):rank]} | {s_full[rank:rank+3]}; {time.time()-t0:.1f}s", flush=True)
    return gs, s_full

if __name__ == '__main__':
    from solve import random_instance
    A_, B_, C_ = random_instance(0)
    E = input_polys(A_, B_, C_)
    a, b, c, d, e, f, g, h = VARS
    s2 = poly_of(a*a+b*b+c*c+d*d)
    s3 = poly_of(e*e+f*f+g*g+h*h)
    out = {}
    for (i, j) in [(4, 4), (5, 4), (4, 5), (5, 5)]:
        gs, s = quotient_piece(E, s2, (2, 0), i, j)
        out[(i, j, 's2')] = gs
        gs3, s3v = quotient_piece(E, s3, (0, 2), i, j)
        out[(i, j, 's3')] = gs3
    pickle.dump(out, open('quot_pieces_seed0.pkl', 'wb'))

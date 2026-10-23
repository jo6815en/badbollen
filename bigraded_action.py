"""Bigraded action matrix for the concentric-balls problem.

V = (P/I*)_{(4,4)}, W = (P/I*)_{(5,4)}, both of dimension 600 (if regularity holds).
I*_{(i,j)} obtained numerically as { g : s2 g in I_raw_{(i+2,j)} } (exponent-1 saturation).
Multiplication maps M_a, M_b : V -> W.  Generalized eigenproblem M_b^T v = (b/a) M_a^T v.
"""
import sys, time, pickle
import numpy as np
import scipy.linalg as sla
from numsat import input_polys, saturate_piece, monos, to_matrix, VARS
from solve import random_instance

NV = 8

def quotient_basis(G, cols, n_basis):
    """Column-pivoted QR on G (rows span ideal piece). Returns (nonbasis idx, basis idx, X) with
    X such that NF(monomial nonbasis[i]) = -X[i] in basis coordinates."""
    Q, R, piv = sla.qr(G, mode='economic', pivoting=True)
    r = G.shape[0]
    nonb, basis = piv[:r], piv[r:]
    assert len(basis) == n_basis, (len(basis), n_basis)
    Gn, Gb = G[:, nonb], G[:, basis]
    X = np.linalg.solve(Gn, Gb) if Gn.shape[0] == Gn.shape[1] else np.linalg.lstsq(Gn, Gb, rcond=None)[0]
    cond = np.linalg.cond(Gn)
    return nonb, basis, X, cond

def mult_matrix(var, BV_monos, colsW, nonbW, basisW, X):
    idxW = {m: i for i, m in enumerate(colsW)}
    pos_nonb = {c: i for i, c in enumerate(nonbW)}
    pos_basis = {c: i for i, c in enumerate(basisW)}
    M = np.zeros((len(BV_monos), len(basisW)))
    for r, m in enumerate(BV_monos):
        mm = tuple(m[i] + (1 if i == var else 0) for i in range(NV))
        c = idxW[mm]
        if c in pos_basis:
            M[r, pos_basis[c]] = 1.0
        else:
            M[r, :] = -X[pos_nonb[c], :]
    return M

def recover_points(Vecs, BV_monos):
    """From eigenvectors (values of basis monomials at the solution, up to scale) get Cayley coords
    (b/a, c/a, d/a, f/e, g/e, h/e) via monomial ratios."""
    idx = {m: i for i, m in enumerate(BV_monos)}
    pairs = {}   # (num var, den var) -> list of (row m', row m)
    for m in BV_monos:
        for num, den in [(1, 0), (2, 0), (3, 0), (5, 4), (6, 4), (7, 4)]:
            if m[den] > 0:
                mm = list(m); mm[den] -= 1; mm[num] += 1; mm = tuple(mm)
                if mm in idx:
                    pairs.setdefault((num, den), []).append((idx[mm], idx[m]))
    out = np.zeros((Vecs.shape[1], 6), dtype=complex)
    for col in range(Vecs.shape[1]):
        v = Vecs[:, col]
        for t, key in enumerate([(1, 0), (2, 0), (3, 0), (5, 4), (6, 4), (7, 4)]):
            pr = pairs[key]
            num = np.array([v[i] for i, j in pr]); den = np.array([v[j] for i, j in pr])
            w = np.abs(den)
            out[col, t] = np.sum(w * num / den) / np.sum(w)      # weighted mean of ratios
    return out

def solve_instance(A, B, C, verbose=True):
    t0 = time.time()
    E = input_polys(A, B, C)
    g44, _ = saturate_piece(E, 4, 4, 1, verbose=verbose)
    g54, _ = saturate_piece(E, 5, 4, 1, verbose=verbose)
    cols44, cols54 = monos(4, 4), monos(5, 4)
    G44, G54 = to_matrix(g44, cols44), to_matrix(g54, cols54)
    if verbose:
        print(f"I*(4,4): {G44.shape} (quotient {len(cols44)-G44.shape[0]}), I*(5,4): {G54.shape} (quotient {len(cols54)-G54.shape[0]})", flush=True)
    if len(cols44) - G44.shape[0] != 600 or len(cols54) - G54.shape[0] != 600:
        print("quotient dimensions are not 600 -> exponent-1 saturation by s2 alone is not enough at these bidegrees")
        return None, None
    nonbV, basisV, _, condV = quotient_basis(G44, cols44, 600)
    nonbW, basisW, X, condW = quotient_basis(G54, cols54, 600)
    if verbose:
        print(f"pivot block condition numbers: V {condV:.1e}, W {condW:.1e}", flush=True)
    BV = [cols44[i] for i in basisV]
    Ma = mult_matrix(0, BV, cols54, nonbW, basisW, X)
    Mb = mult_matrix(1, BV, cols54, nonbW, basisW, X)
    lam, Vecs = sla.eig(Mb.T, Ma.T)
    pts = recover_points(Vecs, BV)
    if verbose:
        print(f"eigen: {len(lam)} eigenvalues, {np.sum(np.abs(lam.imag) < 1e-6)} real; total {time.time()-t0:.1f}s", flush=True)
    return pts, lam

if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    A, B, C = random_instance(seed)
    pts, lam = solve_instance(A, B, C)
    if pts is None:
        sys.exit(1)
    from action2 import verify
    # check eigenvalue consistency: lambda should equal b/a = pts[:,0]
    print(f"max |lambda - b/a| = {np.max(np.abs(lam - pts[:, 0])):.2e}")
    verify(pts, A, B, C, ref_path=f'sols_seed{seed}.pkl')
    with open(f'bigraded_sols_seed{seed}.pkl', 'wb') as fh:
        pickle.dump({'pts': pts, 'lam': lam}, fh)

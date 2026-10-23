"""Action-matrix solver: elimination template from a Macaulay matrix at degree D.

Variables (b,c,d,f,g,h,t); equations: 6 dets (deg 4) + Rabinowitsch t*(1+|v2|^2)(1+|v3|^2)-1 (deg 5).
Monomial basis B of the quotient (600 standard monomials) from the GF(p) Gröbner basis.
"""
import sys, pickle, itertools, time
import numpy as np
import scipy.linalg as sla
import sympy as sp
from solve import random_instance

VARS = sp.symbols('b c d f g h t')
NV = len(VARS)

def build_polys(A, B, C):
    b, c, d, f, g, h, t = VARS
    def rot(x, y, z):
        return sp.Matrix([[1+x*x-y*y-z*z, 2*(x*y-z), 2*(x*z+y)],
                          [2*(x*y+z), 1-x*x+y*y-z*z, 2*(y*z-x)],
                          [2*(x*z-y), 2*(y*z+x), 1-x*x-y*y+z*z]])
    R2, R3 = rot(b, c, d), rot(f, g, h)
    polys = []
    for k in range(6):
        M = sp.Matrix.hstack(sp.Matrix(A[k]), R2 * sp.Matrix(B[k]), R3 * sp.Matrix(C[k]))
        polys.append(sp.Poly(sp.expand(M.det()), *VARS))
    polys.append(sp.Poly(t*(1+b*b+c*c+d*d)*(1+f*f+g*g+h*h) - 1, *VARS))
    return [dict(p.terms()) for p in polys]   # exponent tuple -> float coeff

def parse_kbase(path):
    names = 'bcdfght'
    mons = []
    for tok in open(path).read().strip().split(','):
        if tok == '1':
            mons.append((0,)*NV); continue
        e = [0]*NV
        i = 0
        while i < len(tok):
            v = names.index(tok[i]); i += 1
            k = 0
            while i < len(tok) and tok[i].isdigit():
                k = 10*k + int(tok[i]); i += 1
            e[v] += k if k else 1
        mons.append(tuple(e))
    return mons

def monomials_upto(D):
    out = []
    for deg in range(D+1):
        for c in itertools.combinations_with_replacement(range(NV), deg):
            e = [0]*NV
            for v in c: e[v] += 1
            out.append(tuple(e))
    return out

def solve_action(polys, basis, D, act=0):
    t0 = time.time()
    Bset = set(basis)
    cols = monomials_upto(D)
    # reducible monomials: act * basis, not in basis
    R = sorted({tuple(m[i] + (1 if i == act else 0) for i in range(NV)) for m in basis} - Bset)
    Rset = set(R)
    E = [m for m in cols if m not in Bset and m not in Rset]
    order = E + R + list(basis)
    idx = {m: i for i, m in enumerate(order)}
    rows = []
    for p in polys:
        dp = max(sum(e) for e in p)
        for m in monomials_upto(D - dp):
            rows.append([(tuple(a + b for a, b in zip(m, e)), c) for e, c in p.items()])
    M = np.zeros((len(rows), len(order)))
    for r, terms in enumerate(rows):
        for e, c in terms:
            M[r, idx[e]] = float(c)
    nE, nR, nB = len(E), len(R), len(basis)
    print(f"D={D}: Macaulay {M.shape}, |E|={nE} |R|={nR} |B|={nB}", flush=True)
    # eliminate E: Q^T M, rows beyond rank(M_E) have zero E-part
    Q, Rr, piv = sla.qr(M[:, :nE], mode='full', pivoting=True)
    diag = np.abs(np.diag(Rr))
    rank = int(np.sum(diag > 1e-10 * diag[0]))
    N = Q[:, rank:].T @ M[:, nE:]           # rows with no E-support: (nrows-rank) x (nR+nB)
    print(f"  rank(M_E)={rank}, reduced rows={N.shape[0]}, cond-ish: smallest kept pivot {diag[rank-1]/diag[0]:.1e}, first dropped {diag[rank]/diag[0] if rank < len(diag) else 0:.1e}", flush=True)
    NR, NB = N[:, :nR], N[:, nR:]
    sv = np.linalg.svd(NR, compute_uv=False)
    print(f"  NR: {NR.shape}, sv range {sv[0]:.2e} .. {sv[min(nR, len(sv))-1]:.2e}  (need full column rank {nR})", flush=True)
    if len(sv) < nR or sv[nR-1] < 1e-9 * sv[0]:
        print("  template insufficient at this degree", flush=True)
        return None
    X = np.linalg.lstsq(NR, NB, rcond=None)[0]     # r = -X[r] . B  (since N rows: NR r + NB B = 0)
    Xr = -X
    # action matrix T: (act * basis_j) expressed in basis
    T = np.zeros((nB, nB))
    bidx = {m: i for i, m in enumerate(basis)}
    ridx = {m: i for i, m in enumerate(R)}
    for j, m in enumerate(basis):
        mm = tuple(m[i] + (1 if i == act else 0) for i in range(NV))
        if mm in bidx:
            T[j, bidx[mm]] = 1.0
        else:
            T[j, :] = Xr[ridx[mm], :]
    w, V = np.linalg.eig(T.T)
    one = bidx[(0,)*NV]
    var_rows = [bidx[tuple(1 if i == v else 0 for i in range(NV))] for v in range(6)]
    sols = (V[var_rows, :] / V[one, :]).T      # 600 x 6 (b,c,d,f,g,h)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)
    return sols

if __name__ == '__main__':
    D = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    A, B, C = random_instance(0)
    polys = build_polys(A, B, C)
    basis = parse_kbase('kbase_gfp.txt')
    assert len(basis) == 600
    sols = solve_action(polys, basis, D)
    if sols is None:
        sys.exit(1)
    # compare against monodromy solutions
    from monodromy import F_J_Ft
    P0 = (A.astype(complex), B.astype(complex), C.astype(complex))
    ref = np.array(pickle.load(open('sols_seed0.pkl', 'rb'))['sols'])
    res = np.array([np.linalg.norm(F_J_Ft(s, P0, P0)[0]) for s in sols])
    dist = np.array([np.min(np.linalg.norm(ref - s, axis=1)) for s in sols])
    matched = np.sum(dist < 1e-4)
    print(f"  residuals: median {np.median(res):.1e} max {res.max():.1e}; matched to monodromy set (<1e-4): {matched}/600; real: {np.sum(np.abs(sols.imag).max(axis=1) < 1e-6)}")
    # after 3 Newton polish steps
    pol = []
    for s in sols:
        v = s.copy()
        for _ in range(5):
            F, J, _ = F_J_Ft(v, P0, P0)
            v = v - np.linalg.solve(J, F)
        pol.append(v)
    pol = np.array(pol)
    dist2 = np.array([np.min(np.linalg.norm(ref - s, axis=1)) for s in pol])
    uniq = []
    for s in pol:
        if not any(np.linalg.norm(s - u) < 1e-6 for u in uniq):
            uniq.append(s)
    print(f"  after Newton polish: matched {np.sum(dist2 < 1e-8)}/600, distinct {len(uniq)}")

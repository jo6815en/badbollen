"""Bigraded action matrix, exponent-1 saturation.

V = (P/I*)_{(5,4)}  from (I : s3)_{(5,4)}      (codim 600)
W = (P/I*)_{(5,5)}  from (I : s2)_{(5,5)}      (codim 600)
M_e, M_f : V -> W multiplication by e, f.   M_f^T v = (f/e) M_e^T v at solutions.
"""
import sys, time, pickle
import numpy as np
import scipy.linalg as sla
from numsat import input_polys, monos, to_matrix, VARS
from numsat2 import quotient_piece, poly_of
from bigraded_action import quotient_basis, mult_matrix, recover_points
from solve import random_instance

def solve_instance(A, B, C, verbose=True):
    t0 = time.time()
    E = input_polys(A, B, C)
    a, b, c, d, e, f, g, h = VARS
    s2, s3 = poly_of(a*a+b*b+c*c+d*d), poly_of(e*e+f*f+g*g+h*h)
    gV, _ = quotient_piece(E, s3, (0, 2), 5, 4, verbose=verbose)
    gW, _ = quotient_piece(E, s2, (2, 0), 5, 5, verbose=verbose)
    colsV, colsW = monos(5, 4), monos(5, 5)
    GV, GW = to_matrix(gV, colsV), to_matrix(gW, colsW)
    qV, qW = len(colsV) - GV.shape[0], len(colsW) - GW.shape[0]
    if verbose:
        print(f"quotient dims: V {qV}, W {qW}", flush=True)
    if qV != 600 or qW != 600:
        return None, None
    nonbV, basisV, _, condV = quotient_basis(GV, colsV, 600)
    nonbW, basisW, X, condW = quotient_basis(GW, colsW, 600)
    if verbose:
        print(f"pivot block condition numbers: V {condV:.1e}, W {condW:.1e}", flush=True)
    BV = [colsV[i] for i in basisV]
    Me = mult_matrix(4, BV, colsW, nonbW, basisW, X)
    Mf = mult_matrix(5, BV, colsW, nonbW, basisW, X)
    lam, Vecs = sla.eig(Mf, Me)            # M_f w = (f/e) M_e w, w = evaluation vector on the W basis
    BW = [colsW[i] for i in basisW]
    pts = recover_points(Vecs, BW)
    if verbose:
        print(f"eigen: {len(lam)} eigenvalues, {np.sum(np.abs(lam.imag) < 1e-6)} real; total {time.time()-t0:.1f}s", flush=True)
    return pts, lam

if __name__ == '__main__':
    # usage: bigraded_action2.py <seed>          random unit-normal instance (verifies against sols_seed<seed>.pkl if present)
    #        bigraded_action2.py <config.txt>    18 rows "nx ny nz": ball1 colors 1..6, ball2 ..., ball3 ...
    arg = sys.argv[1] if len(sys.argv) > 1 else '0'
    if arg.isdigit():
        tag = f'seed{arg}'
        A, B, C = random_instance(int(arg))
    else:
        tag = arg.rsplit('.', 1)[0]
        N = np.loadtxt(arg).reshape(3, 6, 3)
        N = N / np.linalg.norm(N, axis=2, keepdims=True)
        A, B, C = N[0], N[1], N[2]
    pts, lam = solve_instance(A, B, C)
    if pts is None:
        sys.exit(1)
    print(f"max |lambda - f/e| = {np.max(np.abs(lam - pts[:, 3])):.2e}")
    import os
    if os.path.exists(f'sols_{tag}.pkl'):
        from verify import verify
        verify(pts, A, B, C, ref_path=f'sols_{tag}.pkl')
    # Newton polish on the original 6 equations, then write real solutions
    from monodromy import F_J_Ft, write_real
    P0 = (A.astype(complex), B.astype(complex), C.astype(complex))
    pol = []
    for s_ in pts:
        v = s_.copy()
        for _ in range(6):
            F, J, _ = F_J_Ft(v, P0, P0)
            v = v - np.linalg.solve(J, F)
        pol.append(v)
    res = [np.linalg.norm(F_J_Ft(v, P0, P0)[0]) for v in pol]
    print(f"after Newton polish: max residual {max(res):.1e}, real solutions {sum(np.linalg.norm(v.imag) < 1e-8 for v in pol)}")
    pickle.dump({'A': A, 'B': B, 'C': C, 'sols': pol, 'lam': lam}, open(f'am_sols_{tag}.pkl', 'wb'))
    write_real(pol, A, B, C, f'am_sols_{tag}_real.txt')

"""Monodromy solver: all complex solutions of the concentric-balls problem.

Cayley coordinates v2, v3 in C^3 (unnormalized R, polynomial equations).
F_c(v; P) = det(a_c, R(v2) b_c, R(v3) c_c),  P = (A, B, C), each 6x3.
Starting from known real solutions of a real instance P0, track along random
complex parameter loops P0 -> P1 -> P2 -> P0 and collect new solutions.
"""
import sys, pickle
import numpy as np
from multiprocessing import Pool
from solve import random_instance, solve

def R_of(v):
    x, y, z = v
    return np.array([
        [1+x*x-y*y-z*z, 2*(x*y-z),     2*(x*z+y)],
        [2*(x*y+z),     1-x*x+y*y-z*z, 2*(y*z-x)],
        [2*(x*z-y),     2*(y*z+x),     1-x*x-y*y+z*z]])

def dR_of(v):
    x, y, z = v
    return np.array([
        [[2*x, 2*y, 2*z], [2*y, -2*x, -2], [2*z, 2, -2*x]],
        [[-2*y, 2*x, 2], [2*x, 2*y, 2*z], [-2, 2*z, -2*y]],
        [[-2*z, -2, 2*x], [2, -2*z, 2*y], [2*x, 2*y, 2*z]]])

def det3(u, v, w):
    return np.einsum('ki,ki->k', u, np.cross(v, w))

def F_J_Ft(v, P, dP):
    """Residual F (6,), Jacobian dF/dv (6,6), and dF/dt (6,) along P + t*dP."""
    A, B, C = P
    dA, dB, dC = dP
    R2, R3 = R_of(v[:3]), R_of(v[3:])
    d2, d3 = dR_of(v[:3]), dR_of(v[3:])
    RB, RC = B @ R2.T, C @ R3.T          # rows: R2 b_c, R3 c_c
    F = det3(A, RB, RC)
    J = np.empty((6, 6), dtype=complex)
    for i in range(3):
        J[:, i] = det3(A, B @ d2[i].T, RC)
        J[:, 3+i] = det3(A, RB, C @ d3[i].T)
    Ft = det3(dA, RB, RC) + det3(A, dB @ R2.T, RC) + det3(A, RB, dC @ R3.T)
    return F, J, Ft

def track(v, P0, P1, tol=1e-10):
    """Track solution v of F(.;P0)=0 to a solution of F(.;P1)=0 along a straight line."""
    P0 = tuple(np.asarray(p, dtype=complex) for p in P0)
    dP = tuple(p1 - p0 for p0, p1 in zip(P0, P1))
    t, h = 0.0, 0.02
    v = np.asarray(v, dtype=complex)
    n_ok = 0
    while t < 1.0:
        h = min(h, 1.0 - t)
        Pt = tuple(p + t*d for p, d in zip(P0, dP))
        F, J, Ft = F_J_Ft(v, Pt, dP)
        # Euler predictor
        vp = v - h * np.linalg.solve(J, Ft)
        # Newton corrector
        Pn = tuple(p + (t+h)*d for p, d in zip(P0, dP))
        ok = False
        for _ in range(4):
            F, J, _ = F_J_Ft(vp, Pn, dP)
            dv = np.linalg.solve(J, F)
            vp = vp - dv
            if np.linalg.norm(dv) < 1e-11 * (1 + np.linalg.norm(vp)):
                ok = True
                break
        if ok and np.linalg.norm(vp - v) < 0.3 * (1 + np.linalg.norm(v)):
            v, t = vp, t + h
            n_ok += 1
            if n_ok >= 3:
                h, n_ok = min(2*h, 0.1), 0
        else:
            h, n_ok = h/2, 0
            if h < 1e-8:
                return None
    # polish
    for _ in range(3):
        F, J, _ = F_J_Ft(v, P1, dP)
        v = v - np.linalg.solve(J, F)
    return v

def track_loop(args):
    v, Ps = args
    for P0, P1 in zip(Ps[:-1], Ps[1:]):
        v = track(v, P0, P1)
        if v is None:
            return None
    return v

def dedupe_add(sols, new, tol=1e-6):
    added = 0
    for v in new:
        if v is None:
            continue
        if not any(np.linalg.norm(v - s) < tol * (1 + np.linalg.norm(s)) for s in sols):
            sols.append(v)
            added += 1
    return added

def cayley_from_R(R):
    from scipy.spatial.transform import Rotation as Rot
    rv = Rot.from_matrix(R).as_rotvec()
    th = np.linalg.norm(rv)
    return rv * (np.tan(th/2)/th if th > 1e-12 else 0.5)

def write_real(sols, A, B, C, path):
    """Real solutions as unit quaternions (w x y z) for R2, R3, plus the 6 meeting points."""
    from scipy.spatial.transform import Rotation as Rot
    lines = ['# ball 1 fixed. Per solution: q2 (w x y z), q3 (w x y z), then meeting point per color (unit vector).']
    for v in sols:
        if np.linalg.norm(v.imag) > 1e-8:
            continue
        v = v.real
        R2, R3 = R_of(v[:3]), R_of(v[3:])
        R2 /= 1 + v[:3] @ v[:3]
        R3 /= 1 + v[3:] @ v[3:]
        q2, q3 = Rot.from_matrix(R2).as_quat(scalar_first=True), Rot.from_matrix(R3).as_quat(scalar_first=True)
        pts = []
        for k in range(6):
            M = np.column_stack([A[k], R2 @ B[k], R3 @ C[k]])
            p = np.linalg.svd(M.T)[2][-1]
            pts.append(' '.join(f'{x:+.6f}' for x in p))
        lines.append(' '.join(f'{x:+.10f}' for x in q2) + '   ' + ' '.join(f'{x:+.10f}' for x in q3) + '   ' + '  '.join(pts))
    open(path, 'w').write('\n'.join(lines) + '\n')
    print(f'wrote {path}')

if __name__ == '__main__':
    # usage: monodromy.py <seed> [stable_loops]      random unit-normal instance
    #        monodromy.py <config.txt> [stable_loops]  18 rows "nx ny nz": ball1 colors 1..6, ball2 ..., ball3 ...
    arg = sys.argv[1] if len(sys.argv) > 1 else '0'
    max_stable = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    if arg.isdigit():
        seed, tag = int(arg), f'seed{arg}'
        A, B, C = random_instance(seed)
    else:
        seed, tag = 0, arg.rsplit('.', 1)[0]
        N = np.loadtxt(arg).reshape(3, 6, 3)
        N = N / np.linalg.norm(N, axis=2, keepdims=True)
        A, B, C = N[0], N[1], N[2]
    P0 = (A.astype(complex), B.astype(complex), C.astype(complex))
    real_sols = solve(A, B, C, n_starts=300, seed=seed)
    sols = [np.concatenate([cayley_from_R(R2), cayley_from_R(R3)]).astype(complex) for R2, R3 in real_sols]
    print(f"start: {len(sols)} real solutions", flush=True)
    rng = np.random.default_rng(100 + seed)
    stable, it = 0, 0
    with Pool() as pool:
        while stable < max_stable:
            it += 1
            P1 = tuple(rng.normal(size=(6, 3)) + 1j*rng.normal(size=(6, 3)) for _ in range(3))
            P2 = tuple(rng.normal(size=(6, 3)) + 1j*rng.normal(size=(6, 3)) for _ in range(3))
            Ps = [P0, P1, P2, P0]
            new = pool.map(track_loop, [(v, Ps) for v in sols], chunksize=4)
            n_fail = sum(v is None for v in new)
            added = dedupe_add(sols, new)
            stable = stable + 1 if added == 0 else 0
            print(f"loop {it}: tracked {len(new)}, failed {n_fail}, new {added}, total {len(sols)}", flush=True)
    # verify and classify
    n_real = sum(np.linalg.norm(v.imag) < 1e-8 for v in sols)
    res = [np.linalg.norm(F_J_Ft(v, P0, P0)[0]) for v in sols]
    print(f"final: {len(sols)} complex solutions, {n_real} real, max residual {max(res):.1e}")
    with open(f'sols_{tag}.pkl', 'wb') as f:
        pickle.dump({'A': A, 'B': B, 'C': C, 'sols': sols}, f)
    write_real(sols, A, B, C, f'sols_{tag}_real.txt')

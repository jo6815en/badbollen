"""Find real solutions of the concentric-balls problem numerically.

Ball 1 is fixed. Unknowns: rotations R2, R3 (Cayley params, 3 each).
Constraint per color c: det(a_c, R2 b_c, R3 c_c) = 0.
Random-restart Levenberg-Marquardt, then dedupe by rotation distance.
"""
import sys
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

def cayley(v):
    x, y, z = v
    s = 1 + x*x + y*y + z*z
    R = np.array([
        [1+x*x-y*y-z*z, 2*(x*y-z),     2*(x*z+y)],
        [2*(x*y+z),     1-x*x+y*y-z*z, 2*(y*z-x)],
        [2*(x*z-y),     2*(y*z+x),     1-x*x-y*y+z*z]])
    return R / s

def residual(p, A, B, C):
    R2, R3 = cayley(p[:3]), cayley(p[3:])
    return np.array([np.linalg.det(np.column_stack([A[k], R2 @ B[k], R3 @ C[k]]))
                     for k in range(6)])

def solve(A, B, C, n_starts=3000, seed=0, tol=1e-10):
    rng = np.random.default_rng(seed)
    sols = []
    for _ in range(n_starts):
        q2, q3 = Rot.random(random_state=rng), Rot.random(random_state=rng)
        p0 = np.concatenate([q2.as_rotvec(), q3.as_rotvec()])
        # rotvec -> Cayley: v = tan(theta/2) * axis
        for i in (0, 3):
            th = np.linalg.norm(p0[i:i+3])
            if th > 1e-12:
                p0[i:i+3] *= np.tan(th/2) / th
        res = least_squares(residual, p0, args=(A, B, C), method='lm', xtol=1e-14, ftol=1e-14, gtol=1e-14)
        if np.linalg.norm(res.fun) > tol:
            continue
        R2, R3 = cayley(res.x[:3]), cayley(res.x[3:])
        if not any(np.linalg.norm(R2 - s[0]) + np.linalg.norm(R3 - s[1]) < 1e-6 for s in sols):
            sols.append((R2, R3))
    return sols

def random_instance(seed):
    rng = np.random.default_rng(seed)
    N = rng.normal(size=(3, 6, 3))
    N /= np.linalg.norm(N, axis=2, keepdims=True)
    return N[0], N[1], N[2]

if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_starts = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    A, B, C = random_instance(seed)
    sols = solve(A, B, C, n_starts=n_starts, seed=seed)
    print(f"instance seed={seed}: {len(sols)} distinct real solutions")
    for i, (R2, R3) in enumerate(sols):
        r2, r3 = Rot.from_matrix(R2), Rot.from_matrix(R3)
        # common intersection points, for verification
        pts = []
        for k in range(6):
            M = np.column_stack([A[k], R2 @ B[k], R3 @ C[k]])
            _, s, vt = np.linalg.svd(M.T)
            pts.append(s[-1])
        print(f"  sol {i}: R2 rotvec={np.round(r2.as_rotvec(), 4)}  R3 rotvec={np.round(r3.as_rotvec(), 4)}  max sing.val={max(pts):.1e}")

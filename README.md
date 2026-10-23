# badbollen — three concentric balls, six colored great circles each

Orient the balls so that, for each color, the three great circles of that color meet in a
common point. Ball 1 is held fixed; the unknowns are the rotations of balls 2 and 3.

`summary.html` is a readable write-up of the result and the method (open it in a browser).

## Formulation
A great circle is a plane through the center with normal n. Three great circles with normals
n1, n2, n3 share a point iff det(n1, n2, n3) = 0. With R2, R3 the rotations of balls 2, 3 and
a_c, b_c, c_c the color-c normals on balls 1, 2, 3:

    det(a_c, R2 b_c, R3 c_c) = 0,   c = 1..6.

## Result
A generic configuration has exactly **600 complex solutions** (Singular, saturated ideal, two
random instances; confirmed by numerical monodromy and by the action-matrix solver). The
multihomogeneous Bezout bound is 1280; the excess is a 3-dimensional spurious component where
both quaternions are null (R2, R3 rank one with a common isotropic column). Random unit-normal
instances had 60 to 72 real solutions.

## The solver: `bigraded_action2.py`
One 600x600 generalized eigenproblem per instance, no path tracking, ~75 s with untuned dense
linear algebra. Quaternions q2 = (a,b,c,d), q3 = (e,f,g,h) are kept bihomogeneous in P^3 x P^3;
the six equations have bidegree (2,2).

1. Each bidegree piece of the raw ideal is spanned exactly by the bounded products m * E_c
   (no hidden low-degree elements, unlike any affine chart).
2. One ideal quotient by s3 = e^2+f^2+g^2+h^2 at bidegree (5,4), and one by s2 = a^2+b^2+c^2+d^2
   at (5,5), already give the exact ideal of the 600 points (codimension 600 in both).
   Bidegree (4,4) is below regularity (580); (I : s2) at (5,4) is not enough (660).
3. Numerically, (I : h) at bidegree (i,j) is the left null space of Bm Z, with Z the orthogonal
   complement of the Macaulay rows at (i,j)+deg h and Bm the rows h * m. Singular-value gap
   ~1e-2 / 1e-13.
4. V = quotient at (5,4), W = quotient at (5,5), bases by pivoted QR (condition ~60),
   multiplication maps M_e, M_f : V -> W, generalized eigenproblem M_f w = (f/e) M_e w.
   Eigenvectors are the values of the W-basis monomials at the solutions; the point is read
   off from monomial ratios and Newton-polished.

Validation: instance 0, all 600 solutions match the monodromy reference (595 to 1e-8 before
polishing); instances 1 and 2, 600 distinct solutions, real counts 64 and 60 equal to the
random-restart counts, all real solutions found.

Why plain elimination templates failed: in the affine Cayley ring, or with a Rabinowitsch
variable t*s = 1, the saturation only appears with multipliers of degree ~15-19, i.e. templates
with hundreds of thousands of columns. Unit quaternions remove the spurious component but move
it to infinity of the affine closure (Grobner basis degree 11, cofactors 15).

## Running
    python -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python bigraded_action2.py 0                 # random instance 0, checks against sols_seed0.pkl
    .venv/bin/python bigraded_action2.py config_seed0.txt  # your own configuration
    .venv/bin/python validate.py 1                         # residuals, distinctness, real count vs restarts

Config file: 18 rows `nx ny nz` (ball 1 colors 1..6, then ball 2, then ball 3; plane normals,
any scale). Output `am_sols_<name>_real.txt`: per real solution the unit quaternions (w x y z)
of R2 and R3 and the six meeting points; `example_output_config_seed0_real.txt` shows the format.

## Files
- `bigraded_action2.py` — the solver (entry point)
- `bigraded_action.py`  — basis choice, multiplication matrices, point recovery
- `numsat.py`, `numsat2.py` — bihomogeneous polynomials, Macaulay blocks, ideal quotient pieces
- `solve.py` — random instances; real solutions by random-restart Levenberg-Marquardt
- `monodromy.py` — reference method (all complex solutions by monodromy) and the residual/Jacobian used for polishing
- `verify.py`, `validate.py` — checks against reference sets and random restarts
- `config_seed0.txt`, `sols_seed0.pkl`, `sols_seed0_real.txt` — instance 0 and its 600 reference solutions
- `experiments/singular/` — Singular scripts behind the numbers: solution count (`count.sing`),
  saturation exponents (`quotients.sing`), bigraded Hilbert functions (`bigraded*.sing`),
  cofactor degrees (`liftstd.sing`, `stage2.sing`), unit-quaternion and 18-variable forms (`gb_*.sing`)
- `experiments/affine_templates/` — the affine elimination-template attempts that failed, kept for reference

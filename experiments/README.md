Exploratory material, not needed to run the solver.

singular/          Singular scripts (run with `Singular -q <file>`; random data over GF(32003)).
                   count.sing: 600 solutions via saturation. quotients.sing: exponent 1 suffices.
                   bigraded.sing / bigraded2.sing: bigraded Hilbert functions (slow, hours).
                   liftstd.sing, stage2.sing: cofactor degrees (why affine templates need degree ~19).
                   gb_rab*.sing: Rabinowitsch form; gb_unit*.sing: unit quaternions; gb18*.sing: 18 matrix entries.
affine_templates/  Dense Macaulay templates in the Cayley ring (action_matrix.py, degrees 8-10: fail) and
                   affine numerical saturation (numsat6.py, action2.py: ill-conditioned). kbase*.txt are the
                   standard-monomial bases exported from Singular that these scripts read.

function [Z, rankA, diagR, tol] = row_space_complement(A, relTol)
%ROW_SPACE_COMPLEMENT Orthonormal basis for the complement of rowspace(A).
% A is m-by-n. Z is n-by-(n-rankA), with A*Z approximately zero.

    if nargin < 2 || isempty(relTol)
        relTol = max(size(A)) * eps;
    end

    % Full pivoted QR: columns of A' span the row space of A.
    [Q, R, ~] = qr(A.', 'vector');

    diagR = abs(diag(R));
    scale = max(diagR);
    tol = relTol * scale;
    rankA = sum(diagR > tol);

    % The remaining columns of Q span the orthogonal complement.
    Z = Q(:, rankA + 1:size(A, 2));
end
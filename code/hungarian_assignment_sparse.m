function [assignments, assigned_rows, success, run_time] = hungarian_assignment_sparse(A, H)
% Compatibility name for the historical dense-input MATLAB matchpairs path.
% For an explicit edge-sparse input call matchpairs_assignment(A,H,'sparse').
    [assignments, assigned_rows, success, run_time] = matchpairs_assignment(A,H,'dense');
end

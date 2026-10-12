function [assignments, assigned_rows, success, run_time] = hungarian_assignment_sparse(A, H)
% HUNGARIAN_ASSIGNMENT_SPARSE - Apply the Hungarian algorithm only on allowed (sparse) assignments
% Input:
%   A: M x N binary matrix (M ≥ N). A(i,j) = 1 if row i can be assigned to column j
%   H: M x N cost matrix
% Output:
%   assignments: M x N logical matrix, with one true per assigned pair
%   assigned_rows: 1 x N vector. assigned_rows(j) = i means row i is assigned to column j
%   success: true if a full assignment is found
%   run_time: time to compute assignment

    [m, n] = size(A);

    if m < n
        error('This implementation requires that rows >= columns');
    end

    % Build full cost matrix, assigning large cost to disallowed entries
    C = ones(m, n) * 1e6;   % Initialize with large cost (infeasible)
    C(A == 1) = H(A == 1);  % Fill in only the allowed assignments

    % Use built-in matchpairs with full cost matrix
    tic;
    matched = matchpairs(C, 1e6);  % Each row: [row col]
    run_time = toc;

    % Build assignment matrix
    assignments = false(m, n);
    assigned_rows = zeros(1, n);

    for k = 1:size(matched, 1)
        i = matched(k, 1);
        j = matched(k, 2);
        assignments(i, j) = true;
        assigned_rows(j) = i;
    end

    % Check if all columns are assigned exactly once and assignments are valid
    success = (length(unique(matched(:,2))) == n) && all(A(sub2ind(size(A), assigned_rows, 1:n)) == 1);

    if success
        fprintf('Perfect matching (sparse) found.\n');
    else
        fprintf('No perfect matching (sparse) possible.\n');
    end

end

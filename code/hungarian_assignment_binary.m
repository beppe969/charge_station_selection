function [assignments, assigned_rows, success, run_time] = hungarian_assignment_binary(A, H)
% HUNGARIAN_ASSIGNMENT_BINARY - Apply the Hungarian algorithm to a binary matrix
% Input:
%   A: M x N binary matrix (M ≥ N). A(i,j) = 1 if row i can be assigned to column j
%   H: M x N cost matrix
% Output:
%   assignements: M x N matrix solution of the Munkres algorithm
%   assigned_rows: n x 1 vector. assigned_rows(j) = i means row i is assigned to column j
%   success: true if a full assignment is possible (i.e., one 1 per column, all in different rows)
%   run_time: running time of the single instance of Munkres

    [m, n] = size(A);

    if m < n
        error('This implementation requires that rows >= columns');
    end

    % Convert binary matrix A into a cost matrix for assignment
    % large penalty where A(i,j)=0
    C = H;
    C(A == 0) = 1e6; % Initialize large cost for infeasible links
    
    % Use built-in assignment algorithm (Munkres)
    % Use matchpairs from Optimization Toolbox (or implement munkres manually)
    % Convert to square cost matrix for munkres
    tic
    assignments = munkres(C);  % assignments(j) = i (row assigned to column j)
    run_time = toc;

    % Check if all real columns were matched to real rows
    assigned_rows = zeros(1, n); % Initialize
    for j = 1:n
        i = find(assignments(:, j));  % Find the row assigned to column j
        if ~isempty(i)
            assigned_rows(j) = i;     % Assigned the row to column j
        else
            assigned_rows(j) = 0;     % No assignment
        end
    end
    % Create mask of valid assignments (row index in range and non-zero)
    valid = assigned_rows > 0 & assigned_rows <= m;
    % Filter only valid row-column pairs
    rows = assigned_rows(valid);
    cols = find(valid);
    
    % Now check success only on valid assignments
    success = length(rows) == n && all(A(sub2ind(size(A), rows, cols)) == 1);
    if success % If it is true
        fprintf('Perfect matching found.\n');
        % Print the assignments
        %disp(assignments);
    else
        fprintf('No perfect matching possible.\n');
        %disp(assignments);
    end
    %disp(C)
    %disp(sum(C.*assignments))

    % Display result
    %for j = 1:n
    %    fprintf('Column %d assigned to Row %d\n', j, assigned_rows(j));
    %end



end

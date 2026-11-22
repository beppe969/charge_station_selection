function [assignments, assigned_rows, success, run_time] = hungarian_assignment_binary(A, H)
% Dense Munkres implementation with cardinality prioritized over total cost.
% Return eligible pairs only, including a partial solution when necessary.
    [m,n] = size(A);
    t0 = tic;
    [ports, vehicles, costs, penalty] = assignment_edge_costs(A, H);
    assignments = false(m,n);
    if ~isempty(costs)
        work = repmat(penalty,m,n);
        work(sub2ind([m,n],ports,vehicles)) = costs;
        assignments = munkres(work) & logical(A);
    end
    run_time = toc(t0);
    [rows,cols] = find(assignments);
    assigned_rows = zeros(1,n);
    assigned_rows(cols) = rows;
    success = numel(cols) == n;
end

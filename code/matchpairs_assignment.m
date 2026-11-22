function [assignments, assigned_rows, success, run_time, setup_time, input_bytes] = matchpairs_assignment(A, H, storage)
% Maximum-cardinality, minimum-cost assignment using MATLAB matchpairs.
% A is port-by-vehicle. An unmatched vehicle has assigned_rows(j) = 0.
% 'dense' and 'sparse' pass the same edge weights with different storage.
    if nargin < 3, storage = 'dense'; end
    [m,n] = size(A);
    assigned_rows = zeros(1,n);
    assignments = false(m,n);
    t0 = tic;
    run_time = 0;
    setup_time = 0;
    input_bytes = 0;
    [ports, vehicles, costs, reward] = assignment_edge_costs(A, H);
    if ~isempty(costs)
        % Matching an allowed pair earns reward. Missing edges and unmatched
        % vertices are worth zero, so an extra allowed pair always wins.
        weights = costs - reward;
        switch storage
            case 'dense'
                edgeCost = zeros(m,n);
                edgeCost(sub2ind([m,n],ports,vehicles)) = weights;
            case 'sparse'
                edgeCost = sparse(ports,vehicles,weights,m,n);
            otherwise
                error('assignment:storage', 'Expected dense or sparse storage.');
        end
        setup_time = toc(t0);
        matrixInfo = whos('edgeCost');
        input_bytes = matrixInfo.bytes;
        tSolve = tic;
        pairs = matchpairs(edgeCost, 0);
        run_time = toc(tSolve);
        indices = sub2ind([m,n], pairs(:,1), pairs(:,2));
        keep = logical(A(indices));  % a zero-cost tie may include a missing edge
        pairs = pairs(keep,:);
        assignments(indices(keep)) = true;
        assigned_rows(pairs(:,2)) = pairs(:,1);
    end
    success = nnz(assigned_rows) == n;
end

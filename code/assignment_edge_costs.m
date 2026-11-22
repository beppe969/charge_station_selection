function [ports, vehicles, costs, reward] = assignment_edge_costs(A, H)
% Extract finite costs on the acceptable edges and normalize them to [0,2].
% A reward exceeding 2*min(M,N) buys one extra match before minimizing cost.
    if ~isequal(size(A), size(H)) || ~ismatrix(A) || ~isreal(H)
        error('assignment:dimensions', 'A and H must be real matrices of the same size.');
    end
    [ports, vehicles] = find(A);
    costs = full(double(H(sub2ind(size(A), ports, vehicles))));
    costs = costs(:);
    if any(~isfinite(costs))
        error('assignment:costs', 'Allowed edges must have finite costs.');
    end
    if ~isempty(costs)
        scale = max(abs(costs));
        if scale > 0
            costs = costs / scale;
        end
        costs = costs - min(costs);
    end
    reward = 2 * min(size(A)) + 1;
end

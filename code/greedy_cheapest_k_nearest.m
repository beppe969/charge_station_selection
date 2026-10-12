function [x, assigned, order] = greedy_cheapest_k_nearest(D, C, A, k, order)
% GREEDY_CHEAPEST_K_NEAREST  "Cheapest among the k nearest ports" baseline.
%
%   [x, assigned, order] = GREEDY_CHEAPEST_K_NEAREST(D, C, A, k, order)
%   is a slightly smarter myopic policy: each EV looks at its k nearest
%   acceptable & still-free ports and picks, among them, the one with the
%   lowest charging price C_i. It interpolates between "go to the nearest"
%   (k = 1) and "go to the globally cheapest reachable port" (k large).
%
%   INPUTS
%     D     : M x N distance matrix
%     C     : M x 1 charging-cost vector [$/kWh]
%     A     : M x N binary acceptable matrix
%     k     : neighbourhood size (number of nearest ports to consider)
%     order : (optional) 1 x N processing order; random if omitted
%
%   OUTPUTS
%     x, assigned, order : as in greedy_nearest_assignment
%
%   This baseline shows that adding a little price-awareness to the myopic
%   rule narrows, but does not close, the gap to the coordinated MILP.

    [M, N] = size(D);
    if nargin < 5 || isempty(order)
        order = randperm(N);
    end

    taken = false(M, 1);
    x = zeros(M, N);
    assigned = 0;

    for t = 1:N
        j = order(t);
        cand = find(A(:, j) == 1 & ~taken);        % free acceptable ports
        if isempty(cand)
            continue;
        end
        % Keep the k nearest candidates (or all of them if fewer than k)
        [~, sortIdx] = sort(D(cand, j), 'ascend');
        kk = min(k, numel(cand));
        near = cand(sortIdx(1:kk));
        % Among those, choose the cheapest port
        [~, cmin] = min(C(near));
        i = near(cmin);
        x(i, j) = 1;
        taken(i) = true;
        assigned = assigned + 1;
    end
end

function [x, assigned, order] = greedy_cost_assignment(c, A, order)
% GREEDY_COST_ASSIGNMENT  Sequential best response on the full criterion.
%
%   [x, assigned, order] = greedy_cost_assignment(c, A, order)
%
% Same sequential commitment as the other greedy rules, but each vehicle
% minimises the complete cost c_ij that the model minimises globally,
% instead of a proxy for it. It is the best response of a perfectly informed
% but isolated driver: one who knows the tariff, the queue and the charging
% speed of every port, and has no idea what the other drivers are about to
% do.
%
% It exists to separate two effects the other baselines conflate. Against
% the nearest-port rule it measures what the criterion is worth; against the
% optimum, what coordination is worth once the criterion is fixed.
%
% Being better informed does not make a sequential rule complete: a vehicle
% processed late can still find all of its acceptable ports taken. The
% unserved count is returned for that reason and must be reported next to
% any average computed over the served vehicles.


    [M, N] = size(c);
    if nargin < 3 || isempty(order)
        order = randperm(N);
    end

    assert(all(size(A) == [M N]), 'A must have the same size as c');

    taken    = false(M, 1);        % availability flag of each port
    x        = zeros(M, N);
    assigned = 0;

    for k = 1:N
        j = order(k);                              % current EV
        cand = find(A(:, j) == 1 & ~taken);        % free acceptable ports
        if isempty(cand)
            continue;                              % EV j remains unserved
        end
        [~, kmin] = min(c(cand, j));               % cheapest such port,
        i = cand(kmin);                            % on the FULL criterion
        x(i, j)  = 1;
        taken(i) = true;
        assigned = assigned + 1;
    end
end

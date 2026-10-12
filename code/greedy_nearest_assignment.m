function [x, assigned, order] = greedy_nearest_assignment(D, A, order)
% GREEDY_NEAREST_ASSIGNMENT  Decentralised "go to the nearest port" baseline.
%
%   [x, assigned, order] = GREEDY_NEAREST_ASSIGNMENT(D, A, order)
%   emulates a myopic policy in which each EV, processed one at a time,
%   drives to the CLOSEST charging port that is (i) acceptable for it and
%   (ii) still free. This mimics uncoordinated real-world behaviour and
%   serves as a lower-effort benchmark against the globally optimal MILP.
%
%   INPUTS
%     D     : M x N distance matrix (port i to EV j)
%     A     : M x N binary acceptable matrix
%     order : (optional) 1 x N permutation giving the processing order of
%             the EVs. If omitted, a random order is drawn. Because the
%             outcome of a decentralised policy depends on who chooses
%             first, callers typically average over many random orders.
%
%   OUTPUTS
%     x        : M x N binary assignment (a column of zeros = unserved EV)
%     assigned : number of EVs that obtained a port (<= N)
%     order    : the processing order actually used
%
%   NOTE: greedy nearest optimises DISTANCE only and ignores the global
%   objective c; it may therefore leave EVs unassigned even when the MILP
%   is feasible (a port grabbed early by one EV may be the only option of a
%   later EV). This is exactly the coordination gap the paper highlights.

    [M, N] = size(D);
    if nargin < 3 || isempty(order)
        order = randperm(N);
    end

    taken = false(M, 1);          % availability flag of each port
    x = zeros(M, N);
    assigned = 0;

    for k = 1:N
        j = order(k);                              % current EV
        cand = find(A(:, j) == 1 & ~taken);        % free acceptable ports
        if isempty(cand)
            continue;                              % EV j remains unserved
        end
        [~, kmin] = min(D(cand, j));               % nearest such port
        i = cand(kmin);
        x(i, j) = 1;
        taken(i) = true;
        assigned = assigned + 1;
    end
end

function [match, num_matches] = hopcroft_karp(A)
% HOPCROFT_KARP - Computes maximum matching in a bipartite graph using Hopcroft–Karp algorithm
% INPUT:
%   A            - m x n binary matrix (adjacency matrix), where A(i,j) = 1 means an edge from row i to column j
% OUTPUT:
%   match        - 1 x n vector, match(j) = i means column j is matched with row i
%   num_matches  - total number of matched pairs

[m, n] = size(A);
match = zeros(1, n);      % match(j): row index assigned to column j
pair_u = zeros(1, m);     % pair_u(i): column index assigned to row i
dist = zeros(1, m);       % distance array for BFS
INF = m + 1;              % a value representing "infinity"

% BFS stage - builds layers (levels) of the search tree
    function found = bfs()
        queue = [];
        for u = 1:m
            if pair_u(u) == 0     % if u is free (unmatched)
                dist(u) = 0;
                queue(end+1) = u;
            else
                dist(u) = INF;
            end
        end
        found = false;
        while ~isempty(queue)
            u = queue(1);
            queue(1) = [];
            if dist(u) < INF
                for v = find(A(u,:))  % for each neighbor v of u
                    if match(v) == 0
                        found = true;  % found an unmatched node on the right
                    elseif dist(match(v)) == INF
                        dist(match(v)) = dist(u) + 1;
                        queue(end+1) = match(v);  % add neighbor to the queue
                    end
                end
            end
        end
    end

% DFS stage - tries to find augmenting paths using levels from BFS
    function res = dfs(u)
        for v = find(A(u,:))  % explore neighbors of u
            if match(v) == 0 || ...
               (dist(match(v)) == dist(u) + 1 && dfs(match(v)))
                pair_u(u) = v;
                match(v) = u;
                res = true;
                return;
            end
        end
        dist(u) = INF;  % mark u as dead end
        res = false;
    end

% MAIN Hopcroft–Karp loop
% Repeat BFS + DFS phases until no more augmenting paths can be found
while bfs()
    for u = 1:m
        if pair_u(u) == 0  % only start DFS from free nodes
            dfs(u);
        end
    end
end

num_matches = sum(match > 0);  % count number of successful matches
end

function [x,setupTime,solveTime,inputBytes,objective] = filtered_assignment_program(A,C,integer,lp,mip)
% Minimum-cost full-service LP or MILP using only eligible edges.
    prep = tic;
    [m,n] = size(A);
    [ports,vehicles] = find(A);
    indices = sub2ind([m,n],ports,vehicles);
    edges = numel(indices); k = (1:edges)';
    capacities = sparse(ports,k,1,m,edges);
    requests = sparse(vehicles,k,1,n,edges);
    f = C(indices); lower = zeros(edges,1); cap = ones(m,1); demand = ones(n,1);
    setupTime = toc(prep);
    info = whos('capacities','requests');
    inputBytes = sum([info.bytes]);  % sparse incidence matrices, excluding other workspace arrays
    clock = tic;
    if integer
        [values,objective,flag] = intlinprog(f,k,capacities,cap,requests,demand,lower,[],mip);
    else
        [values,objective,flag] = linprog(f,capacities,cap,requests,demand,lower,[],lp);
    end
    solveTime = toc(clock);
    assert(flag == 1,'Filtered program failed to certify an optimum.');
    x = zeros(m,n);
    x(indices) = values;
end

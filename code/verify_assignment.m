function trials = verify_assignment(outputDir)
% Validate the budget reduction and benchmark actual dense and sparse inputs.
% Run from MATLAB: addpath('code'); verify_assignment
% Starts each clock from the same mask and costs on the same feasible case.
    if nargin == 0
        outputDir = fullfile(fileparts(fileparts(mfilename('fullpath'))),'results','solver');
    end
    if ~isfolder(outputDir), mkdir(outputDir); end
    lpOptions = optimoptions('linprog','Display','none','Algorithm','dual-simplex');
    mipOptions = optimoptions('intlinprog','Display','off','RelativeGapTolerance',0);
    rng(31027,'twister');
    check_examples(lpOptions,mipOptions);
    repetitions = 5;
    sizes = [80,200];
    densities = [.05,.25,.60];
    methods = {'matchpairs_dense','matchpairs_sparse','filtered_LP','filtered_MILP'};
    rows = cell(numel(sizes)*numel(densities)*repetitions*numel(methods),12);
    row = 0;
    for n = sizes
        m = round(1.2*n);
        for p = densities
            for rep = 1:repetitions
                A = rand(m,n) < p;
                A(sub2ind([m,n],randperm(m,n),1:n)) = true;
                C = 10+90*rand(m,n);
                cost = nan(1,numel(methods));
                order = randperm(numel(methods));
                for k = order
                    wall = tic;
                    if k <= 2
                        storage = {'dense','sparse'};
                        [x,~,ok,solveTime,setupTime,inputBytes] = matchpairs_assignment(A,C,storage{k});
                        assert(ok);
                    else
                        [x,setupTime,solveTime,inputBytes] = filtered_assignment_program(A,C,k==4,lpOptions,mipOptions);
                    end
                    endToEnd = toc(wall);
                    assert(~any(x(~A)>1e-7) && max(abs(sum(x,1)-1))<1e-7);
                    assert(all(sum(x,2)<=1+1e-7) && max(abs(x(:)-round(x(:))))<1e-7);
                    cost(k) = sum(C(:).*x(:));
                    row = row+1;
                    rows(row,:) = {m,n,p,rep,nnz(A),methods{k},setupTime,solveTime, ...
                                   endToEnd,inputBytes,cost(k),max(abs(x(:)-round(x(:))))};
                end
                assert(max(cost)-min(cost)<1e-6);
            end
        end
    end
    trials = cell2table(rows,'VariableNames',{'ports','evs','draw_probability','replicate', ...
        'edges','method','setup_s','solve_s','end_to_end_s','primary_matrix_bytes','objective','integrality_residual'});
    writetable(trials,fullfile(outputDir,'benchmark_trials.csv'));
    groups = findgroups(trials.evs,trials.draw_probability,trials.method);
    first = splitapply(@(v)v(1),reshape(1:height(trials),[],1),groups);
    summary = trials(first,{'ports','evs','draw_probability','method'});
    for field = {'edges','setup_s','solve_s','end_to_end_s','primary_matrix_bytes'}
        values = trials.(field{1});
        summary.([field{1} '_median']) = splitapply(@median,values,groups);
    end
    summary.min_end_to_end_s = splitapply(@min,trials.end_to_end_s,groups);
    summary.max_end_to_end_s = splitapply(@max,trials.end_to_end_s,groups);
    writetable(summary,fullfile(outputDir,'benchmark_summary.csv'));
    try
        cpu = feature('getcpu');
    catch
        cpu = computer;
    end
    try
        bean = java.lang.management.ManagementFactory.getOperatingSystemMXBean();
        ram = sprintf('%.0f',double(bean.getTotalPhysicalMemorySize()));
    catch
        ram = 'unknown';
    end
    toolbox = ver('optim');
    fid = fopen(fullfile(outputDir,'environment.txt'),'w');
    assert(fid >= 0);
    closeLog = onCleanup(@() fclose(fid));
    fprintf(fid,'MATLAB %s\nOptimization Toolbox %s\nArchitecture %s\nCPU %s\nSystem memory (bytes) %s\n', ...
        version,toolbox.Version,computer,strtrim(cpu),strtrim(ram));
    fprintf(fid,'MATLAB computational threads %d\nseed 31027 (twister)\nrepetitions %d\n',maxNumCompThreads,repetitions);
    fprintf(fid,['Original budget-row LP 7, original MILP and reduced LP 11 (one-EV example).\n' ...
        '12 structurally generated full-service instances agree across original MILP, filtered LP and matchpairs.\n' ...
        'Additional exact checks cover tied, rectangular, square, empty, infeasible and high-cost inputs.\n']);
    disp(summary(:,{'evs','draw_probability','method','end_to_end_s_median'}));
end

function check_examples(lp,mip)
    % Enumerated references include ties, no eligible edge, rectangular partial
    % service and a cost exceeding the old constant of one million.
    examples = {logical([1 1;1 1]),[2 2;2 2],2,4; ...
                logical([1 1;0 0]),[3 1;0 0],1,1; ...
                logical([1 1 0;0 0 1]),[4 2 0;0 0 9],2,11; ...
                false(3,2),zeros(3,2),0,0; ...
                logical([1 1;1 0;0 1]),[5e8 5e8;2e8 0;0 4e8],2,6e8; ...
                logical([1 1;1 1]),[-4 5;3 -1],2,-5};
    for e=1:size(examples,1)
        A=examples{e,1}; C=examples{e,2};
        for solver = {@(a,c)matchpairs_assignment(a,c,'dense'), ...
                      @(a,c)matchpairs_assignment(a,c,'sparse'), @hungarian_assignment_binary}
            [x,rows,ok]=solver{1}(A,C);
            assert(nnz(x)==examples{e,3} && abs(sum(C(x))-examples{e,4})<1e-6);
            assert(~any(x(~A)) && all(sum(x,2)<=1) && all(sum(x,1)<=1));
            assert(isequal(rows>0,any(x,1)) && ok==(nnz(x)==size(A,2)));
        end
    end
    R = true(2,1); prices = [1;3]; cap = 2; C=[11;3];
    [costM,ok] = budget_program(R,C,prices,cap,true,lp,mip);
    assert(ok && abs(costM-11)<1e-8);
    [costLP,ok] = budget_program(R,C,prices,cap,false,lp,mip);
    assert(ok && abs(costLP-7)<1e-8);
    A=R & (prices<=cap);
    x = filtered_assignment_program(A,C,false,lp,mip);
    assert(abs(C'*x-11)<1e-8);
    for k=1:12
        n = 2+mod(k,4); m = n+mod(k,3);
        R = rand(m,n)<0.5;
        prices = .2 + .5*rand(m,1);
        cap = .2 + .5*rand(1,n);
        C = 50*rand(m,n);
        ports = randperm(m,n);
        R(sub2ind([m,n],ports,1:n)) = true;
        cap = max(cap,reshape(prices(ports),1,[]));
        A = R & (prices <= cap);
        [cm,ok] = budget_program(R,C,prices,cap,true,lp,mip);
        assert(ok);
        xl = filtered_assignment_program(A,C,false,lp,mip);
        x = matchpairs_assignment(A,C,'sparse');
        assert(nnz(x)==n && abs(sum(C(x))-cm)<1e-7 && abs(sum(C(:).*xl(:))-cm)<1e-7);
    end
    fprintf('Passed original budget-row MILP, LP counterexample, filtered LP and matching checks.\n');
end

function [objective,success] = budget_program(R,C,prices,cap,integer,lp,mip)
    [m,n]=size(R);
    % Original aggregate budget rows. Unreachable pairs alone have zero upper bound.
    demand = kron(speye(n),ones(1,m));
    capacity = repmat(speye(m),1,n);
    budget = kron(speye(n),prices');
    inequalities = [capacity;budget];
    rhs = [ones(m,1);cap(:)];
    if integer
        [~,objective,flag] = intlinprog(C(:),1:numel(C),inequalities,rhs,demand,ones(n,1), ...
            zeros(numel(C),1),double(R(:)),mip);
    else
        [~,objective,flag] = linprog(C(:),inequalities,rhs,demand,ones(n,1), ...
            zeros(numel(C),1),double(R(:)),lp);
    end
    success = flag == 1;
end

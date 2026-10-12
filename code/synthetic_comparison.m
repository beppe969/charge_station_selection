function results = synthetic_comparison(figdir)
% SYNTHETIC_COMPARISON  Optimum against two myopic drivers, synthetic data (Sec. 3.3.2).
%
% M = 500 ports, N = 250 vehicles, 100 instances. Baselines: go-to-the-
% nearest and cheapest-of-k-nearest, each averaged over five random
% processing orders because a sequential rule depends on who chooses first.
%
% The informed sequential rule is deliberately NOT here: on these instances
% prices, waits and powers are i.i.d. across ports, so a rule that takes the
% best free port each time lands close to the optimum almost by
% construction. It only discriminates where the cost matrix is correlated,
% i.e. on the real networks. See realworld_analysis.
%
% Produces the four synthetic_*.pdf boxplots and a LaTeX table.

if nargin < 1;  figdir = 'Figures';  end

% Make sure the output folder exists from the very beginning. The automatic
% save block at the end of the function would create it, but the per-figure
% export lines scattered through the code run BEFORE that, so uncommenting
% one of them would otherwise fail with "No such file or directory".
%
% NOTE: isfolder, not exist(...,'dir'). The latter searches the whole MATLAB
% path, so a Figures folder belonging to some other project would satisfy the
% test while the relative write to ./Figures still fails. isfolder resolves
% against the current directory, which is where exportgraphics will write.
if ~isempty(figdir)
    if ~isfolder(figdir)
        [okDir, msgDir] = mkdir(figdir);
        if ~okDir
            error('Cannot create the output folder\n    %s\nReason: %s', ...
                  fullfile(pwd, figdir), msgDir);
        end
    end
    fprintf('Figures will be written to: %s\n', fullfile(pwd, figdir));
end

rng(1); % set seed

%% Simulation parameters

M     = 500;    % Number of charging ports
N     = 250;    % Number of electric vehicles  (load ratio N/M = 0.5)
REPS  = 100;    % independent instances
ORDERS = 5;     % random processing orders per greedy run
kNear = 3;      % neighbourhood size of the k-nearest policy
p_acc = 0.6;    % acceptance probability of the acceptable matrix A

BIGM = 1e6;     % penalty used to forbid the non-acceptable pairs

% Objective coefficients. Every term of the objective is expressed in
% dollars, so these are conversion factors with a physical meaning
% ($/km and $/min): the model has two of them, since the energy term is
% already money and needs no conversion.
% See objective_weights.m for where the values come from, and
% sensitivity_alpha.m for the evidence that the conclusions do not depend
% on them. objective_weights('low') and ('high') change the value of time.
alpha = objective_weights();

% Energy demand of the EVs [kWh]. It is now DRAWN PER VEHICLE rather than
% fixed: the objective contains C_i*Q_j and Q_j/P_i, so a heterogeneous
% demand is what makes those terms genuinely depend on the vehicle. With a
% common Q the price term would only be a rescaled tariff, which is the
% degenerate case the revised formulation is meant to move away from.
% A vehicle arrives with a partly depleted battery of about 60 kWh and
% charges to roughly 90%, so the energy requested lands in this range.
Qmin = 20;   Qmax = 60;    % [kWh]

% Charging prices are uniform between the min and max US state-level price
Cmin = 0.303;  % [$/kWh]
Cmax = 0.530;  % [$/kWh]

% The sparse Hungarian is the solver of choice; fall back to the dense one
if exist('hungarian_assignment_sparse', 'file') == 2
    solver_name = 'sparse Hungarian';
    solve_fun   = @hungarian_assignment_sparse;
else
    warning('hungarian_assignment_sparse not found: using the dense version.');
    solver_name = 'dense Hungarian';
    solve_fun   = @hungarian_assignment_binary;
end

% NOTE. A fourth baseline, a sequential rule scoring the ports with the
% full cost c, is deliberately NOT run here; it lives in
% realworld_analysis.m instead. On these instances prices, waiting times
% and powers are i.i.d. across ports and distances are uniform, so an
% informed sequential rule picking the best free port at every step lands
% close to the optimum almost by construction. The comparison only
% discriminates where the cost matrix is correlated, which is the case on
% the real networks, where the ports of one station share its tariff and
% its queue. See greedy_cost_assignment.m.
labels      = {'Optimal (Hungarian)', 'Greedy nearest', sprintf('Cheapest of %d-nearest', kNear)};
shortLabels = {'Optimal', 'Nearest', sprintf('Cheapest-%d', kNear)};
nMethods    = 3;

fprintf('\n========= SYNTHETIC COMPARISON (M = %d, N = %d) =========\n', M, N);
fprintf('  solver: %s, %d instances\n\n', solver_name, REPS);

%% ------------------------------------------------------------------
%  Monte Carlo loop
%  ------------------------------------------------------------------

objAcc  = nan(REPS, nMethods);
distAcc = nan(REPS, nMethods);
costAcc = nan(REPS, nMethods);
waitAcc = nan(REPS, nMethods);
servAcc = nan(REPS, nMethods);
timeAcc = nan(REPS, nMethods);

for rep = 1:REPS

    % Simulate charging port parameters
    D = rand(M,N)*10;                       % Distance port i to vehicle j [km]
    W = rand(M,1)*15;                       % Waiting time at each port [min]
    C = Cmin + (Cmax - Cmin) * rand(M,1);   % Charging cost at each port [$/kWh]
    P = draw_port_power(M);        % Charging power of each port [kW]
    Q = Qmin + (Qmax - Qmin) * rand(1,N);   % Energy demand of each EV  [kWh]

    A = double(rand(M,N) < p_acc);          % generate acceptable matrix A

    % Discard instances in which some vehicle cannot be served at all
    [~, num_matches] = hopcroft_karp(A);
    if num_matches < N
        fprintf('  instance %3d: infeasible, skipped\n', rep);
        continue;
    end

    % Objective function term "c"
    c = build_objective_matrix(D, W, C, P, Q, alpha);

    % ---- (1) optimal
    H = c;
    H(~A) = BIGM;   % set a very big number for unadmissible selections
    t0 = tic;
    xo = solve_fun(A, H);
    timeAcc(rep,1) = toc(t0);
    xo = full(double(xo));          % the solver may return a sparse matrix
    s1 = assignment_stats(xo, c, D, W, C, N);

    % ---- (2)(3) greedy, averaged over random processing orders
    [s2, t2] = avg_greedy(@() greedy_nearest(D, A),              ORDERS, c, D, W, C, N);
    [s3, t3] = avg_greedy(@() greedy_cheapest_k(D, C, A, kNear), ORDERS, c, D, W, C, N);
    timeAcc(rep,2) = t2;
    timeAcc(rep,3) = t3;

    S = {s1, s2, s3};
    for k = 1:nMethods
        objAcc(rep,k)  = S{k}.obj;
        distAcc(rep,k) = S{k}.dist;
        costAcc(rep,k) = S{k}.cost;
        waitAcc(rep,k) = S{k}.wait;
        servAcc(rep,k) = S{k}.served;
    end

    if mod(rep, 10) == 0
        fprintf('  instance %3d/%d done | optimal %.1f | nearest %.1f | cheapest-k %.1f\n', ...
                rep, REPS, s1.obj, s2.obj, s3.obj);
    end
end

valid  = all(isfinite(objAcc), 2);
nValid = nnz(valid);

%% ------------------------------------------------------------------
%  Aggregated statistics
%  ------------------------------------------------------------------

mu = @(v) mean(v(valid,:), 1);
sd = @(v) std( v(valid,:), 0, 1);

mObj  = mu(objAcc);   sObj  = sd(objAcc);
mDist = mu(distAcc);  sDist = sd(distAcc);
mCost = mu(costAcc);  sCost = sd(costAcc);
mWait = mu(waitAcc);  sWait = sd(waitAcc);
mServ = mu(servAcc);  sServ = sd(servAcc);
mTime = mu(timeAcc);  sTime = sd(timeAcc);

gap = 100 * (mObj - mObj(1)) / abs(mObj(1));   % relative loss vs the optimum

fprintf('\n========== AGGREGATED OVER %d FEASIBLE INSTANCES ==========\n\n', nValid);
% The solve time is measured and kept in "results", but it is not reported
% here: this comparison is about the QUALITY of the assignment, and the
% timing question is answered separately in the scalability experiment.
% The fraction of served vehicles is omitted for the same kind of reason:
% at this acceptance probability all three methods always serve the whole
% fleet, so the column would be a constant 100% and carry no information.
fprintf('  %-24s %16s %10s %14s %16s %14s\n', ...
        'Method', 'Objective', 'Gap [%]', 'Dist [km]', 'Price [$/kWh]', 'Wait [min]');
for k = 1:nMethods
    fprintf('  %-24s %8.1f+-%-6.1f %10.2f %7.3f+-%-5.3f %8.4f+-%-6.4f %7.2f+-%-5.2f\n', ...
        labels{k}, mObj(k), sObj(k), gap(k), mDist(k), sDist(k), ...
        mCost(k), sCost(k), mWait(k), sWait(k));
end

% Sanity check, printed rather than tabulated: if some method ever failed to
% serve the whole fleet, we would want to know about it.
if any(abs(100*mServ - 100) > 1e-9)
    fprintf('\n  NOTE: coverage is not always 100%%:  ');
    for k = 1:nMethods
        fprintf('%s %.1f%%  ', labels{k}, 100*mServ(k));
    end
    fprintf('\n');
end

%% ------------------------------------------------------------------
%  Ready-to-paste LaTeX table
%  ------------------------------------------------------------------

fprintf('\n\n=========== COPY THE BLOCK BELOW INTO THE PAPER ===========\n\n');
fprintf('%s\n', '\begin{table}[t]');
fprintf('%s\n', '\centering');
fprintf('%s\n', sprintf(['\\caption{Synthetic scenarios: the proposed model against the two ' ...
    'decentralised baselines, over %d independent instances with $M=%d$ ports and $N=%d$ ' ...
    'vehicles. Entries are mean $\\pm$ standard deviation across the instances. Produced by ' ...
    '\\texttt{synthetic\\_comparison.m}.}'], nValid, M, N));
fprintf('%s\n', '\label{tab:synthetic}');
fprintf('%s\n', '\small');
fprintf('%s\n', '\setlength{\tabcolsep}{4pt}');
fprintf('%s\n', '\begin{tabular}{lccccc}');
fprintf('%s\n', '\toprule');
fprintf('%s\n', ['\textbf{Method} & \textbf{Objective $\downarrow$} & \textbf{Gap [\%] $\downarrow$} & ' ...
                 '\textbf{Dist [km] $\downarrow$} & \textbf{Price [\$/kWh] $\downarrow$} & ' ...
                 '\textbf{Wait [min] $\downarrow$} \\']);
fprintf('%s\n', '\midrule');
for k = 1:nMethods
    fprintf('%s\n', sprintf(['%s & $%.1f \\pm %.1f$ & %.1f & $%.2f \\pm %.2f$ & ' ...
        '$%.4f \\pm %.4f$ & $%.2f \\pm %.2f$ \\\\'], ...
        labels{k}, mObj(k), sObj(k), gap(k), mDist(k), sDist(k), ...
        mCost(k), sCost(k), mWait(k), sWait(k)));
end
fprintf('%s\n', '\bottomrule');
fprintf('%s\n', '\end{tabular}');
fprintf('%s\n', '\end{table}');
fprintf('\n===========================================================\n\n');

%% ------------------------------------------------------------------
%  Figures: one boxplot per criterion, same size so they can be aligned
%  ------------------------------------------------------------------

fa = figure('Position', [80 100 560 460]);
boxplot_or_bar(objAcc(valid,:), shortLabels);
ylabel('Realised global objective');
title(sprintf('Objective (%d synthetic instances)', nValid));
grid on; box on; set(gca, 'FontSize', 11);

% Uncomment the next line to export this figure as a vector PDF
% exportgraphics(fa, 'Figures/synthetic_objective.pdf', 'ContentType', 'vector');

fb = figure('Position', [120 100 560 460]);
boxplot_or_bar(costAcc(valid,:), shortLabels);
ylabel('Mean price per EV [$/kWh]');
title(sprintf('Charging price (%d synthetic instances)', nValid));
grid on; box on; set(gca, 'FontSize', 11);

% Uncomment the next line to export this figure as a vector PDF
% exportgraphics(fb, 'Figures/synthetic_price.pdf', 'ContentType', 'vector');

fc = figure('Position', [160 100 560 460]);
boxplot_or_bar(distAcc(valid,:), shortLabels);
ylabel('Mean distance per EV [km]');
title(sprintf('Travel distance (%d synthetic instances)', nValid));
grid on; box on; set(gca, 'FontSize', 11);

% Uncomment the next line to export this figure as a vector PDF
% exportgraphics(fc, 'Figures/synthetic_distance.pdf', 'ContentType', 'vector');

fd = figure('Position', [200 100 560 460]);
boxplot_or_bar(waitAcc(valid,:), shortLabels);
ylabel('Mean waiting time per EV [min]');
title(sprintf('Waiting time (%d synthetic instances)', nValid));
grid on; box on; set(gca, 'FontSize', 11);

% Uncomment the next line to export this figure as a vector PDF
% exportgraphics(fd, 'Figures/synthetic_waiting.pdf', 'ContentType', 'vector');

if ~isempty(figdir)
    if ~exist(figdir, 'dir');  mkdir(figdir);  end
    try
        exportgraphics(fa, fullfile(figdir, 'synthetic_objective.pdf'), 'ContentType', 'vector');
        exportgraphics(fb, fullfile(figdir, 'synthetic_price.pdf'),     'ContentType', 'vector');
        exportgraphics(fc, fullfile(figdir, 'synthetic_distance.pdf'),  'ContentType', 'vector');
        exportgraphics(fd, fullfile(figdir, 'synthetic_waiting.pdf'),   'ContentType', 'vector');
        fprintf('Figures saved in %s\n', figdir);
    catch ME
        warning('Could not save the figures: %s', ME.message);
    end
end

results.M        = M;
results.N        = N;
results.labels   = labels;
results.objAcc   = objAcc;
results.distAcc  = distAcc;
results.costAcc  = costAcc;
results.waitAcc  = waitAcc;
results.servAcc  = servAcc;
results.timeAcc  = timeAcc;
results.valid    = valid;
results.meanObj  = mObj;   results.stdObj  = sObj;
results.meanDist = mDist;  results.stdDist = sDist;
results.meanCost = mCost;  results.stdCost = sCost;
results.meanWait = mWait;  results.stdWait = sWait;
results.meanServ = mServ;  results.stdServ = sServ;
results.meanTime = mTime;  results.stdTime = sTime;
results.gap      = gap;

end % ===================== end of main function =====================


%% ==================================================================
%  LOCAL FUNCTIONS
%  ==================================================================

function x = greedy_nearest(D, A, order)
% GREEDY_NEAREST  "Go to the nearest acceptable free port" baseline.

    [M, N] = size(D);
    if nargin < 3 || isempty(order);  order = randperm(N);  end

    taken = false(M,1);
    x     = zeros(M,N);
    for k = 1:N
        j    = order(k);
        cand = find(A(:,j) == 1 & ~taken);
        if isempty(cand);  continue;  end          % this EV stays unserved
        [~, kmin] = min(D(cand,j));
        x(cand(kmin), j)  = 1;
        taken(cand(kmin)) = true;
    end
end


function x = greedy_cheapest_k(D, C, A, k, order)
% GREEDY_CHEAPEST_K  "Cheapest among the k nearest free ports" baseline.

    [M, N] = size(D);
    if nargin < 5 || isempty(order);  order = randperm(N);  end

    taken = false(M,1);
    x     = zeros(M,N);
    for t = 1:N
        j    = order(t);
        cand = find(A(:,j) == 1 & ~taken);
        if isempty(cand);  continue;  end
        [~, srt] = sort(D(cand,j), 'ascend');
        near     = cand(srt(1:min(k, numel(cand))));
        [~, cmin] = min(C(near));
        x(near(cmin), j)  = 1;
        taken(near(cmin)) = true;
    end
end


function m = assignment_stats(x, c, D, W, C, N)
% ASSIGNMENT_STATS  Performance indicators of one assignment.
% Averages are over the SERVED vehicles; "served" is the fraction of the
% fleet that obtained a port.

    x  = full(double(x));
    Na = nnz(x);

    m.obj    = full(sum(sum(c .* x)));
    m.served = Na / N;

    if Na == 0
        [m.dist, m.cost, m.wait] = deal(NaN);
        return;
    end
    m.dist = full(sum(sum(D .* x))) / Na;
    m.cost = full(sum(C .* sum(x,2))) / Na;
    m.wait = full(sum(W .* sum(x,2))) / Na;
end


function [m, tmean] = avg_greedy(runFcn, ORDERS, c, D, W, C, N)
% AVG_GREEDY  Average a greedy policy over ORDERS random processing orders.
% A decentralised policy depends on who commits to a port first, so a single
% run would not be representative.

    acc = struct('obj',0, 'dist',0, 'cost',0, 'wait',0, 'served',0);
    ttot = 0;

    for o = 1:ORDERS
        t0 = tic;
        x  = runFcn();
        ttot = ttot + toc(t0);

        s = assignment_stats(x, c, D, W, C, N);
        acc.obj    = acc.obj    + s.obj;
        acc.dist   = acc.dist   + s.dist;
        acc.cost   = acc.cost   + s.cost;
        acc.wait   = acc.wait   + s.wait;
        acc.served = acc.served + s.served;
    end

    m.obj    = acc.obj    / ORDERS;
    m.dist   = acc.dist   / ORDERS;
    m.cost   = acc.cost   / ORDERS;
    m.wait   = acc.wait   / ORDERS;
    m.served = acc.served / ORDERS;
    tmean    = ttot / ORDERS;
end


function boxplot_or_bar(Y, labels)
% BOXPLOT_OR_BAR  Boxchart if available, otherwise the classic boxplot.

    nCol   = size(Y,2);
    groups = repmat(1:nCol, size(Y,1), 1);

    if exist('boxchart', 'file')
        boxchart(categorical(groups(:), 1:nCol, labels), Y(:));
    else
        boxplot(Y(:), groups(:), 'Labels', labels);
    end
end

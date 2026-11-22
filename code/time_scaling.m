function results = time_scaling(figdir, measureCvxOverhead)
% TIME_SCALING  Historical R2023b implementation benchmark.
% The dense Munkres routine and dense-input MATLAB matchpairs are distinct
% implementations. The LP/MILP paths here allocate all M*N variables. Use
% verify_assignment to benchmark identical allowed graphs across densities
% with eligible-edge LP/MILP models and dense/sparse matchpairs inputs.
% CVX timings include model construction. Its total time minus a direct
% solve measures a difference between end-to-end routes, not parsing alone.

if nargin < 1,  figdir = '';  end
if nargin < 2 || isempty(measureCvxOverhead),  measureCvxOverhead = true;  end

rng(1);


%% ------------------------------------------------------------------
%  Which solver interface is available
%  ------------------------------------------------------------------
% Optimization Toolbox gives the same mathematical LP/MILP without a
% third-party license. Keep the original MOSEK path as a fallback.
useMosek = exist('intlinprog','file') ~= 2;
if ~useMosek
    solverName = 'intlinprog / linprog (direct)';
elseif any(exist('mosekopt','file') == [2,3])
    solverName = 'Mosek (mosekopt, direct)';
else
    error('Install Optimization Toolbox or a licensed MOSEK solver.');
end

useCvx = measureCvxOverhead && (exist('cvx_begin', 'file') == 2);
if useCvx
    configuredSolver = '';
    try
        configuredSolver = cvx_solver;
        cvx_solver mosek
        cvx_solver_settings('MSK_IPAR_OPTIMIZER', 'MSK_OPTIMIZER_FREE_SIMPLEX')
    catch ME
        if ~isempty(configuredSolver) && ~strcmpi(cvx_solver, configuredSolver)
            cvx_solver(configuredSolver);
        end
        % CVX supports integer variables with these solvers. Numbered names
        % denote additional installations (for example, Gurobi_2).
        solverFamily = regexprep(lower(configuredSolver), '_[0-9]+$', '');
        useCvx = any(strcmp(solverFamily, {'gurobi', 'glpk', 'mosek'}));
        if useCvx
            warning('MOSEK setup failed (%s). Using CVX solver %s.', ME.message, configuredSolver);
        else
            warning('Skipping the optional CVX integer timing after MOSEK setup failed (%s).', ME.message);
        end
    end
end

use_matchpairs = exist('matchpairs', 'file') == 2;

%% ------------------------------------------------------------------
%  Simulation parameters
%  ------------------------------------------------------------------

N_vec = round(linspace(100, 800, 8));   % Number of electric vehicles
M_vec = round(1.2 * N_vec);             % Number of charging ports, M > N

REPS  = 3;      % repetitions per size, averaged
p_acc = 0.6;    % acceptance probability of the acceptable matrix A

alpha   = objective_weights();   % monetary coefficients, see objective_weights.m
Qdemand = 40;                    % energy demand of each EV [kWh]

Cmin = 0.303;  Cmax = 0.530;     % US state-level price bounds [$/kWh]

%% ------------------------------------------------------------------
%  Hardware and software, which the paper is expected to state
%  ------------------------------------------------------------------

env = struct();
env.arch    = computer('arch');
env.matlab  = version;
try,  env.cores = feature('numcores');  catch,  env.cores = NaN;  end
try,  env.threads = maxNumCompThreads;  catch,  env.threads = NaN;  end
env.solver  = solverName;

fprintf('\n=============== TIMING EXPERIMENT ===============\n');
fprintf('  platform      : %s, %d physical cores, %d computational threads\n', ...
        env.arch, env.cores, env.threads);
fprintf('  MATLAB        : %s\n', env.matlab);
fprintf('  MILP solver   : %s\n', env.solver);
fprintf('  CVX overhead  : %s\n', ternary_str(useCvx, 'measured', 'not measured'));
fprintf('  matchpairs    : %s\n', ternary_str(use_matchpairs, 'included', 'not found'));
fprintf('  sizes         : N from %d to %d, M = 1.2 N, %d repetitions\n', ...
        N_vec(1), N_vec(end), REPS);
fprintf('=================================================\n\n');

SIM = numel(N_vec);

% Results table, one row per size. The column map is used below and again
% when the table is displayed, so it is written out once here.
%  1 M | 2 N | 3 M*N
%  4 t_MILP_direct | 5 t_LP_direct | 6 t_hung | 7 t_matchpairs | 8 t_MILP_cvx
%  9 cost_MILP | 10 cost_LP | 11 cost_hung | 12 cost_matchpairs
% 13 costs agree | 14 t_munkres | 15 LP integrality residual
results_mat = nan(SIM, 15);

%% ------------------------------------------------------------------
%  Loop over the problem sizes
%  ------------------------------------------------------------------

for s = 1:SIM

    M = M_vec(s);
    N = N_vec(s);

    tMd = nan(REPS,1); tLP = nan(REPS,1); tH = nan(REPS,1);
    tS  = nan(REPS,1); tCvx = nan(REPS,1); tMunk = nan(REPS,1);
    cMd = nan(REPS,1); cLP = nan(REPS,1); cH = nan(REPS,1); cS = nan(REPS,1);
    intRes = nan(REPS,1);

    for r = 1:REPS

        % ---- instance
        D = rand(M,N)*10;                       % distances [km]
        W = rand(M,1)*15;                       % waiting times [min]
        C = Cmin + (Cmax - Cmin) * rand(M,1);   % tariffs [$/kWh]
        P = draw_port_power(M);                 % connector powers [kW]

        A = double(rand(M, N) < p_acc);         % acceptable matrix

        % All four solvers are compared on EV-saturating instances.
        [~, num_matches] = hopcroft_karp(A);
        if num_matches < N
            continue;
        end

        c = build_objective_matrix(D, W, C, P, Qdemand, alpha);

        % ---------------- MILP, called directly ----------------
        [xM, tMd(r)] = solve_direct(c, A, true,  useMosek);
        cMd(r) = sum(sum(c .* xM));

        % ---------------- LP relaxation, called directly ----------------
        % Total unimodularity says this optimum is attained at an integral
        % vertex. We do not take that on trust: the residual below measures
        % how far the returned solution is from {0,1}, and it is reported.
        [xL, tLP(r)] = solve_direct(c, A, false, useMosek);
        cLP(r)    = sum(sum(c .* xL));
        intRes(r) = max(min(abs(xL(:)), abs(1 - xL(:))));

        % ---------------- MILP through CVX, for the overhead ------------
        if useCvx
            tic
            cvx_begin quiet
                variable x(M,N) binary
                minimize( sum(sum( x .* c )) )
                subject to
                    sum(x,1)' == 1;
                    sum(x,2) <= 1;
                    x <= A;
            cvx_end
            assert(strcmp(cvx_status,'Solved'),'CVX solve failed: %s',cvx_status);
            tCvx(r) = toc;
        end

        % ---------------- dense Hungarian ----------------

        tic
        [assignments, ~, ~, run_time] = hungarian_assignment_binary(A, c);
        tH(r) = toc;
        tMunk(r) = run_time;

        assignments = full(double(assignments));
        assert(nnz(assignments) == N && ~any(~A(logical(assignments))), ...
               'The dense Hungarian used a forbidden pair: instance infeasible.');
        cH(r) = sum(sum(c .* assignments));

        % ---------------- dense-input matchpairs ----------------
        if use_matchpairs
            tic
            assignments_builtin = matchpairs_assignment(A, c);
            tS(r) = toc;
            assignments_builtin = full(double(assignments_builtin));
            assert(nnz(assignments_builtin) == N && ~any(~A(logical(assignments_builtin))), ...
                   'The dense-input matchpairs used a forbidden pair.');
            cS(r) = sum(sum(c .* assignments_builtin));
        end
    end

    results_mat(s,1)  = M;
    results_mat(s,2)  = N;
    results_mat(s,3)  = M*N;
    results_mat(s,4)  = mean(tMd,  'omitnan');
    results_mat(s,5)  = mean(tLP,  'omitnan');
    results_mat(s,6)  = mean(tH,   'omitnan');
    results_mat(s,7)  = mean(tS,   'omitnan');
    results_mat(s,8)  = mean(tCvx, 'omitnan');
    results_mat(s,9)  = mean(cMd,  'omitnan');
    results_mat(s,10) = mean(cLP,  'omitnan');
    results_mat(s,11) = mean(cH,   'omitnan');
    results_mat(s,12) = mean(cS,   'omitnan');
    results_mat(s,14) = mean(tMunk,'omitnan');
    results_mat(s,15) = max(intRes);

    % Do all the methods agree on the optimal cost, at every repetition?
    agree = max(abs(cMd - cH)) < 1e-6 && max(abs(cMd - cLP)) < 1e-6;
    if use_matchpairs
        agree = agree && (max(abs(cMd - cS)) < 1e-6);
    end
    results_mat(s,13) = agree;

    fprintf(['M = %4d, N = %4d | MILP %7.3f s | LP %7.3f s | Hung %7.3f s'], ...
            M, N, results_mat(s,4), results_mat(s,5), results_mat(s,6));
    if use_matchpairs
        fprintf(' | matchpairs %7.3f s', results_mat(s,7));
    end
    if useCvx
        fprintf(' | CVX %7.3f s', results_mat(s,8));
    end
    fprintf(' | agree %d\n', agree);
end

%% ------------------------------------------------------------------
%  Empirical growth exponents
%  ------------------------------------------------------------------
% Fitting log(time) = a + b*log(M*N) gives an exponent b that can be read
% straight off the slope of the log-log plot. Note that the exponent is in
% M*N, not in n = max{M,N}: the cubic bound of the Hungarian method
% therefore corresponds to a slope of about 1.5 here, not 3.

xv = results_mat(:,3);

pM = fit_slope(xv, results_mat(:,4));
pL = fit_slope(xv, results_mat(:,5));
pH = fit_slope(xv, results_mat(:,6));
pS = NaN;  if use_matchpairs,  pS = fit_slope(xv, results_mat(:,7));  end

fprintf('\nEmpirical growth in M*N:\n');
fprintf('  MILP (direct)     ~ (M N)^%.2f\n', pM);
fprintf('  LP relaxation     ~ (M N)^%.2f\n', pL);
fprintf('  Hungarian (dense) ~ (M N)^%.2f\n', pH);
if use_matchpairs
    fprintf('  matchpairs (dense input)~ (M N)^%.2f\n', pS);
end

fprintf('\nLargest deviation of the LP solution from {0,1}: %.2e\n', ...
        max(results_mat(:,15)));
fprintf('(total unimodularity predicts an integral vertex; this measures it)\n');

if useCvx
    ovh = 1 - results_mat(:,4) ./ results_mat(:,8);
    fprintf(['\nDifference between CVX and direct wall times as a share of ' ...
             'CVX time: %.0f%% to %.0f%%\n'], 100*min(ovh), 100*max(ovh));
end

%% ------------------------------------------------------------------
%  FIGURE 1: solve time versus problem size, log-log
%  ------------------------------------------------------------------

f1 = figure('Position', [100 100 660 470]);
hold on; grid on; box on;

plot(xv, results_mat(:,4), '-o', 'LineWidth', 1.8, 'MarkerSize', 7, ...
     'MarkerFaceColor', 'w', ...
     'DisplayName', sprintf('MILP, direct call (slope %.2f)', pM));
plot(xv, results_mat(:,5), '-v', 'LineWidth', 1.8, 'MarkerSize', 7, ...
     'MarkerFaceColor', 'w', ...
     'DisplayName', sprintf('LP relaxation (slope %.2f)', pL));
plot(xv, results_mat(:,6), '-s', 'LineWidth', 1.8, 'MarkerSize', 7, ...
     'MarkerFaceColor', 'w', ...
     'DisplayName', sprintf('Hungarian, dense (slope %.2f)', pH));
if use_matchpairs
    plot(xv, results_mat(:,7), '-^', 'LineWidth', 1.8, 'MarkerSize', 7, ...
         'MarkerFaceColor', 'w', ...
         'DisplayName', sprintf('matchpairs, dense input (slope %.2f)', pS));
end
if useCvx
    % Dashed and grey, because it is a different kind of quantity: it
    % includes the modeling layer and is shown only to expose it.
    plot(xv, results_mat(:,8), '--', 'Color', [0.55 0.55 0.55], ...
         'LineWidth', 1.4, 'DisplayName', 'MILP through CVX (parsing included)');
end

set(gca, 'XScale', 'log', 'YScale', 'log');
xlabel('Problem size M \times N');
ylabel('Mean solve time [s]');
title('Computational time versus problem size');
legend('Location', 'northwest');
set(gca, 'FontSize', 11);

%% ------------------------------------------------------------------
%  FIGURE 2: optimal cost versus problem size
%  ------------------------------------------------------------------

okIdx  = results_mat(:,13) == 1;
badIdx = results_mat(:,13) == 0;

f2 = figure('Position', [100 100 660 470]);
hold on; grid on; box on;

plot(xv, results_mat(:,9),  'b-',  'LineWidth', 1.6, 'DisplayName', 'MILP');
plot(xv, results_mat(:,10), 'm-.', 'LineWidth', 1.6, 'DisplayName', 'LP relaxation');
plot(xv, results_mat(:,11), 'k--', 'LineWidth', 1.6, 'DisplayName', 'Hungarian, dense');
plot(xv(okIdx), results_mat(okIdx,11), 'ko', 'MarkerSize', 5, ...
     'MarkerFaceColor', 'k', 'DisplayName', 'all methods agree');
if any(badIdx)
    plot(xv(badIdx), results_mat(badIdx,11), 'rx', 'MarkerSize', 10, ...
         'LineWidth', 1.8, 'DisplayName', 'MISMATCH');
end
if use_matchpairs
    plot(xv, results_mat(:,12), 'g:', 'LineWidth', 1.8, ...
         'DisplayName', 'matchpairs, dense input');
end

xlabel('Problem size M \times N');
ylabel('Optimal objective value [$]');
title('Optimal cost versus problem size (the curves must coincide)');
legend('Location', 'northwest');
set(gca, 'FontSize', 11);

relGap  = abs(results_mat(:,9) - results_mat(:,11)) ./ abs(results_mat(:,9));
relGapL = abs(results_mat(:,9) - results_mat(:,10)) ./ abs(results_mat(:,9));
fprintf('\nWorst relative cost gap, MILP vs dense Hungarian : %.2e\n', max(relGap));
fprintf('Worst relative cost gap, MILP vs LP relaxation   : %.2e\n', max(relGapL));
if use_matchpairs
    relGapS = abs(results_mat(:,9) - results_mat(:,12)) ./ abs(results_mat(:,9));
    fprintf('Worst relative cost gap, MILP vs dense-input matchpairs: %.2e\n', max(relGapS));
end

%% ------------------------------------------------------------------
%  FIGURE 3: speed-up, and the size of the modeling overhead
%  ------------------------------------------------------------------

f3 = figure('Position', [100 100 1000 400]);

subplot(1,2,1);
hold on; grid on; box on;
plot(xv, results_mat(:,4) ./ results_mat(:,6), '-d', 'LineWidth', 1.8, ...
     'MarkerSize', 7, 'MarkerFaceColor', 'w', 'DisplayName', 'dense Hungarian');
if use_matchpairs
    plot(xv, results_mat(:,4) ./ results_mat(:,7), '-^', 'LineWidth', 1.8, ...
         'MarkerSize', 7, 'MarkerFaceColor', 'w', 'DisplayName', 'dense-input matchpairs');
end
plot(xv, results_mat(:,4) ./ results_mat(:,5), '-v', 'LineWidth', 1.4, ...
     'MarkerSize', 6, 'MarkerFaceColor', 'w', 'DisplayName', 'LP relaxation');
yline(1, 'k--', 'HandleVisibility', 'off');
set(gca, 'XScale', 'log', 'YScale', 'log');
xlabel('Problem size M \times N');
ylabel('Speed-up over the direct MILP');
title('Speed-up, against the MILP timed without its modeling layer');
legend('Location', 'northwest');
set(gca, 'FontSize', 11);

subplot(1,2,2);
if useCvx
    bar(1:SIM, 100*(1 - results_mat(:,4)./results_mat(:,8)));
    set(gca, 'XTick', 1:SIM, 'XTickLabel', compose('%d', results_mat(:,2)));
    grid on;  ylim([0 100]);
    xlabel('Number of vehicles N');
    ylabel('CVX minus direct [% of CVX time]');
    title('Difference between timing protocols');
else
    axis off;
    text(0.5, 0.5, {'CVX overhead not measured', ...
         '(run with measureCvxOverhead = true)'}, ...
         'HorizontalAlignment','center');
end
set(gca, 'FontSize', 11);

%% ------------------------------------------------------------------
%  Save and collect the results
%  ------------------------------------------------------------------

if ~isempty(figdir)
    if ~isfolder(figdir),  mkdir(figdir);  end
    try
        exportgraphics(f1, fullfile(figdir, 'time_vs_size.pdf'), 'ContentType', 'vector');
        exportgraphics(f2, fullfile(figdir, 'cost_vs_size.pdf'), 'ContentType', 'vector');
        exportgraphics(f3, fullfile(figdir, 'speedup.pdf'),      'ContentType', 'vector');
        fprintf('\nFigures saved in %s\n', figdir);
    catch ME
        warning('Could not save the figures: %s', ME.message);
    end
end

% Historical table field t_Sparse (and cost_Sparse) refers to dense-input matchpairs.
T = array2table(results_mat, 'VariableNames', ...
    {'M','N','MN','t_MILP','t_LP','t_Hung','t_Sparse','t_CVX', ...
     'cost_MILP','cost_LP','cost_Hung','cost_Sparse','agree', ...
     't_munkres','LP_int_residual'});
displayTable = T;
displayTable.Properties.VariableNames{'t_Sparse'} = 't_MatchpairsDense';
displayTable.Properties.VariableNames{'cost_Sparse'} = 'cost_MatchpairsDense';
disp(displayTable);

results.table          = T;
results.raw            = results_mat;
results.slope_MILP     = pM;
results.slope_LP       = pL;
results.slope_Hung     = pH;
results.slope_MatchpairsDense = pS;
results.slope_Sparse   = pS;
results.use_matchpairs = use_matchpairs;
results.use_sparse = use_matchpairs; % backwards-compatible name for this implementation
results.cvx_measured   = useCvx;
results.REPS           = REPS;
results.environment    = env;

end


%% ==================================================================
%  LOCAL FUNCTIONS
%  ==================================================================

function [x, t] = solve_direct(c, A, integer, useMosek)
% SOLVE_DIRECT  Solve the assignment problem through the solver's own API.
%
%   min sum_ij c_ij x_ij
%   s.t. sum_i x_ij = 1  (all j),  sum_j x_ij <= 1  (all i),  0 <= x <= A
%
% The acceptability constraint is imposed as an upper BOUND rather than as
% a row, which is both cheaper and closer to what the model says: a
% forbidden pair is a variable that cannot be used, not an inequality to be
% satisfied. This is also why no big-M appears anywhere on this path.
%
% Only the solver call is timed. Building the constraint matrix is part of
% setting up the experiment, not of solving, and including it would repeat
% the very mistake this function exists to avoid.

    [M, N] = size(c);
    MN = M * N;

    % Variable k corresponds to the pair (i,j) with k = (j-1)*M + i, that
    % is to the column-major ordering of an M-by-N matrix, so that x(:)
    % and the solver's vector are the same object.
    [ii, jj] = ndgrid(1:M, 1:N);
    ii = ii(:);  jj = jj(:);
    k  = (1:MN)';

    % Rows 1..N are the EV equalities, rows N+1..N+M the port capacities.
    Amat = sparse([jj; N + ii], [k; k], 1, N + M, MN);
    blc  = [ones(N,1); -inf(M,1)];
    buc  = [ones(N,1);  ones(M,1)];
    blx  = zeros(MN,1);
    bux  = double(A(:));          % forbidden pairs get an upper bound of 0

    if useMosek
        prob.c   = c(:);
        prob.a   = Amat;
        prob.blc = blc;   prob.buc = buc;
        prob.blx = blx;   prob.bux = bux;

        param = struct();
        if integer
            prob.ints.sub = find(bux > 0)';   % the fixed ones need no branching
            cmd = 'minimize echo(0)';
        else
            param.MSK_IPAR_OPTIMIZER = 'MSK_OPTIMIZER_FREE_SIMPLEX';
            cmd = 'minimize echo(0)';
        end

        t0 = tic;
        [status, res] = mosekopt(cmd, prob, param);
        t = toc(t0);
        assert(status == 0 && isfield(res,'sol'),'MOSEK did not return a solution.');
        if integer
            sol = res.sol.int;
        elseif isfield(res.sol,'bas')
            sol = res.sol.bas;
        else
            sol = res.sol.itr;
        end
        assert(any(strcmp(sol.solsta,{'OPTIMAL','INTEGER_OPTIMAL'})), ...
               'MOSEK did not certify an optimal objective.');
        xx = sol.xx;
    else
        opts = optimoptions('intlinprog', 'Display', 'off');
        Aeq  = Amat(1:N, :);        beq = ones(N,1);
        Aub  = Amat(N+1:end, :);    bub = ones(M,1);

        if integer
            intcon = find(bux > 0)';
            t0 = tic;
            [xx,~,flag] = intlinprog(c(:), intcon, Aub, bub, Aeq, beq, blx, bux, opts);
            t  = toc(t0);
        else
            lopts = optimoptions('linprog', 'Display', 'off');
            t0 = tic;
            [xx,~,flag] = linprog(c(:), Aub, bub, Aeq, beq, blx, bux, lopts);
            t  = toc(t0);
        end
        assert(flag == 1,'The direct solver did not certify an optimum.');
    end

    x = reshape(full(xx), M, N);
    if integer
        x = round(x);
    end
    assert(~any(x(~A)>1e-6) && all(abs(sum(x,1)-1)<1e-6) && all(sum(x,2)<=1+1e-6), ...
           'Direct solution violates the assignment constraints.');
end


function b = fit_slope(xv, yv)
% FIT_SLOPE  Empirical growth exponent, from a log-log least-squares fit.

    valid = isfinite(xv) & isfinite(yv) & xv > 0 & yv > 0;
    if nnz(valid) < 2
        b = NaN;
        return;
    end
    p = polyfit(log(xv(valid)), log(yv(valid)), 1);
    b = p(1);
end


function s = ternary_str(cond, a, b)
    if cond,  s = a;  else,  s = b;  end
end

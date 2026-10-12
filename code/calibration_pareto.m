function R = calibration_pareto(figdir)
% CALIBRATION_PARETO  Pick the two free coefficients of the objective (Sec. 3.2).
%
% Sweeps a logarithmic grid over (alpha_1, alpha_3) with alpha_2 = 1, keeps
% the non-dominated settings and takes the knee of the frontier. Writes
% calibrated_alpha.mat, which every other script reads.
%
% The grid bounds are deliberately narrow. The knee is found after min-max
% normalising the criteria, so it is NOT invariant to the bounds: widening
% them to economically meaningless settings drags the ideal point and the
% knee with it. What the bounds encode is the range one can actually defend.
%
% Produces Figures/calibration_pareto.pdf and prints a paste-ready LaTeX
% table with the knee and the three single-criterion corners.


if nargin < 1;  figdir = 'Figures';  end
if ~isempty(figdir) && ~isfolder(figdir)
    [okDir, msgDir] = mkdir(figdir);
    if ~okDir, error('Cannot create %s: %s', figdir, msgDir); end
end

rng(1);

%% ---------------------------------------------------------------- settings
M     = 500;    % charging ports
N     = 250;    % electric vehicles
REPS  = 15;     % scenarios, SHARED by every coefficient setting
p_acc = 0.6;    % acceptance probability of the acceptable matrix A

Cmin = 0.303;  Cmax = 0.530;   % [$/kWh]
Qmin = 20;     Qmax = 60;      % [kWh]

% The grid. Two dimensions instead of four, so a single dense pass replaces
% the coarse-plus-refined procedure of the earlier version.
%
% THE RANGES ARE DELIBERATELY NARROW. The knee point is obtained by min-max
% normalising each criterion over the sampled set, so it is not invariant to
% the bounds of the sweep: including economically meaningless settings would
% stretch the utopia point and drag the knee with it. The ranges below span
% what can actually be defended. A kilometre of urban driving costs between
% a quarter of a dollar, counting energy alone, and about four dollars if one
% charges the driver's time generously; a minute out of service is worth
% between 3 and 60 $/h. Widening them does not add information, it only
% dilutes the compromise.
a1_vals = logspace(log10(0.25), log10(4.0), 13);   % alpha_1 [$/km]
a3_vals = logspace(log10(0.05), log10(1.0), 13);   % alpha_3 [$/min]

if exist('hungarian_assignment_sparse', 'file') == 2
    solve_fun = @hungarian_assignment_sparse;
else
    solve_fun = @hungarian_assignment_binary;
end

fprintf('\n=== Pareto calibration of (alpha_1, alpha_3), alpha_2 = 1 ===\n');
fprintf('  grid      : %d x %d = %d settings\n', ...
        numel(a1_vals), numel(a3_vals), numel(a1_vals)*numel(a3_vals));
fprintf('  alpha_1   : %.2f to %.2f $/km\n', min(a1_vals), max(a1_vals));
fprintf('  alpha_3   : %.2f to %.2f $/min  (%.0f to %.0f $/h)\n', ...
        min(a3_vals), max(a3_vals), 60*min(a3_vals), 60*max(a3_vals));
fprintf('  instances : %d, shared across all settings (M = %d, N = %d)\n\n', REPS, M, N);

%% ------------------------------------------------- pre-draw the instances
% Every setting is evaluated on the SAME scenarios. This is not a detail: it
% removes scenario-to-scenario variability from the comparison, so that the
% differences between points of the frontier are caused by the coefficients
% alone. Infeasible draws are discarded once, here, rather than inside the
% grid loop, so that all settings see exactly the same population.
inst = {};
tries = 0;
while numel(inst) < REPS && tries < 5*REPS
    tries = tries + 1;
    D = rand(M,N)*10;                        % [km]
    W = rand(M,1)*15;                        % [min]
    C = Cmin + (Cmax-Cmin)*rand(M,1);        % [$/kWh]
    P = draw_port_power(M);         % [kW]
    Q = Qmin + (Qmax-Qmin)*rand(1,N);        % [kWh]
    A = double(rand(M,N) < p_acc);
    [~, nm] = hopcroft_karp(A);
    if nm < N,  continue;  end               % some vehicle cannot be served
    inst{end+1} = struct('D',D,'W',W,'C',C,'P',P,'Q',Q,'A',A); %#ok<AGROW>
end
assert(numel(inst) == REPS, 'Could not draw %d feasible scenarios.', REPS);

%% ------------------------------------------------------------- the grid
nS   = numel(a1_vals)*numel(a3_vals);
a1c  = zeros(nS,1);  a3c = zeros(nS,1);
Dbar = zeros(nS,1);  Sbar = zeros(nS,1);  Tbar = zeros(nS,1);

k = 0;
for ia = 1:numel(a1_vals)
    for ib = 1:numel(a3_vals)
        k = k + 1;
        alpha = struct('dist', a1_vals(ia), 'price', 1, 'time', a3_vals(ib));
        [d, s, t] = evaluate(alpha, inst, solve_fun);
        a1c(k) = a1_vals(ia);  a3c(k) = a3_vals(ib);
        Dbar(k) = d;  Sbar(k) = s;  Tbar(k) = t;
    end
    fprintf('  alpha_1 = %7.3f done\n', a1_vals(ia));
end

T = table(a1c, a3c, Dbar, Sbar, Tbar, ...
    'VariableNames', {'alpha1','alpha3','meanDist','meanSpend','meanTime'});

%% -------------------------------------------- Pareto filter and knee point
[ParetoSet, Knee, isDom, paretoIdx, kneeIdx] = ...
    pareto_knee_from_table(T, {'meanDist','meanSpend','meanTime'}, false);

nDom = nS - numel(paretoIdx);
fprintf('\n  Pareto-optimal settings : %d of %d\n', numel(paretoIdx), nS);
if nDom == 0
    fprintf(['      (all of them, as expected: the objective is a positively\n' ...
             '       weighted sum of exactly these three criteria, so every\n' ...
             '       setting is non-dominated by construction)\n']);
end
fprintf('  knee point              : alpha_1 = %.3f $/km, alpha_3 = %.3f $/min\n', ...
        Knee.alpha1, Knee.alpha3);
fprintf('      dist %.4f km, spend %.3f $, time %.2f min\n', ...
        Knee.meanDist, Knee.meanSpend, Knee.meanTime);

%% ------------------------------------------------------------- the figure
% Projection onto the spend-distance plane, coloured by the time out of
% service, in the same style as the figure of the earlier version.
f = figure('Position', [90 90 760 560]);
hold on; grid on; box on;

if any(isDom)
    scatter(Sbar(isDom), Dbar(isDom), 26, Tbar(isDom), 'filled', ...
            'MarkerFaceAlpha', 0.20, 'DisplayName', 'Dominated configurations');
end
scatter(Sbar(paretoIdx), Dbar(paretoIdx), 85, Tbar(paretoIdx), 'filled', ...
        'MarkerEdgeColor', 'k', 'LineWidth', 0.8, ...
        'DisplayName', 'Configurations on the frontier');
plot(Knee.meanSpend, Knee.meanDist, 'p', 'MarkerSize', 20, ...
     'MarkerFaceColor', [0.95 0.85 0.15], 'MarkerEdgeColor', 'k', ...
     'LineWidth', 1.2, 'DisplayName', 'Knee point');
cb = colorbar;  cb.Label.String = 'Average time out of service [min]';
colormap(turbo);
xlabel('Average expenditure per EV [$]');
ylabel('Average travel distance per EV [km]');
title('The trade-off frontier in the spend--distance plane');
legend('Location', 'best');
set(gca, 'FontSize', 11);

% ---- export ------------------------------------------------------------
% Saved as a vector PDF, tightly cropped, ready for \includegraphics.
if ~isempty(figdir)
    pdfPath = fullfile(figdir, 'calibration_pareto.pdf');
    try
        exportgraphics(f, pdfPath, 'ContentType', 'vector');
        fprintf('\nFigure written to %s\n', fullfile(pwd, pdfPath));
    catch ME
        warning('Could not save the figure: %s', ME.message);
    end
end

%% --------------------------------------------------- LaTeX-ready table
% Four rows: the three settings that each minimise one criterion, which mark
% the corners of the frontier, and the knee point that balances them. Printed
% as a complete table environment, so it can be pasted into the paper as is.

[~, iD] = min(Dbar);          % the setting that travels least
[~, iS] = min(Sbar);          % the setting that pays least
[~, iT] = min(Tbar);          % the setting that queues and charges least
rows    = [iD, iS, iT];
names   = {'Shortest trips', 'Cheapest energy', 'Fastest service'};

fprintf('\n%% ================= paste into the paper =================\n');
fprintf('\\begin{table}[t]\n\\centering\n');
fprintf(['\\caption{Calibration of the objective coefficients on %d synthetic ' ...
         'instances with $M=%d$ ports and $N=%d$ vehicles, at $\\alpha_2=1$. ' ...
         'The first three rows are the settings of the grid that minimise each ' ...
         'criterion on its own and mark the corners of the trade-off frontier; ' ...
         'the last is the knee point, the configuration closest to the ' ...
         'normalised utopia, which is the one adopted in all the experiments ' ...
         'that follow.}\n'], REPS, M, N);
fprintf('\\label{tab:calibration}\n');
fprintf('\\begin{tabular}{lccccc}\n\\toprule\n');
fprintf(['\\textbf{Configuration} & $\\alpha_1$ [\\$/km] & $\\alpha_3$ [\\$/min] & ' ...
         '$\\bar D$ [km] & $\\bar S$ [\\$] & $\\bar T$ [min] \\\\\n']);
fprintf('\\midrule\n');
for r = 1:3
    fprintf('%s & %.2f & %.3f & %.4f & %.2f & %.1f \\\\\n', names{r}, ...
            T.alpha1(rows(r)), T.alpha3(rows(r)), ...
            T.meanDist(rows(r)), T.meanSpend(rows(r)), T.meanTime(rows(r)));
end
fprintf('\\midrule\n');
fprintf(['\\textbf{Knee point (adopted)} & $\\mathbf{%.2f}$ & $\\mathbf{%.3f}$ & ' ...
         '$\\mathbf{%.4f}$ & $\\mathbf{%.2f}$ & $\\mathbf{%.1f}$ \\\\\n'], ...
        Knee.alpha1, Knee.alpha3, Knee.meanDist, Knee.meanSpend, Knee.meanTime);
fprintf('\\bottomrule\n\\end{tabular}\n\\end{table}\n');
fprintf('%% ========================================================\n\n');

%% ------------------------------------------- hand the result to the suite
% The knee point is written to disk, and objective_weights.m picks it up
% automatically from there. Without this step the rest of the experiments
% would silently keep running on the literature values, and the claim that
% the calibrated setting is the one adopted everywhere else would be false.
alphaStar = struct('dist', Knee.alpha1, 'price', 1, 'time', Knee.alpha3, ...
                   'source', 'calibration_pareto knee point', ...
                   'date', datestr(now, 'yyyy-mm-dd HH:MM'), ...
                   'M', M, 'N', N, 'REPS', REPS);
save('calibrated_alpha.mat', 'alphaStar');
fprintf('\nWritten: %s\n', fullfile(pwd, 'calibrated_alpha.mat'));
fprintf(['Calibrated coefficients:\n' ...
         '  alpha_1 = %.3f $/km, alpha_2 = 1, alpha_3 = %.3f $/min\n' ...
         'Every script that calls objective_weights() will now use them.\n' ...
         'Delete the file to fall back on the literature values.\n\n'], ...
        Knee.alpha1, Knee.alpha3);

R = struct('table', T, 'pareto', ParetoSet, 'knee', Knee, ...
           'kneeIdx', kneeIdx, 'paretoIdx', paretoIdx, ...
           'corners', T(rows,:), 'M', M, 'N', N, 'REPS', REPS);
end

% =====================================================================

function [dBar, sBar, tBar] = evaluate(alpha, inst, solve_fun)
% Average the three criteria over the shared scenarios, for one setting of
% the coefficients. The criteria are what the driver experiences and do not
% depend on alpha except through the assignment that alpha produces.

    REPS = numel(inst);
    d = zeros(REPS,1);  s = zeros(REPS,1);  t = zeros(REPS,1);

    for r = 1:REPS
        I = inst{r};
        N = numel(I.Q);

        c = build_objective_matrix(I.D, I.W, I.C, I.P, I.Q, alpha);
        H = c;  H(~I.A) = 1e6;                  % forbid the non-acceptable pairs
        % NOTE the argument order: the solver takes the acceptable matrix
        % FIRST and the cost matrix second, as everywhere else in this suite.
        x = full(double(solve_fun(I.A, H)));

        % The forbidden pairs carry a large FINITE penalty, so an infeasible
        % instance does not make the solver fail: it returns a complete
        % assignment that uses forbidden pairs at cost 1e6. Cardinality is
        % therefore not a feasibility test, and the second condition is the
        % one that actually detects the problem.
        if nnz(x) < N || any(~I.A(logical(x)))
            error(['The solver returned an assignment using forbidden pairs, ' ...
                   'or served\nfewer than %d vehicles, although the instance ' ...
                   'was checked with\nhopcroft_karp before being accepted. ' ...
                   'Check the argument order of the solver.'], N);
        end

        Na  = nnz(x);
        sel = zeros(N,1);
        for j = 1:N
            i0 = find(x(:,j), 1);
            if ~isempty(i0),  sel(j) = i0;  end
        end
        ok  = sel > 0;
        idx = sub2ind(size(I.D), sel(ok), find(ok));

        d(r) = sum(I.D(idx)) / Na;                                  % [km]
        s(r) = sum(I.C(sel(ok)) .* I.Q(ok)') / Na;                  % [$]
        t(r) = sum(I.W(sel(ok)) + 60*I.Q(ok)'./I.P(sel(ok))) / Na;  % [min]
    end

    dBar = mean(d);  sBar = mean(s);  tBar = mean(t);
end

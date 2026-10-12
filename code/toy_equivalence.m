function results = toy_equivalence(figdir)
% TOY_EQUIVALENCE  MILP and Hungarian give the same answer (Sec. 3.4.1).
%
% Solves two small instances both ways and plots the assignment matrices
% side by side, then repeats over a batch of random instances and checks
% that the optimal costs agree to solver tolerance.
%
% The assignments agree too, not just the costs, which is more than total
% unimodularity guarantees: the cost coefficients are continuous, so two
% matchings tie with probability zero and the optimum is almost surely
% unique. With quantised data (a tariff menu, rounded distances) the two
% methods could legitimately return different, equally optimal assignments.
%
% Produces toy_assignment_20x20.pdf and toy_assignment_50x30.pdf.


cvx_solver mosek % Set mosek solver which is faster than default
cvx_solver_settings('MSK_IPAR_OPTIMIZER', 'MSK_OPTIMIZER_FREE_SIMPLEX')

rng(1); % set seed

BIGM = 1e6;   % penalty used to forbid the non-acceptable pairs

if nargin < 1;  figdir = 'Figures';  end
if ~isempty(figdir) && ~isfolder(figdir)
    [okDir, msgDir] = mkdir(figdir);
    if ~okDir, error('Cannot create %s: %s', figdir, msgDir); end
end
TOL  = 1e-6;  % tolerance under which two costs are declared identical

% Optimization weights (calibrated knee point, Section 3.2)
% Monetary coefficients of the objective; see objective_weights.m.
alpha   = objective_weights();
Qdemand = 40;   % energy demand of each EV [kWh]

% Charging prices are uniform between the min and max US state-level price
Cmin = 0.303;  % [$/kWh]
Cmax = 0.530;  % [$/kWh]

%% ------------------------------------------------------------------
%  PART 1: one toy instance, fully displayed
%  ------------------------------------------------------------------

fprintf('\n================ TOY INSTANCE (M = 5, N = 3) ================\n\n');

M = 5;
N = 3;

% Simulate charging port parameters (rounded, so the matrices are readable)
D = round(rand(M,N)*10, 2);                    % Distance port i to vehicle j [km]
W = round(rand(M,1)*15, 2);                    % Waiting time at each port [min]
C = round(Cmin + (Cmax-Cmin)*rand(M,1), 3);    % Charging cost at each port [$/kWh]
P = draw_port_power(M);               % Charging power of each port [kW]

% Draw the acceptable matrix until the toy instance admits a full
% assignment: a toy example that cannot serve every EV would be useless.
A = double(rand(M,N) < 0.7);
[~, num_matches] = hopcroft_karp(A);
while num_matches < N
    A = double(rand(M,N) < 0.7);
    [~, num_matches] = hopcroft_karp(A);
end

% Objective function term "c"
c = build_objective_matrix(D, W, C, P, Qdemand, alpha);

% ---- MILP
tic
cvx_begin quiet
    % Using Mosek solver, we are able to solve MILP problems
    variable x(M,N) binary

    % Objective function
    minimize( sum(sum( x .* c )) )

    subject to
        % Each EV is assigned to exactly one port
        sum(x,1)' == 1;
        % Each port serves at most one EV
        sum(x,2) <= 1;
        % Acceptable matrix constraint
        x <= A;
cvx_end
time_milp = toc;

% CVX returns x as a SPARSE matrix: densify it before using or comparing it
x = full(round(x));
cost_milp = cvx_optval;   % cvx_optval is already a plain double

% ---- Hungarian: non-acceptable pairs are forbidden with a large penalty
H = c;
H(~A) = BIGM;
tic
[assignments, ~, ~, ~] = hungarian_assignment_binary(A, H);
time_hung = toc;

% munkres may return a sparse or logical matrix: densify it as well
assignments = full(double(assignments));
cost_hung   = sum(sum(c .* assignments));

sameCost   = abs(cost_milp - cost_hung) < TOL;
sameAssign = isequal(x, assignments);

% ---- display the instance
disp('Acceptable matrix A =');   disp(A);
disp('Edge-cost matrix c =');    disp(round(c,3));
disp('MILP assignment x =');     disp(x);
disp('Hungarian assignment =');  disp(assignments);

% ---- TABLE 1
fprintf('----------------------------------------------------------\n');
fprintf('  %-22s %14s %14s\n', '', 'MILP', 'Hungarian');
fprintf('----------------------------------------------------------\n');
fprintf('  %-22s %14.6f %14.6f\n', 'Optimal cost',    cost_milp, cost_hung);
fprintf('  %-22s %14.4f %14.4f\n', 'Solve time [s]',  time_milp, time_hung);
fprintf('  %-22s %14s %14s\n', 'Same cost?',       bool2str(sameCost),   '');
fprintf('  %-22s %14s %14s\n', 'Same assignment?', bool2str(sameAssign), '');
fprintf('  %-22s %14.2e\n', 'Cost difference', abs(cost_milp - cost_hung));
fprintf('----------------------------------------------------------\n');

%% ------------------------------------------------------------------
%  PART 1b: visual validation on two larger, still readable instances
%  ------------------------------------------------------------------
% The toy instance above is small enough to print, but a 5 x 3 matrix is not
% very convincing to look at. Here the same comparison is repeated on two
% instances that are still small enough to be displayed as images: a square
% one, where every port is used, and a rectangular one, where the optimum
% has spare ports to choose from. Each panel shows the assignment matrix,
% one filled cell per vehicle; the two panels of a pair should be identical.

visSizes = [20 20; 50 30];      % [M N], as displayed in the paper
visNames = {'20x20', '50x30'};
results.visual = struct([]);

for v = 1:size(visSizes,1)
    Mv = visSizes(v,1);
    Nv = visSizes(v,2);

    % Draw until the instance can serve every vehicle
    Dv = rand(Mv,Nv)*10;
    Wv = rand(Mv,1)*15;
    Cv = Cmin + (Cmax-Cmin)*rand(Mv,1);
    Pv = draw_port_power(Mv);
    Qv = 20 + 40*rand(1,Nv);
    Av = double(rand(Mv,Nv) < 0.7);
    [~, nm] = hopcroft_karp(Av);
    while nm < Nv
        Av = double(rand(Mv,Nv) < 0.7);
        [~, nm] = hopcroft_karp(Av);
    end

    cv = build_objective_matrix(Dv, Wv, Cv, Pv, Qv, alpha);

    cvx_begin quiet
        variable xv(Mv,Nv) binary
        minimize( sum(sum( xv .* cv )) )
        subject to
            sum(xv,1)' == 1;
            sum(xv,2)  <= 1;
            xv <= Av;
    cvx_end
    xv = full(round(xv));
    costM = cvx_optval;

    Hv = cv;  Hv(~Av) = BIGM;
    xh = full(double(hungarian_assignment_binary(Av, Hv)));
    costH = sum(sum(cv .* xh));

    agreeCost = abs(costM - costH) < TOL;
    agreeAssg = isequal(xv, xh);
    fprintf(['  visual instance M = %2d, N = %2d :  same cost %s,  ' ...
             'same assignment %s\n'], Mv, Nv, ...
            bool2str(agreeCost), bool2str(agreeAssg));

    % ---- the figure: the two matrices side by side, same colour scale
    f = figure('Position', [80+40*v 80 900 420]);

    subplot(1,2,1);
    plot_assignment(xv, 'MILP solution');
    subplot(1,2,2);
    plot_assignment(xh, 'Hungarian assignment');

    % Uncomment to export this figure as a vector PDF
    % exportgraphics(f, sprintf('Figures/toy_assignment_%s.pdf', visNames{v}), ...
    %                'ContentType', 'vector');

    if ~isempty(figdir)
        pdfName = sprintf('toy_assignment_%s.pdf', visNames{v});
        try
            exportgraphics(f, fullfile(figdir, pdfName), 'ContentType', 'vector');
            fprintf('      figure written to %s\n', fullfile(pwd, figdir, pdfName));
        catch ME
            warning('Could not save the figure: %s', ME.message);
        end
    end

    results.visual(v).M         = Mv;
    results.visual(v).N         = Nv;
    results.visual(v).x_milp    = xv;
    results.visual(v).x_hung    = xh;
    results.visual(v).cost_milp = costM;
    results.visual(v).cost_hung = costH;
    results.visual(v).sameCost  = agreeCost;
    results.visual(v).sameAssg  = agreeAssg;
end

%% ------------------------------------------------------------------
%  PART 2: Monte Carlo batch over increasing sizes
%  ------------------------------------------------------------------

fprintf('\n============ MONTE CARLO EQUIVALENCE CHECK ============\n\n');

sizes  = [6 4; 8 5; 10 7; 12 8; 15 10];   % [M N] pairs to test
TRIALS = 300;                             % attempts per size
p_acc  = 0.6;                             % acceptance probability

nSizes    = size(sizes,1);
fracCost  = zeros(nSizes,1);   % % of instances with identical cost
fracAssig = zeros(nSizes,1);   % % of instances with identical assignment
nTested   = zeros(nSizes,1);   % feasible instances actually compared
maxGap    = zeros(nSizes,1);   % worst absolute cost difference observed

fprintf('  %-12s %8s %14s %18s %14s\n', ...
        'size M x N', 'tested', 'same cost [%]', 'same assignment [%]', 'max |dcost|');
fprintf('  ---------------------------------------------------------------------------\n');

for s = 1:nSizes
    Ms = sizes(s,1);
    Ns = sizes(s,2);

    sameC = 0; sameA = 0; tested = 0; worst = 0;

    for t = 1:TRIALS
        % Simulate charging port parameters
        D = rand(Ms,Ns)*10;
        W = rand(Ms,1)*15;
        C = Cmin + (Cmax-Cmin)*rand(Ms,1);
        P = draw_port_power(Ms);

        A = double(rand(Ms,Ns) < p_acc);

        % Discard instances in which some vehicle cannot be served at all
        [~, num_matches] = hopcroft_karp(A);
        if num_matches < Ns
            continue;
        end

        % Objective function term "c"
        c = build_objective_matrix(D, W, C, P, Qdemand, alpha);

        % ---- MILP
        cvx_begin quiet
            variable x(Ms,Ns) binary
            minimize( sum(sum( x .* c )) )
            subject to
                sum(x,1)' == 1;
                sum(x,2) <= 1;
                x <= A;
        cvx_end
        x  = full(round(x));
        c1 = cvx_optval;

        % ---- Hungarian (console output suppressed with evalc)
        H = c;
        H(~A) = BIGM;
        evalc('assignments = hungarian_assignment_binary(A, H);');
        assignments = full(double(assignments));
        c2 = sum(sum(c .* assignments));

        tested = tested + 1;
        sameC  = sameC + (abs(c1 - c2) < TOL);
        sameA  = sameA + isequal(x, assignments);
        worst  = max(worst, abs(c1 - c2));
    end

    nTested(s)   = tested;
    fracCost(s)  = 100 * sameC / max(tested,1);
    fracAssig(s) = 100 * sameA / max(tested,1);
    maxGap(s)    = worst;

    fprintf('  %-12s %8d %14.2f %18.2f %14.2e\n', ...
            sprintf('%d x %d', Ms, Ns), tested, fracCost(s), fracAssig(s), maxGap(s));
end

fprintf('  ---------------------------------------------------------------------------\n');

%% ------------------------------------------------------------------
%  READY-TO-PASTE LaTeX
%  ------------------------------------------------------------------

fprintf('\n\n=========== COPY THE BLOCKS BELOW INTO THE PAPER ===========\n\n');

% ---- Table of the toy instance (tab:toy)
fprintf('%s\n', '% ---------- Table: toy instance ----------');
fprintf('%s\n', '\begin{table}[t]');
fprintf('%s\n', '\centering');
fprintf('%s\n', sprintf(['\\caption{Toy instance ($M=%d$, $N=%d$). The MILP (CVX/Mosek) and the ' ...
                         'Hungarian algorithm return the same optimal cost and, since the cost ' ...
                         'coefficients are continuous, the same assignment. Produced by ' ...
                         '\\texttt{toy\\_equivalence.m}.}'], M, N));
fprintf('%s\n', '\label{tab:toy}');
fprintf('%s\n', '\begin{tabular}{lcc}');
fprintf('%s\n', '\toprule');
fprintf('%s\n', ' & \textbf{MILP} & \textbf{Hungarian} \\');
fprintf('%s\n', '\midrule');
fprintf('%s\n', sprintf('Optimal cost      & %.4f & %.4f \\\\', cost_milp, cost_hung));
fprintf('%s\n', sprintf('Solve time [s]    & %.4f & %.4f \\\\', time_milp, time_hung));
fprintf('%s\n', sprintf('Same cost?        & \\multicolumn{2}{c}{%s ($|\\Delta| = %.1e$)} \\\\', ...
                        bool2str(sameCost), abs(cost_milp - cost_hung)));
fprintf('%s\n', sprintf('Same assignment?  & \\multicolumn{2}{c}{%s} \\\\', bool2str(sameAssign)));
fprintf('%s\n', '\bottomrule');
fprintf('%s\n', '\end{tabular}');
fprintf('%s\n', '\end{table}');

% ---- Table of the Monte Carlo batch
fprintf('\n%s\n', '% ---------- Table: Monte Carlo batch ----------');
fprintf('%s\n', '\begin{table}[t]');
fprintf('%s\n', '\centering');
fprintf('%s\n', ['\caption{Agreement between the MILP and the Hungarian algorithm over ' ...
                 'random feasible instances of increasing size. Produced by ' ...
                 '\texttt{toy\_equivalence.m}.}']);
fprintf('%s\n', '\label{tab:toy_batch}');
fprintf('%s\n', '\begin{tabular}{lccc}');
fprintf('%s\n', '\toprule');
fprintf('%s\n', '\textbf{Size $M\times N$} & \textbf{Instances} & \textbf{Same cost [\%]} & \textbf{Same assignment [\%]} \\');
fprintf('%s\n', '\midrule');
for s = 1:nSizes
    fprintf('%s\n', sprintf('$%d \\times %d$ & %d & %.1f & %.1f \\\\', ...
            sizes(s,1), sizes(s,2), nTested(s), fracCost(s), fracAssig(s)));
end
fprintf('%s\n', '\bottomrule');
fprintf('%s\n', '\end{tabular}');
fprintf('%s\n', '\end{table}');
fprintf('\n============================================================\n\n');

%% ------------------------------------------------------------------
%  Collect the results
%  ------------------------------------------------------------------

results.toy.A          = A;
results.toy.c          = c;
results.toy.x_milp     = x;
results.toy.x_hung     = assignments;
results.toy.cost_milp  = cost_milp;
results.toy.cost_hung  = cost_hung;
results.toy.time_milp  = time_milp;
results.toy.time_hung  = time_hung;
results.toy.sameCost   = sameCost;
results.toy.sameAssign = sameAssign;

results.sizes     = sizes;
results.nTested   = nTested;
results.fracCost  = fracCost;
results.fracAssig = fracAssig;
results.maxGap    = maxGap;

end

% ---------------------------------------------------------------------
function plot_assignment(x, ttl)
% One panel of the visual validation: the assignment matrix as an image,
% ports on the rows and vehicles on the columns. A filled cell means that
% the pair was selected. The grid is drawn so that individual cells remain
% distinguishable, which is the whole point of showing small instances.

    [M, N] = size(x);
    imagesc(x);
    colormap(gca, [1.00 0.99 0.88; 0.91 0.29 0.21]);   % empty, assigned
    clim([0 1]);

    % A SQUARE BOX, not square cells. "axis image" would keep each cell
    % square and let the panel inherit the aspect ratio of the matrix, which
    % makes the rectangular instance (M > N) come out tall and narrow next to
    % the square one. Since the figure is read as a pattern of filled cells
    % rather than measured, a square box is the better choice: the two
    % sub-figures of the paper then have the same shape and can be stacked.
    % The price is that cells are stretched horizontally when M > N.
    axis square;
    xlim([0.5, N+0.5]);
    ylim([0.5, M+0.5]);
    set(gca, 'XTick', 0.5:1:N+0.5, 'YTick', 0.5:1:M+0.5, ...
             'XTickLabel', [], 'YTickLabel', [], ...
             'GridColor', [0.75 0.75 0.75], 'GridAlpha', 0.6, ...
             'Layer', 'top', 'TickLength', [0 0]);
    grid on;
    xlabel('EV $j$',            'Interpreter', 'latex');
    ylabel('Charging port $i$', 'Interpreter', 'latex');
    title(ttl, 'FontWeight', 'bold');
    set(gca, 'FontSize', 11);
end

% ---------------------------------------------------------------------
function s = bool2str(b)
% Small helper: print a logical as "yes"/"no" instead of 1/0.
    if b
        s = 'yes';
    else
        s = 'no';
    end
end

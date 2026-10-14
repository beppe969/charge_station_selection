function results = structural_analysis(figdir)
% STRUCTURAL_ANALYSIS  Effect of the infrastructure size and of the demand.
%
% Code edited by Luca Ambrosino.
% Optimal Charging Station Selection for EVs.
%
% This function reproduces Section 3.3 of the paper. Starting from the
% baseline configuration M = 500 charging ports and N = 250 EVs, it varies
% one dimension at a time and produces two figures:
%
%   FIGURE 1 - Monotonicity (two panels), optimal objective per EV.
%       Left  : fix the demand N and add charging ports M. Adding ports can
%               only enlarge the feasible set, so the objective decreases.
%       Right : fix M and increase the demand N. The cheap and nearby ports
%               get taken first, so each additional vehicle is served at a
%               progressively worse cost.
%       Both panels report the objective PER EV, so the two curves are
%       directly comparable and cross at the baseline (M = 500, N = 250).
%
%   FIGURE 2 - Objective decomposition.
%       As ports are added, which of the four terms of the objective
%       actually improves? Since distance, waiting time, price and power
%       have incommensurable units, each is reported as a relative
%       improvement with respect to the smallest infrastructure size, and
%       signed so that POSITIVE always means BETTER (lower distance,
%       waiting and price; higher delivered power).
%       This costs nothing extra: the numbers come from the very same MILP
%       solutions computed for the left panel of Figure 1.
%
% NESTED SAMPLING. For each repetition a SINGLE large scenario is drawn,
% and the smaller instances are obtained by keeping only the first M rows
% (or the first N columns) of it. The sub-instances are therefore exactly
% the large one restricted to fewer ports (or fewer vehicles), so the
% monotonicity holds instance by instance and not merely on average.
%
% INPUT
%   figdir  (optional) folder where the figures are saved as PDF.
%
% OUTPUT
%   results struct with the raw numbers, to redraw the plots without
%           re-running the simulations.
%
% REQUIRES: CVX with Mosek (MILP), and hopcroft_karp.m

if nargin < 1
    % Default: save every figure. Pass '' to run without saving.
    figdir = 'Figures';
end

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

cvx_solver mosek % Set mosek solver which is faster than default
cvx_solver_settings('MSK_IPAR_OPTIMIZER', 'MSK_OPTIMIZER_FREE_SIMPLEX')

rng(1); % set seed

%% Simulation parameters

% Baseline configuration, used everywhere else in the paper
M_base = 500;   % Number of charging ports
N_base = 250;   % Number of electric vehicles

% Sweeps. Evenly spaced, so that the shape of the curve is not an artefact
% of where the points happen to fall. Both lists contain the baseline, so
% the two panels can be checked against each other at (M_base, N_base).
M_list = [250 300:100:1000];                   % ports  (N fixed at N_base)
N_list = 50:50:500;                            % EVs    (M fixed at M_base)
% N stops at M_base: at N = M every port must be used and the instance is
% the tightest the model admits. It is still feasible at p_acc = 0.6, but
% any infeasible draw is discarded like the others.

p_acc = 0.6;   % acceptance probability of the acceptable matrix A
REPS  = 30;    % scenarios averaged per size

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

%% ------------------------------------------------------------------
%  PART 1: adding charging ports (N fixed, M grows)
%  ------------------------------------------------------------------

fprintf('--- Part 1: adding charging ports (N = %d) ---\n', N_base);

N    = N_base;
Mmax = max(M_list);

accObj = zeros(numel(M_list), 1);   % accumulated objective
cntM   = zeros(numel(M_list), 1);   % number of feasible instances

% Accumulators for the PHYSICAL quantities, per EV, used in the console table
accD = zeros(numel(M_list), 1);     % average travel distance   [km]
accW = zeros(numel(M_list), 1);     % average waiting time      [min]
accC = zeros(numel(M_list), 1);     % average charging price    [$/kWh]
accP = zeros(numel(M_list), 1);     % average charging time     [min]

% Accumulators for the THREE TERMS OF THE OBJECTIVE, per EV, in dollars.
% Since the revision they share a unit, so they can be decomposed and
% compared directly instead of only through relative improvements.
accTrav  = zeros(numel(M_list), 1);   % alpha_1 * D          [$]
accSpend = zeros(numel(M_list), 1);   % alpha_2 * C*Q        [$]
accOut   = zeros(numel(M_list), 1);   % alpha_3 * (W + Q/P)  [$]

for t = 1:REPS

    % Draw ONE large scenario; the smaller ones are slices of it
    D_all = rand(Mmax,N)*10;                   % Distance port i to vehicle j [km]
    W_all = rand(Mmax,1)*15;                   % Waiting time at each port [min]
    C_all = Cmin + (Cmax-Cmin)*rand(Mmax,1);   % Charging cost at each port [$/kWh]
    P_all = draw_port_power(Mmax);    % Charging power of each port [kW]
    A_all = double(rand(Mmax,N) < p_acc);      % acceptable matrix A
    Q     = Qmin + (Qmax-Qmin)*rand(1,N);      % Energy demand of each EV [kWh]

    for im = 1:numel(M_list)
        M = M_list(im);

        % Sub-instance: the first M ports of the large scenario
        D = D_all(1:M,:);  W = W_all(1:M);
        C = C_all(1:M);    P = P_all(1:M);
        A = A_all(1:M,:);

        % Discard instances in which some vehicle cannot be served at all
        [~, num_matches] = hopcroft_karp(A);
        if num_matches < N
            continue;
        end

        % Objective function term "c"
        [c, tm] = build_objective_matrix(D, W, C, P, Q, alpha);

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

        % CVX returns the variable x as a SPARSE matrix: make it dense
        % before using it, otherwise the sparsity propagates into every
        % quantity computed from it (and sparse values break fprintf).
        x = full(round(x));

        accObj(im) = accObj(im) + cvx_optval;   % cvx_optval is a plain double
        cntM(im)   = cntM(im) + 1;

        % Decomposition: average of each term over the served vehicles.
        % sum(x,2) is the indicator of the ports that were selected, so
        % C'*sum(x,2) is the total price paid, and so on.
        accD(im) = accD(im) + sum(sum(D .* x)) / N;
        accW(im) = accW(im) + sum(W .* sum(x,2)) / N;
        accC(im) = accC(im) + sum(C .* sum(x,2)) / N;
        % The objective now contains the time spent plugged in, not the
        % nominal power, so that is what the decomposition must track.
        accP(im) = accP(im) + sum(sum((60*(1./P)*Q) .* x)) / N;

        % The three terms of the objective, in dollars per vehicle
        accTrav(im)  = accTrav(im)  + sum(sum(tm.travel  .* x)) / N;
        accSpend(im) = accSpend(im) + sum(sum(tm.spend   .* x)) / N;
        accOut(im)   = accOut(im)   + sum(sum(tm.outserv .* x)) / N;
    end
    fprintf('  repetition %d/%d done\n', t, REPS);
end

perEV_M = (accObj ./ max(cntM,1)) / N_base;
perEV_M(cntM == 0) = NaN;

Dbar = accD ./ max(cntM,1);   Dbar(cntM == 0) = NaN;
Wbar = accW ./ max(cntM,1);   Wbar(cntM == 0) = NaN;
Cbar = accC ./ max(cntM,1);   Cbar(cntM == 0) = NaN;
Pbar = accP ./ max(cntM,1);   Pbar(cntM == 0) = NaN;

% Relative improvement with respect to the smallest infrastructure size,
% signed so that POSITIVE always means BETTER. All four terms are now costs
% to be minimised, charging time included: the sign flip that the old power
% term needed has therefore disappeared.
impD = 100 * (Dbar(1) - Dbar) / Dbar(1);
impW = 100 * (Wbar(1) - Wbar) / Wbar(1);
impC = 100 * (Cbar(1) - Cbar) / Cbar(1);
impP = 100 * (Pbar(1) - Pbar) / Pbar(1);

% The same, for the three monetised terms of the objective. These are the
% curves actually plotted: they decompose the criterion itself, and because
% they share a unit their levels are comparable as well as their trends.
Tbar_ = accTrav  ./ max(cntM,1);   Tbar_(cntM == 0) = NaN;
Sbar_ = accSpend ./ max(cntM,1);   Sbar_(cntM == 0) = NaN;
Obar_ = accOut   ./ max(cntM,1);   Obar_(cntM == 0) = NaN;
impTrav  = 100 * (Tbar_(1) - Tbar_) / Tbar_(1);
impSpend = 100 * (Sbar_(1) - Sbar_) / Sbar_(1);
impOut   = 100 * (Obar_(1) - Obar_) / Obar_(1);

%% ------------------------------------------------------------------
%  PART 2: increasing the demand (M fixed, N grows)
%  ------------------------------------------------------------------

fprintf('--- Part 2: increasing the demand (M = %d) ---\n', M_base);

M    = M_base;
Nmax = max(N_list);
accN = zeros(numel(N_list), 1);
cntN = zeros(numel(N_list), 1);

for t = 1:REPS

    % Draw ONE large scenario; the smaller ones keep the first N vehicles.
    % W, C and P describe the ports, so they do not depend on N.
    D_all = rand(M,Nmax)*10;
    W     = rand(M,1)*15;
    C     = Cmin + (Cmax-Cmin)*rand(M,1);
    P     = draw_port_power(M);
    A_all = double(rand(M,Nmax) < p_acc);
    Q_all = Qmin + (Qmax-Qmin)*rand(1,Nmax);   % energy demand, one per EV

    for in = 1:numel(N_list)
        N = N_list(in);

        % Sub-instance: the first N vehicles of the large scenario
        D = D_all(:,1:N);
        A = A_all(:,1:N);
        Q = Q_all(1:N);

        [~, num_matches] = hopcroft_karp(A);
        if num_matches < N
            continue;
        end

        % Objective function term "c"
        c = build_objective_matrix(D, W, C, P, Q, alpha);

        cvx_begin quiet
            variable x(M,N) binary
            minimize( sum(sum( x .* c )) )
            subject to
                sum(x,1)' == 1;
                sum(x,2) <= 1;
                x <= A;
        cvx_end

        % Normalise by N: the TOTAL objective grows with N simply because
        % more vehicles are summed, which would hide the congestion effect.
        accN(in) = accN(in) + cvx_optval / N;
        cntN(in) = cntN(in) + 1;
    end
    fprintf('  repetition %d/%d done\n', t, REPS);
end

perEV_N = accN ./ max(cntN,1);
perEV_N(cntN == 0) = NaN;

%% ------------------------------------------------------------------
%  PART 3: adding CAPACITY instead of variety (port replication)
%  ------------------------------------------------------------------
%
% Part 1 adds brand-new ports at new locations and new prices, so it adds
% VARIETY: each vehicle gets access to closer ports and to cheaper ones.
% Here we ask a different question. What if the operator simply builds more
% sockets at the sites that already exist, so that the network gains
% CAPACITY but no new location and no new tariff?
%
% We therefore take the base network of M0 ports and replicate it r times,
% r = 1,...,5. A replicated port is an exact copy of the original: same
% distance to every vehicle, same price, same waiting time, same
% acceptability. Nothing new becomes reachable and nothing new becomes
% affordable; the only thing that changes is that several vehicles can now
% be served at the same site.
%
% The contrast between this experiment and Part 1 separates the two
% mechanisms through which infrastructure helps: enlarging the choice set,
% and relieving contention for it.

fprintf('--- Part 3: extra ports at existing sites (N = %d) ---\n', N_base);

% This sweep has its own sizes, and deliberately starts at the paper's
% baseline M_base = 500 rather than at M = 250. The reason is structural:
% at M = N every port must be occupied, so the waiting and price terms are
% constant and the optimum there is the pure minimum-distance matching. A
% relative improvement measured from that point would be distorted, because
% the reference is optimal for one criterion and arbitrary for the others.
% Starting at M = 500, that is at a load ratio of 0.5, avoids the artefact.
M_rep = [500 600 700 800 900 1000];
M0    = M_rep(1);             % size of the base network
nExtraMax = max(M_rep) - M0;  % how many extra ports at the largest size

accObjR = zeros(numel(M_rep), 1);
cntR    = zeros(numel(M_rep), 1);
accDR   = zeros(numel(M_rep), 1);
accWR   = zeros(numel(M_rep), 1);
accCR   = zeros(numel(M_rep), 1);

% The three terms of the objective, in dollars per vehicle, as in Part 1
accTravR  = zeros(numel(M_rep), 1);
accSpendR = zeros(numel(M_rep), 1);
accOutR   = zeros(numel(M_rep), 1);

for t = 1:REPS

    % Draw ONE base network of M0 ports.
    N  = N_base;
    D0 = rand(M0,N)*10;
    W0 = rand(M0,1)*15;
    C0 = Cmin + (Cmax-Cmin)*rand(M0,1);
    P0 = draw_port_power(M0);
    A0 = double(rand(M0,N) < p_acc);
    Q  = Qmin + (Qmax-Qmin)*rand(1,N);         % energy demand, one per EV

    % Each extra port is a copy of an EXISTING one, drawn at random: same
    % location, hence the same distance to every vehicle, same tariff, same
    % waiting time and the same acceptability. The sequence of sources is
    % drawn once and then truncated, so that the network at a given size
    % contains the network at every smaller size: the instances stay nested,
    % exactly as in Parts 1 and 2.
    src = randi(M0, nExtraMax, 1);

    for im = 1:numel(M_rep)
        M      = M_rep(im);
        nExtra = M - M0;

        % Index of the base port each of the M ports is a copy of
        idx = [ (1:M0)' ; src(1:nExtra) ];

        D = D0(idx,:);   W = W0(idx);   C = C0(idx);
        P = P0(idx);     A = A0(idx,:);

        [~, num_matches] = hopcroft_karp(A);
        if num_matches < N
            continue;
        end

        % Objective function term "c"
        [c, tm] = build_objective_matrix(D, W, C, P, Q, alpha);

        cvx_begin quiet
            variable x(M,N) binary
            minimize( sum(sum( x .* c )) )
            subject to
                sum(x,1)' == 1;
                sum(x,2) <= 1;
                x <= A;
        cvx_end

        x = full(round(x));

        accObjR(im) = accObjR(im) + cvx_optval;
        cntR(im)    = cntR(im) + 1;
        accDR(im)   = accDR(im) + sum(sum(D .* x)) / N;
        accTravR(im)  = accTravR(im)  + sum(sum(tm.travel  .* x)) / N;
        accSpendR(im) = accSpendR(im) + sum(sum(tm.spend   .* x)) / N;
        accOutR(im)   = accOutR(im)   + sum(sum(tm.outserv .* x)) / N;
        accWR(im)   = accWR(im) + sum(W .* sum(x,2)) / N;
        accCR(im)   = accCR(im) + sum(C .* sum(x,2)) / N;
    end
    fprintf('  repetition %d/%d done\n', t, REPS);
end

perEV_R = (accObjR ./ max(cntR,1)) / N_base;
perEV_R(cntR == 0) = NaN;

DbarR = accDR ./ max(cntR,1);   DbarR(cntR == 0) = NaN;
WbarR = accWR ./ max(cntR,1);   WbarR(cntR == 0) = NaN;
CbarR = accCR ./ max(cntR,1);   CbarR(cntR == 0) = NaN;

% Relative improvement with respect to the un-replicated network (r = 1),
% signed so that a positive value always means better.
impDR = 100 * (DbarR(1) - DbarR) / DbarR(1);
impWR = 100 * (WbarR(1) - WbarR) / WbarR(1);
impCR = 100 * (CbarR(1) - CbarR) / CbarR(1);

TbarR_ = accTravR  ./ max(cntR,1);   TbarR_(cntR == 0) = NaN;
SbarR_ = accSpendR ./ max(cntR,1);   SbarR_(cntR == 0) = NaN;
ObarR_ = accOutR   ./ max(cntR,1);   ObarR_(cntR == 0) = NaN;
impTravR  = 100 * (TbarR_(1) - TbarR_) / TbarR_(1);
impSpendR = 100 * (SbarR_(1) - SbarR_) / SbarR_(1);
impOutR   = 100 * (ObarR_(1) - ObarR_) / ObarR_(1);

% ABSOLUTE values as well. The relative axis of the figure is convenient but
% deceptive here: the distance baseline is a fraction of a kilometre, so a
% few percent on it is a handful of metres, whereas a few percent on the
% waiting time is a matter of minutes. Always read the two together.
fprintf('\nPart 3, absolute per-EV values (extra ports at existing sites):\n');
fprintf('     M     Dbar[km]   Wbar[min]  Cbar[$/kWh]\n');
for im = 1:numel(M_rep)
    fprintf('  %5d   %8.4f   %8.2f   %10.4f\n', ...
            M_rep(im), DbarR(im), WbarR(im), CbarR(im));
end
fprintf('At M = %d: distance %+.1f%%, waiting %+.1f%%, price %+.1f%%\n', ...
        M_rep(end), impDR(end), impWR(end), impCR(end));

%% ------------------------------------------------------------------
%  FIGURE 1: monotonicity, two panels sharing the same height
%  ------------------------------------------------------------------

f1 = figure('Position', [100 100 1000 400]);

% ---- left panel: adding ports
subplot(1,2,1); hold on; grid on; box on;
plot(M_list, perEV_M, '-o', 'LineWidth', 1.8, 'MarkerSize', 7, ...
     'MarkerFaceColor', 'w', 'DisplayName', 'MILP optimum');
ib = find(M_list == M_base, 1);
plot(M_base, perEV_M(ib), 'p', 'MarkerSize', 16, ...
     'MarkerFaceColor', [0.85 0.33 0.10], 'MarkerEdgeColor', 'k', ...
     'DisplayName', sprintf('baseline M = %d', M_base));
xlabel('Number of charging ports M');
ylabel('Mean optimal objective per EV');
title(sprintf('Adding ports (N = %d)', N_base));
legend('Location', 'northeast');
set(gca, 'FontSize', 11);

% ---- right panel: increasing demand
subplot(1,2,2); hold on; grid on; box on;
plot(N_list, perEV_N, '-o', 'LineWidth', 1.8, 'MarkerSize', 7, ...
     'MarkerFaceColor', 'w', 'DisplayName', 'MILP optimum');
ib = find(N_list == N_base, 1);
plot(N_base, perEV_N(ib), 'p', 'MarkerSize', 16, ...
     'MarkerFaceColor', [0.85 0.33 0.10], 'MarkerEdgeColor', 'k', ...
     'DisplayName', sprintf('baseline N = %d', N_base));
xlabel('Number of EVs N');
ylabel('Mean optimal objective per EV');
title(sprintf('Increasing demand (M = %d)', M_base));
legend('Location', 'northwest');
set(gca, 'FontSize', 11);

% Uncomment the next line to export this figure as a vector PDF
% exportgraphics(f1, 'Figures/monotonicity.pdf', 'ContentType', 'vector');

%% ------------------------------------------------------------------
%  FIGURE 2: objective decomposition
%  ------------------------------------------------------------------

f2 = figure('Position', [100 100 640 460]);
hold on; grid on; box on;

% The three curves are the three TERMS OF THE OBJECTIVE, each in dollars per
% vehicle. Before the revision the terms had incommensurable units and only
% the physical quantities could be compared; now the decomposition is of the
% criterion itself, which is what makes the reading unambiguous.
plot(M_list, impTrav, '-o', 'LineWidth', 1.8, 'MarkerSize', 7, ...
     'MarkerFaceColor', 'w', 'DisplayName', 'Travel $\alpha_1 D$');
plot(M_list, impOut, '-s', 'LineWidth', 1.8, 'MarkerSize', 7, ...
     'MarkerFaceColor', 'w', 'DisplayName', 'Time out of service $\alpha_3 T$');
plot(M_list, impSpend, '-^', 'LineWidth', 1.8, 'MarkerSize', 7, ...
     'MarkerFaceColor', 'w', 'DisplayName', 'Energy bill $\alpha_2 CQ$');

yline(0, 'k:', 'HandleVisibility', 'off');
xlabel('Number of charging ports M');
ylabel(sprintf('Relative improvement w.r.t. M = %d [%%]', M_list(1)));
title(sprintf('New ports at new sites (N = %d)', N_base));
legend('Location', 'northwest', 'Interpreter', 'latex');
set(gca, 'FontSize', 11);

% Uncomment the next line to export this figure as a vector PDF
% exportgraphics(f2, 'Figures/objective_decomposition.pdf', 'ContentType', 'vector');

%% ------------------------------------------------------------------
%  FIGURE 3: same decomposition, but adding CAPACITY instead of variety
%  ------------------------------------------------------------------
% Deliberately identical in style, axes and quantities to Figure 2, so that
% the two can be placed side by side and read as a single comparison. Both
% start from the same base network of 250 ports; they differ only in how
% the additional ports are built.

f3 = figure('Position', [140 100 640 460]);
hold on; grid on; box on;

plot(M_rep, impTravR, '-o', 'LineWidth', 1.8, 'MarkerSize', 7, ...
     'MarkerFaceColor', 'w', 'DisplayName', 'Travel $\alpha_1 D$');
plot(M_rep, impOutR, '-s', 'LineWidth', 1.8, 'MarkerSize', 7, ...
     'MarkerFaceColor', 'w', 'DisplayName', 'Time out of service $\alpha_3 T$');
plot(M_rep, impSpendR, '-^', 'LineWidth', 1.8, 'MarkerSize', 7, ...
     'MarkerFaceColor', 'w', 'DisplayName', 'Energy bill $\alpha_2 CQ$');

yline(0, 'k:', 'HandleVisibility', 'off');
xlabel('Number of charging ports M');
ylabel(sprintf('Relative improvement w.r.t. M = %d [%%]', M_rep(1)));
title(sprintf('Extra ports at existing sites (N = %d)', N_base));
legend('Location', 'northwest', 'Interpreter', 'latex');
set(gca, 'FontSize', 11);
xlim([min(M_rep)-30, max(M_rep)+30]);

% Uncomment the next line to export this figure as a vector PDF
% exportgraphics(f3, 'Figures/decomposition_replication.pdf', 'ContentType', 'vector');

%% ------------------------------------------------------------------
%  Save and collect the results
%  ------------------------------------------------------------------

% Consistency check: both panels of Figure 1 describe the same baseline
% instance, so the two highlighted points should agree up to MC noise.
fprintf('\nBaseline (M = %d, N = %d): %.4f (ports sweep) vs %.4f (EV sweep)\n', ...
    M_base, N_base, perEV_M(M_list == M_base), perEV_N(N_list == N_base));

fprintf('\nDecomposition at M = %d (improvement w.r.t. M = %d):\n', ...
    max(M_list), M_list(1));
fprintf('  distance %+6.1f%%   waiting %+6.1f%%   price %+6.1f%%   charge time %+6.1f%%\n', ...
    impD(end), impW(end), impC(end), impP(end));
fprintf('Objective terms at M = %d, in $ per EV (improvement w.r.t. M = %d):\n', ...
    max(M_list), M_list(1));
fprintf('  travel %5.2f (%+.1f%%)   energy %5.2f (%+.1f%%)   out of service %6.2f (%+.1f%%)\n', ...
    Tbar_(end), impTrav(end), Sbar_(end), impSpend(end), Obar_(end), impOut(end));

if ~isempty(figdir)
    if ~exist(figdir, 'dir')
        mkdir(figdir);
    end
    try
        exportgraphics(f1, fullfile(figdir, 'monotonicity.pdf'), 'ContentType', 'vector');
        exportgraphics(f2, fullfile(figdir, 'objective_decomposition.pdf'), 'ContentType', 'vector');
        exportgraphics(f3, fullfile(figdir, 'decomposition_replication.pdf'), 'ContentType', 'vector');
        fprintf('Figures saved in %s\n', figdir);
    catch ME
        warning('Could not save the figures: %s', ME.message);
    end
end

results.M_base  = M_base;
results.N_base  = N_base;
results.M_list  = M_list;
results.perEV_M = perEV_M;
results.N_list  = N_list;
results.perEV_N = perEV_N;
results.Dbar    = Dbar;
results.Wbar    = Wbar;
results.Cbar    = Cbar;
results.Pbar    = Pbar;
results.impD    = impD;
results.impW    = impW;
results.impC    = impC;
results.impP     = impP;       % charging-time improvement [%]
results.impTrav  = impTrav;    % travel term,       improvement [%]
results.impSpend = impSpend;   % energy-bill term,  improvement [%]
results.impOut   = impOut;     % out-of-service term, improvement [%]
results.levels   = [Tbar_, Sbar_, Obar_];   % absolute levels [$ per EV]
results.REPS    = REPS;
results.p_acc   = p_acc;

% Part 3: extra ports at existing sites
results.M_rep   = M_rep;
results.perEV_R = perEV_R;
results.DbarR   = DbarR;
results.WbarR   = WbarR;
results.CbarR   = CbarR;
results.impDR   = impDR;
results.impWR   = impWR;
results.impCR   = impCR;
results.impTravR  = impTravR;
results.impSpendR = impSpendR;
results.impOutR   = impOutR;

%% ------------------------------------------------------------------
%  PART 4: comparison against the decentralised baselines
%  ------------------------------------------------------------------
% The three parts above characterise the OPTIMUM as the network changes.
% What they do not say is how much the optimum is worth compared with the
% myopic rules a driver would actually follow. That comparison is run by a
% separate function, kept apart so that this one stays readable, and it
% produces the summary table (console and LaTeX) together with its own
% boxplots.

if exist('synthetic_comparison', 'file') == 2
    fprintf('\n\n');
    results.comparison = synthetic_comparison(figdir);
else
    warning(['synthetic_comparison.m not found: skipping the comparison ' ...
             'against the greedy baselines and its table.']);
end

end

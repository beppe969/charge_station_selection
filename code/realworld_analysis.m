function results = realworld_analysis(csvFile, figdir, cityTag)
% REALWORLD_ANALYSIS  The case study on one real network (Sec. 4.1).
%
%   results = realworld_analysis(csvFile)
%   results = realworld_analysis(csvFile, figdir, cityTag)
%
% Expands the stations of the file into individual ports, simulates the
% demand on top of the real coordinates, and compares five methods over 100
% random layouts: the proposed model, two myopic rules, a sequential rule
% that uses the full criterion, and a minimum-distance matching. The last
% two fill the off-diagonal cells of the two-by-two design discussed in the
% paper.
%
% Run it once per city:
%   realworld_analysis('../data/CS_location_DC_complete.csv')
%   realworld_analysis('../data/CS_location_Chicago.xlsx')
%   realworld_analysis('../data/CS_location_Manhattan.xlsx')
%
% Note on feasibility. draw_instance may grant a stranded vehicle its
% nearest affordable port. That only gives THAT vehicle an option; several
% repaired vehicles can still depend on the same port, so the instance is
% not feasible by construction and is tested with hopcroft_karp before
% anything is solved. At Rmax = 5 km the repair never fires anyway.
%
% Prints a paste-ready LaTeX table and writes seven figures per city.


if nargin < 1 || isempty(csvFile);  csvFile = 'CS_location_DC_complete.csv';  end
% Default: save every figure. Pass '' to run without saving.
if nargin < 2;                      figdir  = 'Figures';                       end

% Every figure is saved with the name of the city appended, so that running
% the function on the three networks in a row does not overwrite the same
% seven files. Without a tag the third run would silently leave the paper
% illustrated with a single city, which is the sort of mistake one notices
% only at proof stage.
if nargin < 3 || isempty(cityTag)
    [~, base, ~] = fileparts(csvFile);
    cityTag = erase(base, 'CS_location_');
    if contains(cityTag, 'DC', 'IgnoreCase', true)
        cityTag = 'Washington';            % the DC file is named differently
    end
end
cityTag = char(cityTag);

% cityTag is what goes into file names, so it has to stay short and free of
% spaces. For figure titles we want the name as it appears in the paper.
switch lower(cityTag)
    case 'washington', cityName = 'Washington DC';
    case 'manhattan',  cityName = 'Manhattan (New York)';
    otherwise,         cityName = cityTag;
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

rng(1); % set seed

%% ==================================================================
%  SETTINGS
%  ==================================================================

load_ratio = 0.5;    % N / M, the baseline occupancy of Section 3.3
% Driving-range radius, which defines reachability R_ij. The value matters:
% with 3 km a handful of vehicles sampled at the edge of the network find no
% port at all and the instance becomes infeasible; at 5 km every vehicle has
% several dozen acceptable ports and the instances are comfortably feasible.
Rmax_km    = 5.0;    % [km]
% Energy demand of each EV [kWh]. Drawn per vehicle in the interval below:
% a car arrives with a partly depleted battery of about 60 kWh and charges
% to roughly 90%, so the energy requested varies from one driver to the next.
Qrange     = [20 60];  % [kWh]
% Willingness to pay, in $/kWh. We draw THIS rather than the budget in
% dollars, and then set B_j = rho_j * Q_j. The reason is that the budget has
% to scale with the energy requested: a driver who needs 60 kWh expects to
% pay about three times what a driver who needs 20 kWh pays, and drawing a
% flat B in [16,24] $ against a demand in [20,60] kWh would instead leave
% part of the fleet unable to afford ANY port (the ratio B/Q would fall to
% 0.27 $/kWh, below the cheapest tariff of 0.303). The range below is the
% one the fixed-demand version implied, so admissibility is unchanged.
rhoMin     = 0.40;   % [$/kWh]
rhoMax     = 0.60;   % [$/kWh]
kNear      = 3;      % neighbourhood size of the k-nearest policy
REPS       = 100;    % random EV layouts in Stage 2
ORDERS     = 5;      % random processing orders per greedy run
sigmaKm    = 2.5;    % Gaussian spread of the EV density around the centroid

BIGM = 1e6;          % penalty used to forbid the non-acceptable pairs


% Objective coefficients. Every term of the objective is expressed in
% dollars, so these are conversion factors with a physical meaning
% ($/km and $/min): the model has two of them, since the energy term is
% already money and needs no conversion.
% See objective_weights.m for where the values come from, and
% sensitivity_alpha.m for the evidence that the conclusions do not depend
% on them. objective_weights('low') and ('high') change the value of time.
alpha = objective_weights();

% Charging prices are uniform between the min and max US state-level price
Cmin = 0.303;  % [$/kWh]
Cmax = 0.530;  % [$/kWh]

% Nominal power associated with each connector type [kW]
pow_level1 = 7;    % Level 1 (slow); only a couple of them in the DC subset
pow_level2 = 19;   % Level 2 (the bulk of the network)
pow_dcfast = 50;   % DC fast

% The sparse Hungarian is the solver of choice; fall back to the dense one
if exist('hungarian_assignment_sparse', 'file') == 2
    solver_name = 'sparse Hungarian';
    solve_fun   = @hungarian_assignment_sparse;
else
    warning('hungarian_assignment_sparse not found: using the dense version.');
    solver_name = 'dense Hungarian';
    solve_fun   = @hungarian_assignment_binary;
end

%% ==================================================================
%  PART 0: load the network and expand the stations into ports
%  ==================================================================

assert(isfile(csvFile), 'Cannot find %s', csvFile);

% Two input formats are supported, because the cities were geocoded with
% two different services. The original Washington file is a CSV written in
% European convention, with ';' as separator and the comma as decimal mark;
% the city files produced from the national dataset are spreadsheets, where
% numbers are stored as numbers. The Delimiter option is only meaningful
% for text files and errors out on a spreadsheet, hence the branch.
[~, ~, dataExt] = fileparts(csvFile);
if any(strcmpi(dataExt, {'.csv', '.txt', '.dat'}))
    T = readtable(csvFile, 'Delimiter', ';', 'TextType', 'string');
else
    T = readtable(csvFile, 'TextType', 'string');
end

% Works for both: on a decimal-comma string it substitutes the separator,
% on a genuine number the substitution is a no-op.
toNum = @(z) str2double(strrep(string(z), ',', '.'));

statLat  = toNum(T.Latitude);
statLon  = toNum(T.Longitude);
nLevel1  = round(toNum(T.EVLevel1EVSENum));
nLevel2  = round(toNum(T.EVLevel2EVSENum));
nDCfast  = round(toNum(T.EVDCFastCount));

% Drop stations without usable coordinates
ok = isfinite(statLat) & isfinite(statLon);
statLat = statLat(ok);  statLon = statLon(ok);
nLevel1 = nLevel1(ok);  nLevel2 = nLevel2(ok);  nDCfast = nDCfast(ok);
nLevel1(~isfinite(nLevel1)) = 0;
nLevel2(~isfinite(nLevel2)) = 0;
nDCfast(~isfinite(nDCfast)) = 0;

nStations = numel(statLat);

% ---- expand each station into its individual charging ports
portLat = [];  portLon = [];  portPow = [];  portStation = [];

for s = 1:nStations
    % A station with no declared connector still counts as one port, so
    % that no site of the network is silently discarded.
    counts = [nLevel1(s), nLevel2(s), nDCfast(s)];
    powers = [pow_level1, pow_level2, pow_dcfast];
    if sum(counts) == 0
        counts = [0 1 0];
    end
    for k = 1:3
        for r = 1:counts(k)
            portLat(end+1,1)     = statLat(s);   %#ok<AGROW>
            portLon(end+1,1)     = statLon(s);   %#ok<AGROW>
            portPow(end+1,1)     = powers(k);    %#ok<AGROW>
            portStation(end+1,1) = s;            %#ok<AGROW>
        end
    end
end

M = numel(portLat);              % Number of charging ports
N = round(load_ratio * M);       % Number of electric vehicles

nPortsPerStation = accumarray(portStation, 1, [nStations 1]);

fprintf('\n=============== %s NETWORK ===============\n', upper(cityTag));
fprintf('  stations           : %d\n', nStations);
fprintf('  charging ports M   : %d  (L1 %d, L2 %d, DC fast %d)\n', ...
        M, sum(nLevel1), sum(nLevel2), sum(nDCfast));
fprintf('  simulated EVs  N   : %d   (load ratio N/M = %.2f)\n', N, N/M);
fprintf('  solver             : %s\n', solver_name);
fprintf('=====================================================\n\n');

% Centroid and admissible polygon, computed once and reused at every layout
centLat = mean(statLat, 'omitnan');
centLon = mean(statLon, 'omitnan');
try
    K = boundary(statLon, statLat, 0.6);     % concave hull, shrink factor 0.6
catch
    K = convhull(statLon, statLat);          % fallback for old releases
end
hullPoly = simplify(rmholes(polyshape(statLon(K), statLat(K))));

%% ==================================================================
%  PART 1: one single instance, fully visualised
%  ==================================================================

fprintf('--- Stage 1: single instance ---\n');

[evLat, evLon] = sample_ev_positions(hullPoly, centLat, centLon, N, sigmaKm);
D = haversine_matrix(portLat, portLon, evLat, evLon);      % M x N [km]

[W, C, P, ~, A, nRep, Q] = draw_instance(portStation, nStations, portPow, D, ...
                                      Cmin, Cmax, rhoMin, rhoMax, Qrange, Rmax_km, N);

% Objective function term "c"
c = build_objective_matrix(D, W, C, P, Q, alpha);

% Diagnostic only: can every vehicle be served at all?
[~, num_matches] = hopcroft_karp(A);
fprintf('  acceptable pairs   : %.1f%% of M*N\n', 100*nnz(A)/(M*N));
fprintf('  acceptable per EV  : min %d, mean %.0f\n', min(sum(A,1)), mean(sum(A,1)));
fprintf('  repaired EVs       : %d (had no acceptable port)\n', nRep);
fprintf('  max matching       : %d of %d EVs %s\n', num_matches, N, ...
        ternary(num_matches == N, '(feasible)', '(INFEASIBLE)'));

% ---- optimal assignment via the sparse Hungarian algorithm
H = c;
H(~A) = BIGM;   % set a very big number for unadmissible selections
t0 = tic;
x_opt = solve_fun(A, H);
t_opt = toc(t0);
x_opt = full(double(x_opt));    % the solver may return a sparse matrix

% ---- greedy baselines, one random order each
t0 = tic;  x_near = greedy_nearest(D, A);              t_near = toc(t0);
t0 = tic;  x_kche = greedy_cheapest_k(D, C, A, kNear); t_kche = toc(t0);
t0 = tic;  x_gcos = greedy_cost(c, A);                 t_gcos = toc(t0);

% ---- coordinated, but single-objective: minimum total distance.
% This baseline exists to separate two things the other comparisons leave
% entangled, namely the benefit of COORDINATING and the benefit of
% coordinating on the RIGHT criterion. It is solved by the same exact
% matching algorithm on the same acceptable graph, so it is globally
% optimal; only the cost matrix is different, being the distance alone.
% Against the greedy nearest-port rule it isolates coordination on a fixed
% criterion; against the proposed model it isolates the criterion itself.
Hd = D;
Hd(~A) = BIGM;
t0 = tic;  x_mind = full(double(solve_fun(A, Hd)));  t_mind = toc(t0);

m_opt  = assignment_stats(x_opt,  c, D, W, C, N, Q, portStation, nStations);
m_near = assignment_stats(x_near, c, D, W, C, N, Q, portStation, nStations);
m_kche = assignment_stats(x_kche, c, D, W, C, N, Q, portStation, nStations);
m_gcos = assignment_stats(x_gcos, c, D, W, C, N, Q, portStation, nStations);
m_mind = assignment_stats(x_mind, c, D, W, C, N, Q, portStation, nStations);

fprintf('\n  %-26s %12s %12s %12s %12s %12s\n', '', 'optimal', 'nearest', 'cheapest-k', 'greedy-cost', 'min-dist');
fprintf('  %-26s %12.1f %12.1f %12.1f %12.1f %12.1f\n', 'objective',        m_opt.obj,  m_near.obj,  m_kche.obj,  m_gcos.obj,  m_mind.obj);
fprintf('  %-26s %12.3f %12.3f %12.3f %12.3f %12.3f\n', 'avg distance [km]',m_opt.dist, m_near.dist, m_kche.dist, m_gcos.dist, m_mind.dist);
fprintf('  %-26s %12.3f %12.3f %12.3f %12.3f %12.3f\n', 'p90 distance [km]',m_opt.d90,  m_near.d90,  m_kche.d90,  m_gcos.d90,  m_mind.d90);
fprintf('  %-26s %12.3f %12.3f %12.3f %12.3f %12.3f\n', 'max distance [km]',m_opt.dmax, m_near.dmax, m_kche.dmax, m_gcos.dmax, m_mind.dmax);
fprintf('  %-26s %12.4f %12.4f %12.4f %12.4f %12.4f\n', 'avg price [$/kWh]',m_opt.cost, m_near.cost, m_kche.cost, m_gcos.cost, m_mind.cost);
fprintf('  %-26s %12.2f %12.2f %12.2f %12.2f %12.2f\n', 'avg spend [$]',    m_opt.spend,m_near.spend,m_kche.spend,m_gcos.spend,m_mind.spend);
fprintf('  %-26s %12.2f %12.2f %12.2f %12.2f %12.2f\n', 'avg waiting [min]',m_opt.wait, m_near.wait, m_kche.wait, m_gcos.wait, m_mind.wait);
fprintf('  %-26s %12d %12d %12d %12d %12d\n',           'idle stations',    m_opt.unused, m_near.unused, m_kche.unused, m_gcos.unused, m_mind.unused);
fprintf('  %-26s %12.1f %12.1f %12.1f %12.1f %12.1f\n', 'served [%]',   100*m_opt.served, 100*m_near.served, 100*m_kche.served, 100*m_gcos.served, 100*m_mind.served);
fprintf('  %-26s %12.3f %12.3f %12.3f %12.3f %12.3f\n', 'solve time [s]',   t_opt, t_near, t_kche, t_gcos, t_mind);

%% ==================================================================
%  PART 2: many instances, aggregated statistics
%  ==================================================================

fprintf('\n--- Stage 2: %d random layouts ---\n', REPS);

% Five methods, arranged so that the comparison forms a two-by-two design.
% Along one axis, whether the rule coordinates or commits sequentially;
% along the other, whether it looks at the full criterion or at distance
% alone. The proposed model is coordinated and complete, the greedy
% nearest-port rule is neither, and the remaining two fill the off-diagonal
% cells, which is what allows the gap to be decomposed into the value of
% coordinating and the value of choosing the right criterion.
nMethods = 5;
objAcc  = nan(REPS, nMethods);
distAcc = nan(REPS, nMethods);
costAcc = nan(REPS, nMethods);
waitAcc = nan(REPS, nMethods);
servAcc = nan(REPS, nMethods);
timeAcc = nan(REPS, nMethods);
d90Acc  = nan(REPS, nMethods);   % 90th percentile of the per-vehicle distance
dmaxAcc = nan(REPS, nMethods);   % worst trip of the fleet
spendAcc= nan(REPS, nMethods);   % money paid for the energy, C_i * Q_j [$]
unusedAcc = nan(REPS, nMethods); % stations that serve nobody
repAcc  = nan(REPS, 1);          % EVs repaired in each layout

% Per-EV outcomes pooled across layouts, for the trade-off figure
% Per-station utilisation, accumulated across layouts (vehicles served)
utilAcc = zeros(nStations, nMethods);

for rep = 1:REPS

    % ---- a completely new instance: new EV positions, new station prices,
    %      new waiting times, new budgets. Only the physical network is kept.
    [evLat_r, evLon_r] = sample_ev_positions(hullPoly, centLat, centLon, N, sigmaKm);
    D = haversine_matrix(portLat, portLon, evLat_r, evLon_r);

    [W, C, P, ~, A, nRep_r, Q] = draw_instance(portStation, nStations, portPow, D, ...
                                            Cmin, Cmax, rhoMin, rhoMax, Qrange, Rmax_km, N);
    repAcc(rep) = nRep_r;

    % Objective function term "c"
    c = build_objective_matrix(D, W, C, P, Q, alpha);

    % ---- FEASIBILITY OF THE WHOLE FLEET, CHECKED EXPLICITLY.
    %      draw_instance may have granted a stranded vehicle reachability of
    %      its nearest affordable port. That guarantees the vehicle has at
    %      least one option, which is Hall's condition on the singletons; it
    %      does NOT guarantee that all N vehicles can be served at the same
    %      time at distinct ports, since several repaired vehicles may end up
    %      depending on the same one. The instance is therefore not feasible
    %      by construction and has to be tested, which a maximum-cardinality
    %      matching does in O(|E| sqrt(M+N)).
    [~, nMatch] = hopcroft_karp(A);
    if nMatch < N
        fprintf('  layout %3d: no EV-saturating matching exists, skipped\n', rep);
        objAcc(rep,:) = NaN;
        continue;
    end

    % ---- (1) optimal.
    %      A WORD ON THE TEST BELOW. The forbidden pairs are given to the
    %      Hungarian algorithm as a large FINITE penalty, not as infinity,
    %      so on an infeasible instance the algorithm does not fail: it
    %      returns a complete assignment that silently uses forbidden pairs
    %      at cost BIGM. Counting the assigned vehicles therefore detects
    %      nothing, since nnz(xo) equals N either way. What has to be
    %      checked is that no pair outside the acceptable matrix was used.
    H = c;
    H(~A) = BIGM;
    t0 = tic;
    xo = solve_fun(A, H);
    timeAcc(rep,1) = toc(t0);
    xo = full(double(xo));

    if nnz(xo) < N || any(~A(logical(xo)))
        fprintf('  layout %3d: infeasible, skipped\n', rep);
        objAcc(rep,:) = NaN;
        continue;
    end
    s1 = assignment_stats(xo, c, D, W, C, N, Q, portStation, nStations);

    % ---- (2)(3)(4) decentralised rules, averaged over ORDERS orders.
    %      The third of them is the informed one: it commits ports
    %      sequentially like the others, but each vehicle minimises the same
    %      complete cost c that the optimum minimises globally. Comparing it
    %      with (2) isolates the value of the criterion under decentralised
    %      behaviour; comparing it with (1) isolates the value of
    %      coordination at a fixed criterion.
    [s2, t2, last2] = avg_greedy(@() greedy_nearest(D, A),              ORDERS, c, D, W, C, N, Q, portStation, nStations);
    [s3, t3, last3] = avg_greedy(@() greedy_cheapest_k(D, C, A, kNear), ORDERS, c, D, W, C, N, Q, portStation, nStations);
    [s4, t4, last4] = avg_greedy(@() greedy_cost(c, A),                 ORDERS, c, D, W, C, N, Q, portStation, nStations);
    timeAcc(rep,2) = t2;
    timeAcc(rep,3) = t3;
    timeAcc(rep,4) = t4;

    % ---- (5) coordinated, minimum total distance. Same exact solver, same
    %      acceptable graph, different criterion: it is globally optimal for
    %      distance and blind to price and time. Its assignment is evaluated
    %      with the full cost c, which is what makes the comparison fair.
    Hd = D;
    Hd(~A) = BIGM;
    t0 = tic;
    xd = full(double(solve_fun(A, Hd)));
    timeAcc(rep,5) = toc(t0);
    if nnz(xd) < N || any(~A(logical(xd)))
        fprintf('  layout %3d: min-distance matching infeasible, skipped\n', rep);
        objAcc(rep,:) = NaN;
        continue;
    end
    s5 = assignment_stats(xd, c, D, W, C, N, Q, portStation, nStations);

    S = {s1, s2, s3, s4, s5};
    for k = 1:nMethods
        objAcc(rep,k)    = S{k}.obj;
        distAcc(rep,k)   = S{k}.dist;
        costAcc(rep,k)   = S{k}.cost;
        waitAcc(rep,k)   = S{k}.wait;
        servAcc(rep,k)   = S{k}.served;
        d90Acc(rep,k)    = S{k}.d90;
        dmaxAcc(rep,k)   = S{k}.dmax;
        spendAcc(rep,k)  = S{k}.spend;
        unusedAcc(rep,k) = S{k}.unused;
    end

    % ---- per-method post-processing: how many vehicles each STATION serves
    Xall = {xo, last2.x, last3.x, last4.x, xd};
    for k = 1:nMethods
        utilAcc(:,k) = utilAcc(:,k) + ...
                       accumarray(portStation, sum(Xall{k},2), [nStations 1]);
    end

    if mod(rep, 10) == 0
        fprintf('  layout %3d/%d done | optimal %.1f | nearest %.1f | cheapest-k %.1f\n', ...
                rep, REPS, s1.obj, s2.obj, s3.obj);
    end
end

valid  = all(isfinite(objAcc), 2);
nValid = nnz(valid);

%% ------------------------------------------------------------------
%  Aggregated table: mean +/- standard deviation
%  ------------------------------------------------------------------

labels      = {'Optimal (Hungarian)', 'Greedy nearest', ...
               sprintf('Cheapest of %d-nearest', kNear), ...
               'Greedy on full cost', 'Min-distance matching'};
shortLabels = {'Optimal', 'Nearest', sprintf('Cheapest-%d', kNear), ...
               'Greedy-cost', 'Min-dist'};

mu = @(v) mean(v(valid,:), 1);
sd = @(v) std( v(valid,:), 0, 1);

mObj  = mu(objAcc);   sObj  = sd(objAcc);
mDist = mu(distAcc);  sDist = sd(distAcc);
mCost = mu(costAcc);  sCost = sd(costAcc);
mWait = mu(waitAcc);  sWait = sd(waitAcc);
mServ = mu(servAcc);  sServ = sd(servAcc);
mTime = mu(timeAcc);  sTime = sd(timeAcc);
mD90  = mu(d90Acc);   sD90  = sd(d90Acc);
mDmax = mu(dmaxAcc);  sDmax = sd(dmaxAcc);
mSpend= mu(spendAcc); sSpend= sd(spendAcc);
mUnused = mu(unusedAcc);  sUnused = sd(unusedAcc);

gap = 100 * (mObj - mObj(1)) / abs(mObj(1));   % relative loss vs optimum

fprintf('\n========== AGGREGATED OVER %d FEASIBLE LAYOUTS ==========\n\n', nValid);
fprintf('  %-24s %16s %10s %14s %16s %12s %12s\n', ...
        'Method', 'Objective', 'Gap [%]', 'Dist [km]', 'Price [$/kWh]', 'Served [%]', 'Time [s]');
for k = 1:nMethods
    fprintf('  %-24s %8.1f+-%-6.1f %10.2f %7.3f+-%-5.3f %8.4f+-%-6.4f %6.1f+-%-4.1f %6.3f+-%-5.3f\n', ...
        labels{k}, mObj(k), sObj(k), gap(k), mDist(k), sDist(k), ...
        mCost(k), sCost(k), 100*mServ(k), 100*sServ(k), mTime(k), sTime(k));
end

% ---- what the averages hide: the tail of the distance distribution, and
%      the money actually spent on the charge.
fprintf('\n---------- Tail of the trip length, and the energy bill ----------\n');
fprintf('  %-24s %14s %14s %14s %16s %16s\n', ...
        'Method', 'mean [km]', 'p90 [km]', 'max [km]', 'spend [$/EV]', ...
        sprintf('idle of %d stat.', nStations));
for k = 1:nMethods
    fprintf('  %-24s %7.3f+-%-5.3f %7.3f+-%-5.3f %7.3f+-%-5.3f %8.2f+-%-6.2f %9.1f+-%-5.1f\n', ...
        labels{k}, mDist(k), sDist(k), mD90(k), sD90(k), ...
        mDmax(k), sDmax(k), mSpend(k), sSpend(k), mUnused(k), sUnused(k));
end
fprintf(['\n  Read the last column against the first: the min-distance ' ...
         'matching is\n  globally optimal on the trip length and pays for ' ...
         'it on the energy bill.\n']);
fprintf('\n  EVs repaired per layout: mean %.1f of %d (%.2f%%)\n', ...
        mean(repAcc(valid)), N, 100*mean(repAcc(valid))/N);

%% ------------------------------------------------------------------
%  Ready-to-paste LaTeX table
%  ------------------------------------------------------------------

% ---- the last column of the table: stations never used
%
% It counts STATIONS, not ports. Ports would carry no information at all,
% since every method serves the same N vehicles with one socket each and
% therefore leaves exactly M - N ports idle whatever the assignment.
%
% It is also computed ACROSS the layouts, not within one. Two different
% quantities are easily confused here, and they rank the methods in
% opposite ways. Within a single snapshot the optimum leaves MORE stations
% idle than a nearest-port rule, because it consolidates the demand on the
% sites that are worth using, while the greedy scatters vehicles over
% whatever happens to be closest. Across the layouts, where prices and
% queues are redrawn each time, the picture reverses: the optimum reaches
% almost every station sooner or later, whereas the greedy keeps ignoring
% the same peripheral ones, since its choice depends only on a geometry
% that never changes. The second quantity is the one reported, because it
% is a property of the network rather than of one snapshot, and because
% every other column of the table is likewise aggregated over the layouts.
% A station counts as never used when its mean utilisation stays below
% 0.05 vehicles, that is when it is chosen in fewer than one layout in
% twenty.
neverUsed = nan(1, nMethods);
if nValid > 0
    utilMeanTab = utilAcc / nValid;
    for k = 1:nMethods
        neverUsed(k) = nnz(utilMeanTab(:,k) < 0.05);
    end
end

fprintf('\n\n=========== COPY THE BLOCK BELOW INTO THE PAPER ===========\n\n');
fprintf('%s\n', '\begin{table}[t]');
fprintf('%s\n', '\centering');
fprintf('%s\n', sprintf(['\\caption{%s case study: the proposed model against three ' ...
    'decentralized rules and a coordinated matching that minimizes travel distance ' ...
    'alone, over %d random EV layouts on the real network ($M=%d$ ports on %d stations, ' ...
    '$N=%d$ vehicles). Entries are mean $\\pm$ standard deviation across the layouts. ' ...
    'All per-vehicle figures are averaged over the vehicles each method actually ' ...
    'serves, so they must be read together with the last column: a rule that leaves ' ...
    'part of the fleet unserved is averaging over a different population. For the ' ...
    'record, %d of the %d installed stations are never used by the optimum.}'], ...
    cityTag, nValid, M, nStations, N, neverUsed(1), nStations));
fprintf('%s\n', '\label{tab:realworld}');
fprintf('%s\n', '\small');
fprintf('%s\n', '\setlength{\tabcolsep}{5pt}');
% SERVED IS REPORTED NEXT TO THE AVERAGES, AND NOT INSTEAD OF THEM.
% Every per-vehicle figure in this table is an average over the vehicles a
% method actually serves, so two methods with different service rates are
% averaging over different populations. Printing the objective without the
% share served would therefore flatter whichever rule abandons part of the
% fleet, and the two columns have to be read together.
fprintf('%s\n', '\begin{tabular}{lccccccc}');
fprintf('%s\n', '\toprule');
fprintf('%s\n', ['\textbf{Method} & \textbf{Objective} & \textbf{Gap [\%]} & ' ...
                 '\textbf{Dist [km]} & \textbf{Price [\$/kWh]} & ' ...
                 '\textbf{Wait [min]} & \textbf{Served [\%]} \\']);
fprintf('%s\n', '\midrule');
for k = 1:nMethods
    fprintf('%s\n', sprintf(['%s & $%.1f \\pm %.1f$ & $%.1f$ & $%.2f \\pm %.2f$ & ' ...
        '$%.4f \\pm %.4f$ & $%.2f \\pm %.2f$ & $%.1f$ \\\\'], ...
        labels{k}, mObj(k), sObj(k), gap(k), mDist(k), sDist(k), ...
        mCost(k), sCost(k), mWait(k), sWait(k), 100*mServ(k)));
end
fprintf('%s\n', '\bottomrule');
fprintf('%s\n', '\end{tabular}');
fprintf('%s\n', '\end{table}');
fprintf('\n===========================================================\n\n');

%% ==================================================================
%  FIGURES
%  ==================================================================

% ---------- FIGURE 1a: the charging network on a real Washington basemap --
f1 = figure('Position', [80 80 760 700]);
try
    gx = geoaxes(f1);
    % A light basemap keeps the city recognisable without competing with the
    % data. The list is tried in order and the first one that loads is kept:
    % the Esri basemaps need an internet connection and give street detail at
    % city zoom, while the Natural Earth ones are much coarser.
    basemaps = {'streets-light', 'topographic', 'grayland'};
    for b = 1:numel(basemaps)
        try
            geobasemap(gx, basemaps{b});
            break;
        catch
        end
    end
    hold(gx, 'on');
    geoscatter(gx, statLat, statLon, 14 + 7*nPortsPerStation, ...
               [0.00 0.35 0.75], 'filled', 'MarkerFaceAlpha', 0.70, ...
               'MarkerEdgeColor', 'k', 'MarkerEdgeAlpha', 0.35);
    pad = 0.004;
    geolimits(gx, [min(statLat)-pad, max(statLat)+pad], ...
                  [min(statLon)-pad, max(statLon)+pad]);
    title(gx, sprintf('Charging network: %d stations, %d ports', nStations, M));
    set(gx, 'FontSize', 11);
catch
    % Mapping Toolbox unavailable: fall back to a plain scatter
    axLeft = axes(f1);
    hold(axLeft, 'on'); grid(axLeft, 'on'); box(axLeft, 'on');
    scatter(axLeft, statLon, statLat, 14 + 7*nPortsPerStation, ...
            [0.00 0.35 0.75], 'filled', 'MarkerFaceAlpha', 0.70);
    xlabel(axLeft, 'Longitude'); ylabel(axLeft, 'Latitude');
    title(axLeft, sprintf('Charging network: %d stations, %d ports', nStations, M));
    axis(axLeft, 'equal'); set(axLeft, 'FontSize', 11);
end

% Uncomment the next line to export this figure as a vector PDF
% exportgraphics(f1, sprintf('Figures/realworld_network_%s.pdf', cityTag), 'ContentType', 'vector');

% ---------- FIGURE 1b: the optimal assignment of the single instance ------
f1b = figure('Position', [120 80 760 700]);
ax  = axes(f1b);
hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');

plot(ax, hullPoly, 'FaceColor', [0.90 0.93 0.98], 'FaceAlpha', 0.5, ...
     'EdgeColor', [0.55 0.60 0.70], 'LineWidth', 1.0, 'HandleVisibility', 'off');

[pi_, ej_] = find(x_opt == 1);
for e = 1:numel(pi_)
    plot(ax, [evLon(ej_(e)) portLon(pi_(e))], [evLat(ej_(e)) portLat(pi_(e))], '-', ...
         'Color', [0.10 0.60 0.30 0.35], 'LineWidth', 0.6, 'HandleVisibility', 'off');
end
plot(ax, nan, nan, '-', 'Color', [0.10 0.60 0.30], 'LineWidth', 1.5, ...
     'DisplayName', 'optimal assignment');
scatter(ax, statLon, statLat, 12 + 6*nPortsPerStation, [0.00 0.35 0.75], 'filled', ...
        'MarkerFaceAlpha', 0.75, 'DisplayName', 'charging stations');
scatter(ax, evLon, evLat, 14, [0.85 0.20 0.15], 'filled', ...
        'MarkerFaceAlpha', 0.85, 'DisplayName', 'electric vehicles');

xlabel(ax, 'Longitude'); ylabel(ax, 'Latitude');
title(ax, sprintf('Optimal assignment, %s (M = %d ports, N = %d EVs)', cityName, M, N));
legend(ax, 'Location', 'southwest');
axis(ax, 'equal'); set(ax, 'FontSize', 11);

% Uncomment the next line to export this figure as a vector PDF
% exportgraphics(f1b, sprintf('Figures/realworld_map_%s.pdf', cityTag), 'ContentType', 'vector');

% ---------- FIGURES 2a-2c: one boxplot per criterion ---------------------
% One separate figure per criterion, so that each can be sized and placed
% independently in the paper. Together they tell a single story: on the real
% network the coordinated assignment dominates both baselines on the
% aggregate objective AND on each of its components taken separately.
%
% All three share the same figure size, so that placing them side by side in
% LaTeX with a common width yields a common height.

f2a = figure('Position', [80 100 560 460]);
boxplot_or_bar(objAcc(valid,:), shortLabels);
ylabel('Realised global objective');
title(sprintf('Objective (%d instances)', nValid));
grid on; box on; set(gca, 'FontSize', 11);

% Uncomment the next line to export this figure as a vector PDF
% exportgraphics(f2a, sprintf('Figures/realworld_objective_%s.pdf', cityTag), 'ContentType', 'vector');

f2b = figure('Position', [120 100 560 460]);
boxplot_or_bar(costAcc(valid,:), shortLabels);
ylabel('Mean price per EV [$/kWh]');
title(sprintf('Charging price (%d instances)', nValid));
grid on; box on; set(gca, 'FontSize', 11);

% Uncomment the next line to export this figure as a vector PDF
% exportgraphics(f2b, sprintf('Figures/realworld_price_%s.pdf', cityTag), 'ContentType', 'vector');

f2c = figure('Position', [160 100 560 460]);
boxplot_or_bar(distAcc(valid,:), shortLabels);
ylabel('Mean distance per EV [km]');
title(sprintf('Travel distance (%d instances)', nValid));
grid on; box on; set(gca, 'FontSize', 11);

% Uncomment the next line to export this figure as a vector PDF
% exportgraphics(f2c, sprintf('Figures/realworld_distance_%s.pdf', cityTag), 'ContentType', 'vector');

f2d = figure('Position', [200 100 560 460]);
boxplot_or_bar(waitAcc(valid,:), shortLabels);
ylabel('Mean waiting time per EV [min]');
title(sprintf('Waiting time (%d instances)', nValid));
grid on; box on; set(gca, 'FontSize', 11);

% Uncomment the next line to export this figure as a vector PDF
% exportgraphics(f2d, sprintf('Figures/realworld_waiting_%s.pdf', cityTag), 'ContentType', 'vector');

% ---------- FIGURE 3: where the fleet is sent (station utilisation) ------
% The most informative view is not how the outcomes are distributed but
% WHERE the vehicles end up. Each station is drawn at its real position,
% with marker area and colour proportional to the average number of
% vehicles it serves. Comparing the two panels shows the mechanism behind
% the aggregate numbers: the myopic policy piles the fleet onto the central
% stations, while the optimum spreads it over the network.
utilMean = utilAcc / max(nValid,1);      % average vehicles per station

% Only the optimum and the plain nearest policy are mapped: they are the two
% extremes of the comparison, and adding the intermediate rule would make the
% figure wider without changing the message.
shown = [1 2];
cmax  = max(max(utilMean(:,shown)));     % ONE colour scale for both panels:
                                         % without it the maps cannot be compared

% The two panels cover exactly the same geographical window, so the latitude
% ticks of the right one would repeat the left ones and, being drawn on the
% inner side, would collide with the left map. They are therefore switched
% off: only the first panel carries the latitude axis.
f3 = figure('Position', [80 80 1200 580]);
panelW = 0.42;
panelX = [0.055, 0.480];
for k = 1:numel(shown)
    util_panel(f3, [panelX(k) 0.10 panelW 0.80], statLat, statLon, ...
               utilMean(:,shown(k)), cmax, labels{shown(k)}, k == 1);
end
cb = colorbar('Position', [0.925 0.13 0.016 0.74]);
cb.Label.String = 'Vehicles served per station';
cb.FontSize = 11;

% Uncomment the next line to export this figure as a vector PDF
% exportgraphics(f3, sprintf('Figures/realworld_utilisation_%s.pdf', cityTag), 'ContentType', 'vector');

% The busiest station is the compact summary of the same picture: the more
% myopic the policy, the more demand piles up on a single site.
fprintf('\nStation utilisation, vehicles served by the busiest station:\n');
for k = 1:nMethods
    fprintf('  %-26s %5.1f   (stations left unused: %d of %d)\n', ...
            labels{k}, max(utilMean(:,k)), nnz(utilMean(:,k) < 0.05), nStations);
end

%% ------------------------------------------------------------------
%  Save the figures and collect the results
%  ------------------------------------------------------------------

if ~isempty(figdir)
    % isfolder, not exist(...,'dir'): the latter searches the whole MATLAB
    % path, so a Figures folder belonging to another project would satisfy
    % the test while the relative write to ./Figures still fails.
    if ~isfolder(figdir);  mkdir(figdir);  end
    try
        exportgraphics(f1, fullfile(figdir, sprintf('realworld_network_%s.pdf', cityTag)), 'ContentType', 'vector');
        exportgraphics(f1b, fullfile(figdir, sprintf('realworld_map_%s.pdf', cityTag)), 'ContentType', 'vector');
        exportgraphics(f2a, fullfile(figdir, sprintf('realworld_objective_%s.pdf', cityTag)), 'ContentType', 'vector');
        exportgraphics(f2b, fullfile(figdir, sprintf('realworld_price_%s.pdf', cityTag)), 'ContentType', 'vector');
        exportgraphics(f2c, fullfile(figdir, sprintf('realworld_distance_%s.pdf', cityTag)), 'ContentType', 'vector');
        exportgraphics(f2d, fullfile(figdir, sprintf('realworld_waiting_%s.pdf', cityTag)), 'ContentType', 'vector');
        exportgraphics(f3, fullfile(figdir, sprintf('realworld_utilisation_%s.pdf', cityTag)), 'ContentType', 'vector');
        fprintf('Figures saved in %s, tagged "_%s"\n', fullfile(pwd, figdir), cityTag);
    catch ME
        warning('Could not save the figures: %s', ME.message);
    end
end

results.city       = cityTag;
results.M          = M;
results.N          = N;
results.nStations  = nStations;
results.labels     = labels;
results.single.x   = x_opt;
results.single.time= t_opt;
results.objAcc     = objAcc;
results.distAcc    = distAcc;
results.costAcc    = costAcc;
results.waitAcc    = waitAcc;
results.servAcc    = servAcc;
results.timeAcc    = timeAcc;
results.repaired   = repAcc;
results.valid      = valid;
results.meanObj    = mObj;   results.stdObj  = sObj;
results.meanDist   = mDist;  results.stdDist = sDist;
results.meanCost   = mCost;  results.stdCost = sCost;
results.meanServ   = mServ;  results.stdServ = sServ;
results.meanTime   = mTime;  results.stdTime = sTime;
results.meanD90    = mD90;   results.stdD90   = sD90;
results.meanDmax   = mDmax;  results.stdDmax  = sDmax;
results.meanSpend  = mSpend; results.stdSpend = sSpend;
results.meanUnused = mUnused; results.stdUnused = sUnused;   % per layout
results.neverUsed  = neverUsed;    % across the layouts, as in the table
results.unusedAcc  = unusedAcc;
results.d90Acc     = d90Acc;
results.dmaxAcc    = dmaxAcc;
results.spendAcc   = spendAcc;
results.labels     = labels;
results.gap        = gap;
results.util       = utilMean;    % nStations x nMethods, vehicles per station

end % ===================== end of main function =====================


%% ==================================================================
%  LOCAL FUNCTIONS
%  ==================================================================

function [W, C, P, B, A, nRepaired, Q] = draw_instance(portStation, nStations, portPow, D, ...
                                                    Cmin, Cmax, rhoMin, rhoMax, Qrange, Rmax_km, N)
% DRAW_INSTANCE  Draw the port attributes and build the acceptable matrix.
%
% Price and waiting time are drawn once per STATION and then shared by all
% the ports of that station: a tariff and a queue are properties of the
% site, not of the individual socket. The power, in contrast, is a property
% of the connector and comes from the dataset.
%
% The acceptable matrix follows Definition 2.4, A = delta .* R, with
%   R_ij = 1      if port i is within the driving range of vehicle j;
%   delta_ij = 1  if C_i <= B_j / Q_j, that is if vehicle j can afford it.

    % Station-level attributes, replicated to the ports
    statC = Cmin + (Cmax - Cmin) * rand(nStations,1);   % price [$/kWh]
    statW = rand(nStations,1) * 15;                     % waiting time [min]

    C = statC(portStation);      % M x 1
    W = statW(portStation);      % M x 1
    P = portPow;                 % M x 1, from the connector type

    % Energy demand, one value per vehicle. Drawing it rather than fixing it
    % matters: the objective contains C_i*Q_j and Q_j/P_i, so with a common Q
    % the price term would collapse to a rescaled tariff and the charging
    % time would differ across ports only, never across vehicles.
    Q     = Qrange(1) + (Qrange(2) - Qrange(1)) * rand(1,N);   % 1 x N [kWh]

    % Affordability threshold, drawn directly, and the budget that follows
    % from it. See the note in the settings on why the ratio and not B.
    ratio = rhoMin + (rhoMax - rhoMin) * rand(1,N);     % 1 x N  [$/kWh]
    B     = (ratio .* Q)';                              % N x 1  [$]

    % Reachability and affordability, then the acceptable matrix
    R     = D <= Rmax_km;                               % M x N
    delta = C <= ratio;                                 % M x N by expansion
    A     = double(R & delta);

    % REPAIR OF STRANDED VEHICLES. A vehicle sampled at the edge of the
    % network may find no port that is at once within range and within
    % budget. Such a vehicle is not a modelling case of interest: the model
    % assumes that every EV in the system needs a charge AND can obtain one.
    % For those vehicles only, we grant reachability of their nearest
    % affordable port, which amounts to assuming the driver stretches the
    % trip slightly. The count is returned so the intervention stays visible.
    stranded  = find(sum(A,1) == 0);
    nRepaired = numel(stranded);
    for j = stranded
        affordable_j = find(delta(:,j));
        if isempty(affordable_j)
            [~, i0] = min(D(:,j));           % not even affordable: nearest
        else
            [~, k0] = min(D(affordable_j, j));
            i0 = affordable_j(k0);
        end
        A(i0, j) = 1;
    end
end


function [evLat, evLon] = sample_ev_positions(hullPoly, centLat, centLon, N, sigmaKm)
% SAMPLE_EV_POSITIONS  Draw N vehicle positions inside the network hull.
% Positions lie inside the concave hull of the stations, with a Gaussian
% bias towards the network centroid, so that demand is denser in the centre
% of the city. Sampling is by rejection.

    degkmLat = 110.574;
    degkmLon = 111.320 * cosd(centLat);

    vx = hullPoly.Vertices(:,1);  vy = hullPoly.Vertices(:,2);
    lonMin = min(vx);  lonMax = max(vx);
    latMin = min(vy);  latMax = max(vy);

    evLat = zeros(N,1);  evLon = zeros(N,1);
    got = 0;
    while got < N
        L    = max(4*N, 2000);
        cLon = rand(L,1)*(lonMax-lonMin) + lonMin;
        cLat = rand(L,1)*(latMax-latMin) + latMin;

        inside = isinterior(hullPoly, cLon, cLat);
        dLat   = (cLat - centLat) * degkmLat;
        dLon   = (cLon - centLon) * degkmLon;
        wgt    = exp(-(dLat.^2 + dLon.^2) / (2*sigmaKm^2));
        acc    = inside & (rand(L,1) < wgt);

        idx  = find(acc);
        take = min(numel(idx), N - got);
        evLat(got+1:got+take) = cLat(idx(1:take));
        evLon(got+1:got+take) = cLon(idx(1:take));
        got = got + take;
    end
end


function D = haversine_matrix(pLat, pLon, vLat, vLon)
% HAVERSINE_MATRIX  Great-circle distances between ports and vehicles [km].

    Rearth = 6371;
    d2r = @(z) z * pi / 180;

    latP = d2r(pLat(:));    lonP = d2r(pLon(:));      % M x 1
    latV = d2r(vLat(:)');   lonV = d2r(vLon(:)');     % 1 x N

    dlat = latP - latV;
    dlon = lonP - lonV;
    a    = sin(dlat/2).^2 + (cos(latP) .* cos(latV)) .* sin(dlon/2).^2;
    D    = Rearth * 2 .* atan2(sqrt(a), sqrt(1-a));
end


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


function x = greedy_cost(c, A, order)
% GREEDY_COST  Sequential best response on the FULL criterion.
%
% Same sequential commitment as the other greedy rules, but each vehicle
% minimises the complete cost c_ij that the proposed model minimises
% globally, instead of a proxy for it. It is the best response of a
% perfectly informed but isolated driver, and it is what separates the
% value of coordinating from the value of looking at the right criterion.
% See greedy_cost_assignment.m for the full discussion.

    [M, N] = size(c);
    if nargin < 3 || isempty(order);  order = randperm(N);  end

    taken = false(M,1);
    x     = zeros(M,N);
    for k = 1:N
        j    = order(k);
        cand = find(A(:,j) == 1 & ~taken);
        if isempty(cand);  continue;  end          % this EV stays unserved
        [~, kmin] = min(c(cand,j));
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


function m = assignment_stats(x, c, D, W, C, N, Q, portStation, nStations)
% ASSIGNMENT_STATS  Performance indicators of one assignment.
% Averages are over the SERVED vehicles; "served" is the fraction of the
% fleet that obtained a port.
%
% Besides the averages, two TAIL statistics of the travel distance are
% returned, the 90th percentile and the maximum over the served vehicles.
% The reason is that the model minimises a SUM, so an assignment can lower
% the average by sending a few peripheral vehicles very far, and the mean
% alone cannot reveal it. The question "who pays for coordination" is
% answered by the tail, not by the centre, of the distribution.
%
% The seventh argument is optional. When the per-vehicle energy demand Q is
% supplied, the field "spend" gives the money actually paid for the charge,
% C_i * Q_j averaged over the served vehicles. Unlike the tariff, which is
% a rate, this is the quantity that enters the objective and the budget
% constraint, and it is what separates a policy that finds cheap ports from
% one that merely finds close ones.

    x  = full(double(x));
    Na = nnz(x);

    m.obj    = full(sum(sum(c .* x)));
    m.served = Na / N;

    if Na == 0
        [m.dist, m.cost, m.wait, m.d90, m.dmax, m.spend, m.unused] = deal(NaN);
        return;
    end

    m.dist = full(sum(sum(D .* x))) / Na;
    m.cost = full(sum(C .* sum(x,2))) / Na;
    m.wait = full(sum(W .* sum(x,2))) / Na;

    % Distance actually travelled by each SERVED vehicle. Unserved columns
    % are all-zero and would enter the quantile as a spurious 0 km trip,
    % which would flatter a policy precisely where it performs worst.
    dvec   = full(sum(D .* x, 1));            % 1 x N
    served = full(sum(x, 1)) > 0;
    dvec   = dvec(served);

    m.d90  = quantile(dvec, 0.90);
    m.dmax = max(dvec);

    if nargin >= 7 && ~isempty(Q)
        Qrow    = reshape(full(Q), 1, []);
        spend   = full(C' * x) .* Qrow;       % 1 x N, dollars paid by each EV
        m.spend = sum(spend(served)) / Na;
    else
        m.spend = NaN;
    end

    % Stations that receive no vehicle at all. Counting idle PORTS would be
    % pointless, since every method serves the same N vehicles with one
    % socket each and therefore leaves exactly M - N ports idle, whatever
    % the assignment. Stations discriminate, because a policy that
    % concentrates the fleet on a few sites leaves whole sites untouched.
    if nargin >= 9 && ~isempty(portStation)
        perStation = accumarray(portStation, sum(x,2), [nStations 1]);
        m.unused   = nnz(perStation == 0);
    else
        m.unused = NaN;
    end
end


function [m, tmean, last] = avg_greedy(runFcn, ORDERS, c, D, W, C, N, Q, portStation, nStations)
% AVG_GREEDY  Average a greedy policy over ORDERS random processing orders.
% A decentralised policy depends on who commits to a port first, so a single
% run is not representative. The assignment of the LAST run is returned in
% "last", to feed the station-utilisation figure.
%
% Every field produced by assignment_stats is averaged, tail statistics
% included. Averaging a 90th percentile across processing orders is a
% legitimate thing to do here, since each order is a separate realisation
% of the same policy and we report the expected tail of that policy, not
% the tail of the pooled sample.

    if nargin < 8,   Q = [];            end
    if nargin < 10,  portStation = [];  nStations = [];  end

    fields = {'obj','dist','cost','wait','served','d90','dmax','spend','unused'};
    acc    = cell2struct(num2cell(zeros(numel(fields),1)), fields, 1);
    ttot   = 0;
    last   = struct('x', []);

    for o = 1:ORDERS
        t0 = tic;
        x  = runFcn();
        ttot = ttot + toc(t0);

        s = assignment_stats(x, c, D, W, C, N, Q, portStation, nStations);
        for f = 1:numel(fields)
            acc.(fields{f}) = acc.(fields{f}) + s.(fields{f});
        end

        % The assignment of the last run is returned, so that the caller can
        % compute station utilisation from it.
        last.x = x;
    end

    for f = 1:numel(fields)
        m.(fields{f}) = acc.(fields{f}) / ORDERS;
    end
    tmean = ttot / ORDERS;
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


function util_panel(figHandle, pos, sLat, sLon, util, cmax, ttl, showLat)
% UTIL_PANEL  One map panel showing how much each station is used.
% Marker area and colour both encode the average number of vehicles served,
% so that busy sites stand out even in the dense downtown cluster. The
% colour limits are passed in, so that the two panels share one scale and
% can therefore be compared.
%
% showLat (optional, default true) draws the latitude axis. Set it to false
% for every panel after the first: the panels share the same window, so the
% repeated tick labels carry no information and overlap the map to their left.

    if nargin < 8 || isempty(showLat);  showLat = true;  end

    try
        gx = geoaxes(figHandle, 'Position', pos);
        basemaps = {'streets-light', 'topographic', 'grayland'};
        for b = 1:numel(basemaps)
            try
                geobasemap(gx, basemaps{b});
                break;
            catch
            end
        end
        hold(gx, 'on');
        geoscatter(gx, sLat, sLon, 12 + 60*util/max(cmax,eps), util, 'filled', ...
                   'MarkerFaceAlpha', 0.80, 'MarkerEdgeColor', 'k', ...
                   'MarkerEdgeAlpha', 0.30);
        pad = 0.004;
        geolimits(gx, [min(sLat)-pad, max(sLat)+pad], ...
                      [min(sLon)-pad, max(sLon)+pad]);
        title(gx, ttl);
        set(gx, 'FontSize', 11);
        colormap(gx, turbo);
        clim(gx, [0 cmax]);
        if ~showLat
            % Blank the tick labels rather than hiding the ruler, so that the
            % grid lines survive and the two maps still look identical.
            gx.LatitudeAxis.TickLabels = repmat({''}, size(gx.LatitudeAxis.TickValues));
            gx.LatitudeLabel.String    = '';
        end
    catch
        % Mapping Toolbox unavailable: plain scatter in lon/lat
        ax = axes(figHandle, 'Position', pos);
        hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
        scatter(ax, sLon, sLat, 12 + 60*util/max(cmax,eps), util, 'filled', ...
                'MarkerFaceAlpha', 0.80);
        xlabel(ax, 'Longitude');
        if showLat
            ylabel(ax, 'Latitude');
        else
            set(ax, 'YTickLabel', []);
        end
        title(ax, ttl); axis(ax, 'equal');
        set(ax, 'FontSize', 11);
        colormap(ax, turbo);
        clim(ax, [0 cmax]);
    end
end


function out = ternary(cond, a, b)
% Small helper for inline conditional strings.
    if cond;  out = a;  else;  out = b;  end
end

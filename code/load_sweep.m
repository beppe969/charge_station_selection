function results = load_sweep(csvFile, figdir, cityTag, REPS)
% LOAD_SWEEP  How the value of coordination changes with occupancy (Sec. 4.3).
%
%   results = load_sweep(csvFile, figdir, cityTag, REPS)
%
% Repeats the case study at N/M = 0.25, 0.50, 0.75 and 0.95. The middle
% ratio is the configuration of Section 4.1 and reproduces it, which is a
% free consistency check.
%
% Two things keep the comparison honest. Everything is per SERVED vehicle,
% because the sequential rules abandon a growing share of the fleet as the
% load rises and a sum over fewer terms would flatter them. And feasibility
% is verified with a maximum-cardinality matching before each solve, since
% at high occupancy a complete assignment need not exist.
%
% Produces load_sweep_<city>.pdf and a LaTeX table.


    if nargin < 2 || isempty(figdir),  figdir = 'Figures';  end
    if nargin < 4 || isempty(REPS),    REPS   = 30;         end
    if nargin < 3 || isempty(cityTag)
        [~, base] = fileparts(csvFile);
        if     contains(lower(base), 'dc'),    cityTag = 'Washington';
        elseif contains(lower(base), 'chic'),  cityTag = 'Chicago';
        elseif contains(lower(base), 'manh') || contains(lower(base), 'ny')
                                               cityTag = 'Manhattan';
        else,                                  cityTag = base;
        end
    end

%% ==================================================================
%  SETTINGS
%  ==================================================================

loadGrid = [0.25 0.50 0.75 0.95];   % N / M
Rmax_km  = 5.0;                     % as in the case studies
Qrange   = [20 60];                 % [kWh]
rhoMin   = 0.40;   rhoMax = 0.60;   % [$/kWh]
kNear    = 3;
ORDERS   = 5;                       % processing orders per greedy run
sigmaKm  = 2.5;
BIGM     = 1e6;

Cmin = 0.303;  Cmax = 0.530;                      % [$/kWh]
pow_level1 = 7;  pow_level2 = 19;  pow_dcfast = 50;  % [kW]

alpha = objective_weights();

rng(11, 'twister');

if exist('hungarian_assignment_sparse', 'file') == 2
    solve_fun = @hungarian_assignment_sparse;
else
    warning('hungarian_assignment_sparse not found: using the dense version.');
    solve_fun = @hungarian_assignment_binary;
end

%% ==================================================================
%  PART 0: load the network and expand the stations into ports
%  ==================================================================

assert(isfile(csvFile), 'Cannot find %s', csvFile);

[~, ~, dataExt] = fileparts(csvFile);
if any(strcmpi(dataExt, {'.csv', '.txt', '.dat'}))
    T = readtable(csvFile, 'Delimiter', ';', 'TextType', 'string');
else
    T = readtable(csvFile, 'TextType', 'string');
end

toNum = @(z) str2double(strrep(string(z), ',', '.'));

statLat = toNum(T.Latitude);   statLon = toNum(T.Longitude);
nLevel1 = round(toNum(T.EVLevel1EVSENum));
nLevel2 = round(toNum(T.EVLevel2EVSENum));
nDCfast = round(toNum(T.EVDCFastCount));

ok = isfinite(statLat) & isfinite(statLon);
statLat = statLat(ok);  statLon = statLon(ok);
nLevel1 = nLevel1(ok);  nLevel2 = nLevel2(ok);  nDCfast = nDCfast(ok);
nLevel1(~isfinite(nLevel1)) = 0;
nLevel2(~isfinite(nLevel2)) = 0;
nDCfast(~isfinite(nDCfast)) = 0;

nStations = numel(statLat);

portLat = [];  portLon = [];  portPow = [];  portStation = [];
for s = 1:nStations
    counts = [nLevel1(s), nLevel2(s), nDCfast(s)];
    powers = [pow_level1, pow_level2, pow_dcfast];
    if sum(counts) == 0,  counts = [0 1 0];  end
    for k = 1:3
        for r = 1:counts(k)
            portLat(end+1,1)     = statLat(s);   %#ok<AGROW>
            portLon(end+1,1)     = statLon(s);   %#ok<AGROW>
            portPow(end+1,1)     = powers(k);    %#ok<AGROW>
            portStation(end+1,1) = s;            %#ok<AGROW>
        end
    end
end

M = numel(portLat);

centLat = mean(statLat, 'omitnan');
centLon = mean(statLon, 'omitnan');
try
    K = boundary(statLon, statLat, 0.6);
catch
    K = convhull(statLon, statLat);
end
hullPoly = simplify(rmholes(polyshape(statLon(K), statLat(K))));

nLoad    = numel(loadGrid);
nMethods = 5;
labels   = {'Optimal (proposed model)', 'Greedy nearest', ...
            sprintf('Cheapest of %d-nearest', kNear), ...
            'Greedy on full cost', 'Min-distance matching'};

fprintf('\n============ LOAD SWEEP: %s ============\n', upper(cityTag));
fprintf('  ports M       : %d\n', M);
fprintf('  load ratios   : %s\n', mat2str(loadGrid));
fprintf('  layouts each  : %d\n', REPS);
fprintf('  all quantities are PER SERVED VEHICLE\n');
fprintf('===============================================\n\n');

%% ==================================================================
%  PART 1: the sweep
%  ==================================================================

objPS   = nan(nLoad, nMethods);   % objective per served vehicle, mean
objPSsd = nan(nLoad, nMethods);
servPct = nan(nLoad, nMethods);   % share of the fleet served
distM   = nan(nLoad, nMethods);
d90M    = nan(nLoad, nMethods);
spendM  = nan(nLoad, nMethods);
Nvec    = nan(nLoad, 1);
nInfeas = zeros(nLoad, 1);

for a = 1:nLoad

    N = round(loadGrid(a) * M);
    Nvec(a) = N;

    accObj  = nan(REPS, nMethods);
    accServ = nan(REPS, nMethods);
    accDist = nan(REPS, nMethods);
    accD90  = nan(REPS, nMethods);
    accSpd  = nan(REPS, nMethods);

    for rep = 1:REPS

        [evLat, evLon] = sample_ev_positions(hullPoly, centLat, centLon, N, sigmaKm);
        D = haversine_matrix(portLat, portLon, evLat, evLon);

        [W, C, P, A, Q] = draw_instance(portStation, nStations, portPow, D, ...
                              Cmin, Cmax, rhoMin, rhoMax, Qrange, Rmax_km, N);

        % Feasibility first. See the header for why the solver cannot be
        % trusted to report it on its own.
        [~, nMatch] = hopcroft_karp(A);
        if nMatch < N
            nInfeas(a) = nInfeas(a) + 1;
            continue;
        end

        c = build_objective_matrix(D, W, C, P, Q, alpha);

        % ---- (1) the proposed model
        H = c;  H(~A) = BIGM;
        xo = full(double(solve_fun(A, H)));
        if nnz(xo) < N || any(~A(logical(xo)))
            nInfeas(a) = nInfeas(a) + 1;
            continue;
        end

        % ---- (4) minimum total distance, same solver, different criterion
        Hd = D;  Hd(~A) = BIGM;
        xd = full(double(solve_fun(A, Hd)));

        % ---- (2)(3)(4) the decentralised rules, averaged over orders.
        % The third commits ports sequentially like the others but scores
        % them with the full criterion c, so it separates the value of
        % coordinating from the value of looking at the right criterion.
        s2 = avg_policy(@() greedy_nearest_assignment(D, A), ...
                        ORDERS, c, D, C, N, Q);
        s3 = avg_policy(@() greedy_cheapest_k_nearest(D, C, A, kNear), ...
                        ORDERS, c, D, C, N, Q);
        s4 = avg_policy(@() greedy_cost_assignment(c, A), ...
                        ORDERS, c, D, C, N, Q);

        s1 = per_served(xo, c, D, C, N, Q);
        s5 = per_served(xd, c, D, C, N, Q);

        S = {s1, s2, s3, s4, s5};
        for k = 1:nMethods
            accObj(rep,k)  = S{k}.objPS;
            accServ(rep,k) = S{k}.served;
            accDist(rep,k) = S{k}.dist;
            accD90(rep,k)  = S{k}.d90;
            accSpd(rep,k)  = S{k}.spend;
        end
    end

    good = all(isfinite(accObj), 2);
    objPS(a,:)   = mean(accObj(good,:), 1);
    objPSsd(a,:) = std( accObj(good,:), 0, 1);
    servPct(a,:) = mean(accServ(good,:), 1);
    distM(a,:)   = mean(accDist(good,:), 1);
    d90M(a,:)    = mean(accD90(good,:), 1);
    spendM(a,:)  = mean(accSpd(good,:), 1);

    gp = 100 * (objPS(a,:) - objPS(a,1)) / objPS(a,1);
    fprintf(['  N/M = %.2f (N = %4d) | gap: nearest %5.2f%%, cheapest-%d ' ...
             '%5.2f%%, greedy-cost %5.2f%%, min-dist %5.2f%% | served: ' ...
             '%5.1f%% (nearest) %5.1f%% (greedy-cost)'], ...
            loadGrid(a), N, gp(2), kNear, gp(3), gp(4), gp(5), ...
            100*servPct(a,2), 100*servPct(a,4));
    if nInfeas(a) > 0
        fprintf(' | %d infeasible layouts', nInfeas(a));
    end
    fprintf('\n');
end

gap = 100 * (objPS - objPS(:,1)) ./ objPS(:,1);

%% ==================================================================
%  PART 2: figure
%  ==================================================================

% ---- one colour and one marker per METHOD, shared by both panels.
% Without an explicit assignment the two panels would disagree: the left
% one omits the optimum, which is the reference at zero, so MATLAB would
% shift the colour order by one and the same method would appear in two
% different colours depending on the panel.
pal   = lines(5);
colOf = [5 1 2 3 4];                    % method -> row of pal
mkOf  = {'d', 'o', 's', 'v', '^'};      % method -> marker
% Only the optimum is dashed: in the right panel it coincides with the
% min-distance matching at 100%, and the two would otherwise be drawn one
% exactly on top of the other.
lsOf  = {'--', '-', '-', '-', '-'};

fig = figure('Color','w','Position',[100 100 1000 390]);

subplot(1,2,1);
hold on; grid on; box on;
for k = 2:nMethods
    plot(loadGrid, gap(:,k), [lsOf{k} mkOf{k}], 'LineWidth', 1.9, ...
         'Color', pal(colOf(k),:), 'MarkerFaceColor', 'w', ...
         'DisplayName', labels{k});
end
xlabel('Load ratio N / M');
ylabel('Excess cost over the optimum [%]');
title('Value of coordination against the load');
legend('Location','northeast','Box','off');
set(gca,'FontSize',11);
hold off;

subplot(1,2,2);
hold on; grid on; box on;
% The two coordinated methods sit exactly on top of each other at 100%,
% since both are matchings and neither can strand a vehicle; that they
% coincide whatever their criterion is the point of the panel. The optimum
% is therefore drawn dashed AND last, so that it lies above the solid line
% of the min-distance matching instead of being buried under it. The
% handles are collected so that the legend keeps the method order even
% though the drawing order does not.
h = gobjects(1, nMethods);
for k = [2:nMethods, 1]
    h(k) = plot(loadGrid, 100*servPct(:,k), [lsOf{k} mkOf{k}], ...
                'LineWidth', 1.9, 'Color', pal(colOf(k),:), ...
                'MarkerFaceColor', 'w');
end
xlabel('Load ratio N / M');
ylabel('Share of the fleet served [%]');
title('Vehicles served');
legend(h, labels, 'Location','southwest','Box','off');
set(gca,'FontSize',11);
hold off;

if ~isfolder(figdir),  mkdir(figdir);  end
outPdf = fullfile(figdir, sprintf('load_sweep_%s.pdf', cityTag));
exportgraphics(fig, outPdf, 'ContentType', 'vector');
fprintf('\n  figure written to %s\n', outPdf);

%% ==================================================================
%  PART 3: paste-ready LaTeX table
%  ==================================================================

fprintf('\n%% ---------- copy from here ----------\n');
fprintf('\\begin{table}[t]\n\\centering\n');
fprintf(['\\caption{Effect of the load ratio on the value of coordination, ' ...
         '%s network ($M=%d$ ports), over $%d$ random layouts per row. ' ...
         'All quantities are per served vehicle, since the decentralised ' ...
         'rules leave part of the fleet unserved as the load grows.}\n'], ...
        cityTag, M, REPS);
fprintf('\\label{tab:load_sweep_%s}\n\\small\n', lower(cityTag));
fprintf('\\setlength{\\tabcolsep}{5pt}\n');
fprintf('\\begin{tabular}{cccccccc}\n\\toprule\n');
fprintf([' & & \\multicolumn{4}{c}{\\textbf{Excess cost over the optimum [\\%%]}} & ' ...
         '\\multicolumn{2}{c}{\\textbf{Fleet served [\\%%]}} \\\\\n']);
fprintf('\\cmidrule(lr){3-6}\\cmidrule(lr){7-8}\n');
fprintf(['$N/M$ & $N$ & Nearest & Cheapest-%d & Greedy-cost & Min-dist & ' ...
         'Nearest & Greedy-cost \\\\\n\\midrule\n'], kNear);
for a = 1:nLoad
    fprintf('$%.2f$ & $%d$ & $%.1f$ & $%.1f$ & $%.1f$ & $%.1f$ & $%.1f$ & $%.1f$ \\\\\n', ...
            loadGrid(a), Nvec(a), gap(a,2), gap(a,3), gap(a,4), gap(a,5), ...
            100*servPct(a,2), 100*servPct(a,4));
end
fprintf('\\bottomrule\n\\end{tabular}\n\\end{table}\n');
fprintf('%% ---------- to here ----------\n\n');

%% ==================================================================
%  OUTPUT
%  ==================================================================

results = struct('city', cityTag, 'M', M, 'N', Nvec, 'REPS', REPS, ...
                 'loadGrid', loadGrid, 'labels', {labels}, ...
                 'objPerServed', objPS, 'objPerServedStd', objPSsd, ...
                 'gap', gap, 'servedFraction', servPct, ...
                 'meanDist', distM, 'meanD90', d90M, 'meanSpend', spendM, ...
                 'infeasibleLayouts', nInfeas);

save(sprintf('load_sweep_%s.mat', cityTag), 'results');
fprintf('  results saved to load_sweep_%s.mat\n\n', cityTag);
end


%% ==================================================================
%  LOCAL FUNCTIONS
%  ==================================================================

function m = per_served(x, c, D, C, N, Q)
% PER_SERVED  Indicators of one assignment, normalised by served vehicles.
%
% The objective is divided by the number of vehicles that actually obtained
% a port. Without this, a policy that abandons part of the fleet would
% appear cheaper than one that serves it, which is the opposite of what the
% comparison is meant to measure.

    x  = full(double(x));
    Na = nnz(x);

    m.served = Na / N;
    if Na == 0
        [m.objPS, m.dist, m.d90, m.spend] = deal(NaN);
        return;
    end

    m.objPS = full(sum(sum(c .* x))) / Na;

    dvec   = full(sum(D .* x, 1));
    servd  = full(sum(x, 1)) > 0;
    dvec   = dvec(servd);
    m.dist = mean(dvec);
    m.d90  = quantile(dvec, 0.90);

    Qrow    = reshape(full(Q), 1, []);
    spend   = full(C' * x) .* Qrow;
    m.spend = sum(spend(servd)) / Na;
end


function m = avg_policy(runFcn, ORDERS, c, D, C, N, Q)
% AVG_POLICY  Average a decentralised rule over random processing orders.

    fields = {'objPS','served','dist','d90','spend'};
    acc    = cell2struct(num2cell(zeros(numel(fields),1)), fields, 1);

    for o = 1:ORDERS
        s = per_served(runFcn(), c, D, C, N, Q);
        for f = 1:numel(fields)
            acc.(fields{f}) = acc.(fields{f}) + s.(fields{f});
        end
    end
    for f = 1:numel(fields)
        m.(fields{f}) = acc.(fields{f}) / ORDERS;
    end
end


function [W, C, P, A, Q] = draw_instance(portStation, nStations, portPow, D, ...
                              Cmin, Cmax, rhoMin, rhoMax, Qrange, Rmax_km, N)
% DRAW_INSTANCE  Port attributes and acceptable matrix, as in Section 4.3.
% Price and waiting time are drawn once per STATION and shared by its
% ports; the power comes from the connector type in the dataset.

    statC = Cmin + (Cmax - Cmin) * rand(nStations,1);
    statW = rand(nStations,1) * 15;

    C = statC(portStation);
    W = statW(portStation);
    P = portPow;

    Q     = Qrange(1) + (Qrange(2) - Qrange(1)) * rand(1,N);   % [kWh]
    ratio = rhoMin + (rhoMax - rhoMin) * rand(1,N);            % [$/kWh]

    R     = D <= Rmax_km;
    delta = C <= ratio;
    A     = double(R & delta);

    % Same repair convention as the case studies: a vehicle with no
    % acceptable port is granted its nearest affordable one, because a
    % stranded vehicle is a different problem from the one studied here.
    stranded = find(sum(A,1) == 0);
    for j = stranded
        affordable_j = find(delta(:,j));
        if isempty(affordable_j)
            [~, i0] = min(D(:,j));
        else
            [~, k0] = min(D(affordable_j, j));
            i0 = affordable_j(k0);
        end
        A(i0, j) = 1;
    end
end


function [evLat, evLon] = sample_ev_positions(hullPoly, centLat, centLon, N, sigmaKm)
% SAMPLE_EV_POSITIONS  N vehicle positions inside the network hull, with a
% Gaussian bias towards the centroid. Sampling is by rejection.

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


function Dm = haversine_matrix(pLat, pLon, vLat, vLon)
% HAVERSINE_MATRIX  Great-circle distances between ports and vehicles [km].

    Rearth = 6371;
    d2r = @(z) z * pi / 180;

    latP = d2r(pLat(:));    lonP = d2r(pLon(:));
    latV = d2r(vLat(:)');   lonV = d2r(vLon(:)');

    dlat = latP - latV;
    dlon = lonP - lonV;
    a    = sin(dlat/2).^2 + (cos(latP) .* cos(latV)) .* sin(dlon/2).^2;
    Dm   = Rearth * 2 .* atan2(sqrt(a), sqrt(1-a));
end

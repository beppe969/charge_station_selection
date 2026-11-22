function S = sensitivity_alpha(csvFile, figdir)
% Legacy MATLAB exploration. The revised manuscript and its paired uncertainty
% estimates are reproduced by revised_city_study.py.
% SENSITIVITY_ALPHA  Does the result survive the coefficients? (Sec. 4.4)
%
% Re-solves the SAME instances of one real network at every point of a wide
% grid over (alpha_1, alpha_3), so that nothing changes across the sweep
% except the coefficients. The calibrated pair is inserted into the grid, so
% the sentence "at the calibrated setting" can be read off the figure
% instead of interpolated between two neighbouring runs.
%
% Everything is per served vehicle: the greedy rule occasionally leaves a
% vehicle unserved and comparing totals would flatter it.
%
% Prints the numbers used in the robustness paragraph, including the point
% where the distance comparison changes sign.


if nargin < 1 || isempty(csvFile);  csvFile = 'CS_location_DC_complete.csv';  end
if nargin < 2;                      figdir  = 'Figures';                      end
if ~isempty(figdir) && ~isfolder(figdir)
    [okDir, msgDir] = mkdir(figdir);
    if ~okDir, error('Cannot create %s: %s', figdir, msgDir); end
end

rng(1);

%% ---------------------------------------------------------------- settings
load_ratio = 0.5;
Rmax_km    = 5.0;
Qrange     = [20 60];   % energy demand per EV [kWh], drawn per vehicle
rhoMin = 0.40;  rhoMax = 0.60;   % willingness to pay [$/kWh]; the
                                 % budget follows as B_j = rho_j*Q_j
Cmin = 0.303;  Cmax = 0.530;
sigmaKm = 2.5;
REPS    = 20;           % instances per setting
ORDERS  = 5;            % random processing orders for the greedy policy

pow_level1 = 1.4;  pow_level2 = 7.2;  pow_dcfast = 50;

% The sweep. alpha_1 spans a factor of about three hundred, which brackets
% any defensible generalised cost of an urban kilometre; alpha_3 takes a
% low, a central and a high value of time.
%
% The grids are built so that the CALIBRATED pair is one of the points
% actually evaluated. Otherwise the sentence "at the calibrated setting the
% comparison behaves as follows" could not be read off the figure, and the
% value quoted in the text would have to be interpolated between two
% neighbouring runs, which is exactly the kind of small inconsistency a
% referee notices. The calibrated values are inserted into the grids and
% duplicates removed, so re-calibrating automatically moves the point.
alphaStarNow = objective_weights();
alphaDist = unique([0.25 0.5 1 2 5 10 20 40 80, alphaStarNow.dist]);  % [$/km]
alphaTime = unique([0.15 0.35 0.70,              alphaStarNow.time]); % [$/min]
fprintf(['Calibrated setting included in the sweep: alpha_1 = %.3f $/km, ' ...
         'alpha_3 = %.3f $/min\n'], alphaStarNow.dist, alphaStarNow.time);

solve_fun = @matchpairs_assignment;

%% ------------------------------------------------- load and expand network
assert(isfile(csvFile), 'Cannot find %s', csvFile);
[~, ~, ext] = fileparts(csvFile);
if any(strcmpi(ext, {'.csv','.txt','.dat'}))
    T = readtable(csvFile, 'Delimiter', ';', 'TextType', 'string');
else
    T = readtable(csvFile, 'TextType', 'string');
end
toNum = @(z) str2double(strrep(string(z), ',', '.'));

statLat = toNum(T.Latitude);   statLon = toNum(T.Longitude);
n1 = round(toNum(T.EVLevel1EVSENum));
n2 = round(toNum(T.EVLevel2EVSENum));
nf = round(toNum(T.EVDCFastCount));
ok = isfinite(statLat) & isfinite(statLon);
statLat = statLat(ok);  statLon = statLon(ok);
n1 = n1(ok);  n2 = n2(ok);  nf = nf(ok);
n1(~isfinite(n1)) = 0;  n2(~isfinite(n2)) = 0;  nf(~isfinite(nf)) = 0;
nStations = numel(statLat);

portLat = [];  portLon = [];  portPow = [];  portStation = [];
for s = 1:nStations
    counts = [n1(s), n2(s), nf(s)];
    powers = [pow_level1, pow_level2, pow_dcfast];
    if sum(counts) == 0, counts = [0 1 0]; end
    for k = 1:3
        for r = 1:counts(k)
            portLat(end+1,1) = statLat(s);  portLon(end+1,1) = statLon(s); %#ok<AGROW>
            portPow(end+1,1) = powers(k);   portStation(end+1,1) = s;      %#ok<AGROW>
        end
    end
end
M = numel(portLat);
N = round(load_ratio * M);

centLat = mean(statLat);  centLon = mean(statLon);
try
    K = boundary(statLon, statLat, 0.6);
catch
    K = convhull(statLon, statLat);
end
hullPoly = simplify(rmholes(polyshape(statLon(K), statLat(K))));

fprintf('\n=== Sensitivity of the results to the objective coefficients ===\n');
fprintf('  network  : %s\n', csvFile);
fprintf('  ports M  : %d,  EVs N : %d,  instances per setting : %d\n', M, N, REPS);
fprintf('  sweep    : %d values of alpha_1 x %d values of alpha_3 (alpha_2 = 1)\n\n', ...
        numel(alphaDist), numel(alphaTime));

%% ------------------------------------------------- pre-draw the instances
% The SAME instances are reused for every coefficient setting. This is not a
% detail: it removes the sampling noise from the comparison, so that whatever
% moves across the sweep is caused by the coefficients and by nothing else.
inst = cell(REPS,1);
for r = 1:REPS
    [evLat, evLon] = local_sample(hullPoly, centLat, centLon, N, sigmaKm);
    D = local_haversine(portLat, portLon, evLat, evLon);
    Cst = Cmin + (Cmax-Cmin)*rand(nStations,1);
    Wst = 15*rand(nStations,1);
    C = Cst(portStation);   W = Wst(portStation);
    Q   = Qrange(1) + (Qrange(2)-Qrange(1))*rand(1,N);
    rho = rhoMin + (rhoMax-rhoMin)*rand(1,N);   % [$/kWh], budget = rho.*Q
    A   = (D <= Rmax_km) & (C <= repmat(rho, M, 1));
    ord = zeros(ORDERS, N);
    for o = 1:ORDERS, ord(o,:) = randperm(N); end
    inst{r} = struct('D',D,'C',C,'W',W,'A',A,'ord',ord,'Q',Q);
end

%% ------------------------------------------------------------- the sweep
nD = numel(alphaDist);  nT = numel(alphaTime);
nUnserved = 0;          % greedy vehicles left without a port, over the whole run
gap   = nan(nD, nT);    % greedy objective excess over the optimum   [%]
relD  = nan(nD, nT);    % optimum distance relative to greedy        [%]
relC  = nan(nD, nT);    % optimum price    relative to greedy        [%]
relW  = nan(nD, nT);    % optimum wait     relative to greedy        [%]

relBill = nan(nD,nT); relDowntime = nan(nD,nT);
for a = 1:nD
  for b = 1:nT
        alpha = struct('dist', alphaDist(a), 'price', 1, 'time', alphaTime(b));
        [g, rd, rc, rw, rb, rt] = deal(nan(REPS,1));
        for r = 1:REPS
            I = inst{r};
            c = build_objective_matrix(I.D, I.W, I.C, portPow, I.Q, alpha);

            % Compare each sequential order against the exact allocation of
            % the same EVs. The physical quantities are computed on this
            % fixed cohort, including on layouts without a full matching.
            bill = I.C * I.Q;
            downtime = I.W + 60 .* I.Q ./ portPow;
            rel = nan(ORDERS, 6);
            for o = 1:ORDERS
                xg = logical(greedy_nearest_assignment(I.D, I.A, I.ord(o,:)));
                served = any(xg, 1);
                nUnserved = nUnserved + nnz(~served);
                if ~any(served), continue; end
                xb = false(M,N);
                xb(:,served) = logical(solve_fun(I.A(:,served), c(:,served)));
                assert(nnz(xb) == nnz(served) && all(I.A(xb)), 'Invalid cohort assignment');
                cg = sum(c(xg)); cb = sum(c(xb));
                chosen_g = sum(xg,2); chosen_b = sum(xb,2);
                dg = sum(I.D(xg)); db = sum(I.D(xb));
                pg = I.C' * chosen_g; pb = I.C' * chosen_b;
                wg = I.W' * chosen_g; wb = I.W' * chosen_b;
                rel(o,:) = 100 * [(cg-cb)/cb, (db-dg)/dg, (pb-pg)/pg, ...
                    (wb-wg)/wg, (sum(bill(xb))-sum(bill(xg)))/sum(bill(xg)), ...
                    (sum(downtime(xb))-sum(downtime(xg)))/sum(downtime(xg))];
            end
            av = mean(rel,1,'omitnan');
            g(r)=av(1); rd(r)=av(2); rc(r)=av(3); rw(r)=av(4);
            rb(r)=av(5); rt(r)=av(6);
        end
        gap(a,b) = mean(g,'omitnan'); relD(a,b) = mean(rd,'omitnan');
        relC(a,b) = mean(rc,'omitnan'); relW(a,b) = mean(rw,'omitnan');
        relBill(a,b) = mean(rb,'omitnan'); relDowntime(a,b) = mean(rt,'omitnan');
  end
  fprintf('  alpha_1 = %6.2f $/km   gap %.1f-%.1f%%   distance vs greedy %+.1f%%\n', ...
          alphaDist(a), min(gap(a,:)), max(gap(a,:)), mean(relD(a,:)));
end

fprintf('\n  optimality gap over the whole sweep : %.1f%% to %.1f%%\n', ...
        min(gap,[],'all'), max(gap,[],'all'));
fprintf('  the optimum wins the objective in   : %.0f%% of the settings\n', ...
        100*mean(gap(:) > 0));
fprintf('  it also travels less than greedy in : %.0f%% of the settings\n', ...
        100*mean(relD(:) < 0));
nGreedyRuns = numel(gap) * REPS * ORDERS;
fprintf('  skipped requests: %d out of %d order/settings/requests\n', ...
        nUnserved, nGreedyRuns*N);
a = find(alphaDist == alphaStarNow.dist, 1);
b = find(alphaTime == alphaStarNow.time, 1);
fprintf('  at the reference (%.3f $/km, %.3f $/min), same served cohorts:\n', ...
        alphaDist(a), alphaTime(b));
fprintf('    greedy objective excess %.2f%%; distance %+.2f%%; bill %+.2f%%; downtime %+.2f%%\n', ...
        gap(a,b), relD(a,b), relBill(a,b), relDowntime(a,b));

%% ------------------------------------------------------------- the figure
% Left: the headline result against the only coefficient that could threaten
% it. Right: the per-criterion comparison, where the distance curve crosses
% zero and the other two never do.
f = figure('Position', [80 80 1100 430]);

subplot(1,2,1); hold on; grid on; box on;
plot(alphaDist, mean(gap, 2), '-o', ...
     'LineWidth', 1.8, 'Color', [0.00 0.35 0.75], 'MarkerFaceColor', [0.00 0.35 0.75]);
fill([alphaDist fliplr(alphaDist)], ...
     [min(gap,[],2)' fliplr(max(gap,[],2)')], ...
     [0.00 0.35 0.75], 'FaceAlpha', 0.15, 'EdgeColor', 'none');
set(gca, 'XScale', 'log', 'FontSize', 11);
xlabel('\alpha_1   [$/km]');  ylabel('Greedy excess over the optimum [%]');
title('The advantage of coordination'); ylim([0 max(gap,[],'all')*1.25]);

subplot(1,2,2); hold on; grid on; box on;
plot(alphaDist, relD(:,b), '-o', 'LineWidth', 1.8, ...
     'DisplayName', 'travel distance');
plot(alphaDist, relBill(:,b), '-s', 'LineWidth', 1.8, ...
     'DisplayName', 'energy expenditure');
plot(alphaDist, relDowntime(:,b), '-^', 'LineWidth', 1.8, ...
     'DisplayName', 'wait plus charge');
yline(0, 'k--', 'HandleVisibility', 'off');
set(gca, 'XScale', 'log', 'FontSize', 11);
xlabel('\alpha_1   [$/km]');  ylabel('Optimum relative to greedy [%]');
title(sprintf('Common cohort, time value %.3f $/min',alphaTime(b)));
legend('Location', 'best');

% Uncomment the next line to export this figure as a vector PDF
% exportgraphics(f, 'Figures/sensitivity_alpha.pdf', 'ContentType', 'vector');

if ~isempty(figdir)
    try
        exportgraphics(f, fullfile(figdir, 'sensitivity_alpha.pdf'), 'ContentType', 'vector');
        fprintf('Figure saved in %s\n', figdir);
    catch ME
        warning('Could not save the figure: %s', ME.message);
    end
end

S = struct('alphaDist', alphaDist, 'alphaTime', alphaTime, ...
           'gap', gap, 'relDist', relD, 'relPrice', relC, 'relWait', relW, ...
           'relBill',relBill,'relDowntime',relDowntime, ...
           'M', M, 'N', N, 'REPS', REPS);
end

% =====================================================================
%  local helpers (compact copies of the ones in realworld_analysis)
% =====================================================================

function [evLat, evLon] = local_sample(poly, cLat, cLon, N, sigmaKm)
% Gaussian-biased rejection sampling inside the admissible polygon.
    kLat = 110.574;  kLon = 111.320*cosd(cLat);
    [xl, yl] = boundingbox(poly);
    evLat = zeros(N,1);  evLon = zeros(N,1);  got = 0;
    while got < N
        nTry = 4*(N-got);
        la = yl(1) + (yl(2)-yl(1))*rand(nTry,1);
        lo = xl(1) + (xl(2)-xl(1))*rand(nTry,1);
        keep = isinterior(poly, lo, la);
        d2 = ((la-cLat)*kLat).^2 + ((lo-cLon)*kLon).^2;
        keep = keep & (rand(nTry,1) < exp(-d2/(2*sigmaKm^2)));
        la = la(keep);  lo = lo(keep);
        take = min(numel(la), N-got);
        evLat(got+1:got+take) = la(1:take);
        evLon(got+1:got+take) = lo(1:take);
        got = got + take;
    end
end

function D = local_haversine(pLat, pLon, eLat, eLon)
% Great-circle distances, ports on the rows and vehicles on the columns.
    Re = 6371;
    p1 = deg2rad(pLat(:));          p2 = deg2rad(eLat(:)');
    dphi = p2 - p1;
    dlam = deg2rad(eLon(:)') - deg2rad(pLon(:));
    a = sin(dphi/2).^2 + cos(p1).*cos(p2).*sin(dlam/2).^2;
    D = Re * 2 .* atan2(sqrt(a), sqrt(1-a));
end

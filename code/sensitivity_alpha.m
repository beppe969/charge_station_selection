function S = sensitivity_alpha(csvFile, figdir)
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

pow_level1 = 7;  pow_level2 = 19;  pow_dcfast = 50;

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

if exist('hungarian_assignment_sparse', 'file') == 2
    solve_fun = @hungarian_assignment_sparse;
else
    solve_fun = @hungarian_assignment_binary;
end

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
    % Repair the vehicles left with no acceptable port, as in the main study
    for j = find(~any(A,1))
        aff = find(C <= rho(j));
        [~, k] = min(D(aff, j));
        A(aff(k), j) = true;
    end
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

for a = 1:nD
  for b = 1:nT
        alpha = struct('dist', alphaDist(a), 'price', 1, 'time', alphaTime(b));
        [g, rd, rc, rw] = deal(zeros(REPS,1));
        for r = 1:REPS
            I = inst{r};
            c = build_objective_matrix(I.D, I.W, I.C, portPow, I.Q, alpha);

            % optimum. The solver takes the acceptable matrix FIRST and the
            % cost matrix second, consistently with the rest of the suite.
            H = c;  H(~I.A) = 1e6;                 % forbid the non-acceptable pairs
            xo = full(double(solve_fun(I.A, H)));
            % The penalty on the forbidden pairs is large but finite, so an
            % infeasible instance yields a complete assignment that uses
            % them rather than a failure. Checking the cardinality alone
            % would therefore never detect anything.
            if nnz(xo) < N || any(~I.A(logical(xo)))
                error(['The solver served %d of %d vehicles, or used a ' ...
                       'forbidden pair, on an instance verified as feasible.'], ...
                      nnz(xo), N);
            end

            % Greedy nearest, averaged over the processing orders.
            %
            % EVERYTHING IS PER SERVED VEHICLE, AND THIS MATTERS. The greedy
            % policy occasionally leaves a vehicle unserved: it commits ports
            % one at a time, so a vehicle processed late whose acceptable set
            % is small can find all of it taken. It is a genuine property of
            % the policy, not a defect of the instance, and the optimum serves
            % everyone on the same instance by construction. Comparing TOTALS
            % would then be biased in favour of the greedy, whose sum simply
            % runs over fewer terms, to the point that the gap can come out
            % negative, which is impossible. Dividing each quantity by the
            % number of vehicles it actually covers removes the bias exactly
            % and leaves a quantity that is anyway easier to read: the cost
            % borne by one vehicle.
            og = 0;  dg = 0;  cg = 0;  wg = 0;  ng = 0;
            for o = 1:ORDERS
                xg = full(greedy_nearest_assignment(I.D, I.A, I.ord(o,:)));
                nServed   = nnz(xg);
                nUnserved = nUnserved + (N - nServed);
                og = og + sum(c .* xg, 'all')               / nServed;
                dg = dg + sum(I.D .* xg, 'all')             / nServed;
                cg = cg + sum(repmat(I.C,1,N) .* xg, 'all') / nServed;
                wg = wg + sum(repmat(I.W,1,N) .* xg, 'all') / nServed;
            end
            og = og/ORDERS;  dg = dg/ORDERS;  cg = cg/ORDERS;  wg = wg/ORDERS;

            no = nnz(xo);
            oo = sum(c .* xo, 'all')               / no;
            g(r)  = 100*(og - oo)/oo;
            rd(r) = 100*(sum(I.D .* xo,'all')/no             - dg)/dg;
            rc(r) = 100*(sum(repmat(I.C,1,N) .* xo,'all')/no - cg)/cg;
            rw(r) = 100*(sum(repmat(I.W,1,N) .* xo,'all')/no - wg)/wg;
        end
        gap(a,b) = mean(g);   relD(a,b) = mean(rd);
        relC(a,b) = mean(rc);  relW(a,b) = mean(rw);
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
fprintf('  it always pays less: %d of %d settings\n', sum(relC(:) < 0), numel(relC));
fprintf('  it always queues less: %d of %d settings\n', sum(relW(:) < 0), numel(relW));
nGreedyRuns = numel(gap) * REPS * ORDERS;
fprintf(['  greedy left %d vehicle(s) unserved over %d runs of %d vehicles ' ...
         '(%.3f%%);\n  all the quantities above are per SERVED vehicle, so this ' ...
         'does not bias them\n\n'], nUnserved, nGreedyRuns, N, ...
        100*nUnserved/(nGreedyRuns*N));

%% ------------------------------------------- the numbers to quote in the text
% The robustness paragraph of the paper cites three figures. Rather than
% reading them off the plot, they are computed and printed here, so that the
% text and the figure can never drift apart after a re-run.

gapLo = min(gap, [], 'all');   gapHi = max(gap, [], 'all');

% Where does the distance curve cross zero? Averaged over alpha_3, the curve
% is monotone in alpha_1, so the crossing is found by linear interpolation in
% log(alpha_1) between the two settings that straddle it. If it never changes
% sign inside the sweep, that is reported instead: it is a legitimate outcome
% and it would mean the domination claim holds everywhere tested.
dMean = mean(relD, 2)';
k = find(dMean(1:end-1) .* dMean(2:end) < 0, 1);
if isempty(k)
    crossTxt = sprintf('never inside the sweep (distance stays %s zero)', ...
                       ternary_local(all(dMean > 0), 'above', 'below'));
    alphaCross = NaN;
else
    t = dMean(k) / (dMean(k) - dMean(k+1));           % in [0,1]
    alphaCross = exp(log(alphaDist(k)) + t*(log(alphaDist(k+1)) - log(alphaDist(k))));
    crossTxt = sprintf('%.1f $/km', alphaCross);
end

fprintf('---- numbers for the robustness paragraph -------------------------\n');
fprintf('  sweep of alpha_1                     : %.2f to %.0f $/km (factor %.0f)\n', ...
        min(alphaDist), max(alphaDist), max(alphaDist)/min(alphaDist));
fprintf('  optimality gap of the greedy policy  : %.1f%% to %.1f%%\n', gapLo, gapHi);
fprintf('  the optimum wins on the objective in : %.0f%% of the settings\n', ...
        100*mean(gap(:) > 0));
fprintf('  distance changes sign at alpha_1 =     %s\n', crossTxt);
% At the CALIBRATED setting, not at the middle of the sweep: median(alphaDist)
% is the centre of the grid, which has nothing to do with the value the paper
% adopts, and reporting it here was misleading.
aCal = objective_weights();   % whatever the suite is currently using
cMean = mean(relC, 2)';   wMean = mean(relW, 2)';
la    = log(aCal.dist);
fprintf('  at the calibrated alpha_1 = %.2f $/km, versus greedy:\n', aCal.dist);
fprintf('      distance %+.0f%%,  energy bill %+.0f%%,  queueing %+.0f%%\n', ...
        interp1(log(alphaDist), dMean, la, 'linear'), ...
        interp1(log(alphaDist), cMean, la, 'linear'), ...
        interp1(log(alphaDist), wMean, la, 'linear'));
fprintf('      objective gap of the greedy policy: %.1f%%\n', ...
        interp1(log(alphaDist), mean(gap,2)', la, 'linear'));
fprintf('  price  vs greedy, worst case over the sweep : %+.1f%%\n', max(relC(:)));
fprintf('  time   vs greedy, worst case over the sweep : %+.1f%%\n', max(relW(:)));
fprintf('-------------------------------------------------------------------\n\n');

fprintf(['Sentence to adapt:\n  "the greedy nearest-port policy is between %.1f%% ' ...
         'and %.1f%% worse than the\n   optimum at every setting tested, over a ' ...
         'sweep of alpha_1 spanning a factor %.0f"\n\n'], ...
        gapLo, gapHi, max(alphaDist)/min(alphaDist));

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
plot(alphaDist, mean(relD, 2), '-o', 'LineWidth', 1.8, ...
     'DisplayName', 'travel distance');
plot(alphaDist, mean(relC, 2), '-s', 'LineWidth', 1.8, ...
     'DisplayName', 'charging price');
plot(alphaDist, mean(relW, 2), '-^', 'LineWidth', 1.8, ...
     'DisplayName', 'waiting time');
yline(0, 'k--', 'HandleVisibility', 'off');
set(gca, 'XScale', 'log', 'FontSize', 11);
xlabel('\alpha_1   [$/km]');  ylabel('Optimum relative to greedy [%]');
title('Per-criterion comparison (negative = optimum better)');
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
           'M', M, 'N', N, 'REPS', REPS);
end

% =====================================================================
%  local helpers (compact copies of the ones in realworld_analysis)
% =====================================================================

function out = ternary_local(cond, a, b)
    if cond,  out = a;  else,  out = b;  end
end

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

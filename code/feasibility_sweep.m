function results = feasibility_sweep(csvFile, figdir, cityTag, REPS)
% FEASIBILITY_SWEEP  Where the feasible region ends (Sec. 4.2).
%
%   results = feasibility_sweep(csvFile, figdir, cityTag, REPS)
%
% The case studies fix Rmax = 5 km and repair stranded vehicles, so
% feasibility is never actually tested there. Here it is the output: the
% repair is switched off and the two quantities that build the acceptable
% matrix are swept, the driving range and the willingness to pay.
%
% For every layout we record whether an EV-saturating matching exists
% (Hopcroft-Karp), whether the sufficient condition certifies it, whether
% the necessary prefix condition is satisfied, and, when it fails, which of
% the two resources is binding. That last one is what makes the theory
% useful for planning rather than just for verification.
%
% The budget band must stay above the cheapest tariff. Below it, some
% vehicles can afford nothing at all and the instance is infeasible at any
% driving range, which would measure the sampling convention and not the
% competition for cheap ports.


    if nargin < 2 || isempty(figdir),  figdir  = 'Figures';  end
    if nargin < 4 || isempty(REPS),    REPS    = 25;         end
    if nargin < 3 || isempty(cityTag)
        [~, base] = fileparts(csvFile);
        if contains(lower(base), 'dc'),        cityTag = 'Washington';
        elseif contains(lower(base), 'chic'),  cityTag = 'Chicago';
        elseif contains(lower(base), 'manh') || contains(lower(base), 'ny')
                                               cityTag = 'Manhattan';
        else,                                  cityTag = base;
        end
    end

%% ==================================================================
%  SETTINGS
%  ==================================================================

load_ratio = 0.5;        % N / M, as in the case studies
Qrange     = [20 60];    % energy demand per EV [kWh]
sigmaKm    = 2.5;        % Gaussian spread of the EV density
halfBand   = 0.10;       % rho_j is drawn uniformly on rhoBar +/- halfBand

% THE TWO SWEPT AXES.
% The driving range controls reachability. Below 2 km the network of a
% large city cannot serve the periphery at all; above 5 km reachability
% stops binding in every city we looked at, so the interval below brackets
% the transition.
%
% The willingness to pay controls affordability. The band is rhoBar +/-
% halfBand, and its LOWER END must stay above the cheapest tariff Cmin =
% 0.303 $/kWh. Otherwise the vehicles drawn below Cmin can afford no port
% whatsoever and the instance is infeasible by construction, at any driving
% range: the sweep would then measure the sampling convention rather than
% the competition for cheap ports, which is what we want to observe. With
% halfBand = 0.10 the smallest admissible centre is therefore about 0.41,
% and the grid below starts just above it. The upper end of the grid makes
% essentially the whole network affordable to everyone, so affordability
% stops binding there and the two axes can be compared.
RmaxGrid   = [1.5 2.0 2.5 3.0 3.5 4.0 5.0];     % [km]
rhoBarGrid = [0.42 0.46 0.50 0.55 0.60];        % [$/kWh]

rhoNominal = 0.50;       % the band used in the case studies is 0.40-0.60
Cmin = 0.303;  Cmax = 0.530;                    % [$/kWh]
pow_level1 = 7;  pow_level2 = 19;  pow_dcfast = 50;   % [kW]

rng(7, 'twister');       % so the grid is reproducible

%% ==================================================================
%  PART 0: load the network and expand the stations into ports
%  (identical to realworld_analysis, so the two experiments describe
%   exactly the same infrastructure)
%  ==================================================================

assert(isfile(csvFile), 'Cannot find %s', csvFile);

[~, ~, dataExt] = fileparts(csvFile);
if any(strcmpi(dataExt, {'.csv', '.txt', '.dat'}))
    T = readtable(csvFile, 'Delimiter', ';', 'TextType', 'string');
else
    T = readtable(csvFile, 'TextType', 'string');
end

toNum = @(z) str2double(strrep(string(z), ',', '.'));

statLat = toNum(T.Latitude);
statLon = toNum(T.Longitude);
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

portLat = [];  portLon = [];  portStation = [];
for s = 1:nStations
    counts = [nLevel1(s), nLevel2(s), nDCfast(s)];
    if sum(counts) == 0,  counts = [0 1 0];  end
    for k = 1:3
        for r = 1:counts(k)
            portLat(end+1,1)     = statLat(s);      %#ok<AGROW>
            portLon(end+1,1)     = statLon(s);      %#ok<AGROW>
            portStation(end+1,1) = s;               %#ok<AGROW>
        end
    end
end

M = numel(portLat);
N = round(load_ratio * M);

centLat = mean(statLat, 'omitnan');
centLon = mean(statLon, 'omitnan');
try
    K = boundary(statLon, statLat, 0.6);
catch
    K = convhull(statLon, statLat);
end
hullPoly = simplify(rmholes(polyshape(statLon(K), statLat(K))));

nR   = numel(RmaxGrid);
nRho = numel(rhoBarGrid);

fprintf('\n=========== FEASIBILITY SWEEP: %s ===========\n', upper(cityTag));
fprintf('  ports M            : %d\n', M);
fprintf('  vehicles N         : %d\n', N);
fprintf('  grid               : %d ranges x %d budget bands x %d layouts\n', ...
        nR, nRho, REPS);
fprintf('  stranded EVs are NOT repaired: infeasibility is the observable\n');
fprintf('=================================================\n\n');

%% ==================================================================
%  PART 1: the two-dimensional sweep
%  ==================================================================

% Everything below is a fraction of the REPS layouts of that cell.
pInfeas   = zeros(nR, nRho);   % no EV-saturating matching exists
pSuffCert = zeros(nR, nRho);   % the sufficient condition certifies feasibility
pPrefixOK = zeros(nR, nRho);   % the necessary prefix condition is satisfied
minAccept = zeros(nR, nRho);   % smallest |A^(j)| observed
meanAccept= zeros(nR, nRho);   % average |A^(j)|
pEmptySet = zeros(nR, nRho);   % fraction of EVs with no acceptable port at all
defMean   = zeros(nR, nRho);   % average number of EVs left unmatched
% The singleton case of Hall's condition: every EV must have at least one
% acceptable port. It is trivially necessary. The interesting question,
% which the sweep answers, is how far it is from being sufficient on a real
% network: pNoStranded is how often it holds, and pFeasNoStranded is how
% often the instance is feasible among the layouts where it does.
pNoStranded     = zeros(nR, nRho);
pFeasNoStranded = zeros(nR, nRho);
bindReach = zeros(nR, nRho);   % of the infeasible ones, share where range binds
bindAfford= zeros(nR, nRho);   % ... where affordability binds
bindBoth  = zeros(nR, nRho);   % ... where neither alone would be enough

tStart = tic;
for a = 1:nR
    Rmax = RmaxGrid(a);
    for b = 1:nRho
        rhoBar = rhoBarGrid(b);
        rhoLo  = rhoBar - halfBand;
        rhoHi  = rhoBar + halfBand;

        nInf = 0;  nSuff = 0;  nPre = 0;  accMin = inf;  accSum = 0;
        emptySum = 0;  defSum = 0;  bR = 0;  bA = 0;  bB = 0;
        nNoStr = 0;  nFeasNoStr = 0;

        for r = 1:REPS
            [evLat, evLon] = sample_ev_positions(hullPoly, centLat, centLon, ...
                                                 N, sigmaKm);
            D = haversine_matrix(portLat, portLon, evLat, evLon);   % M x N

            % Station-level tariffs, shared by all the ports of a station
            statC = Cmin + (Cmax - Cmin) * rand(nStations,1);
            C     = statC(portStation);                             % M x 1

            Q     = Qrange(1) + diff(Qrange) * rand(1,N);           % 1 x N
            ratio = rhoLo + (rhoHi - rhoLo) * rand(1,N);            % 1 x N

            Rmat  = D <= Rmax;                                      % M x N
            delta = C <= ratio;                                     % M x N
            A     = Rmat & delta;               % NO repair, on purpose

            % ---- the exact answer
            [~, nMatch] = hopcroft_karp(double(A));
            feasible    = (nMatch == N);
            nInf        = nInf + ~feasible;
            defSum      = defSum + (N - nMatch);

            % ---- the two conditions of Section 2.3
            cardA  = sum(A, 1);                 % 1 x N, the |A^(j)|
            accMin = min(accMin, min(cardA));
            accSum = accSum + mean(cardA);
            emptySum = emptySum + sum(cardA == 0) / N;

            nSuff = nSuff + (min(cardA) >= N);              % Proposition 2.1
            nPre  = nPre  + prefix_condition(A, ratio);     % Theorem 2.1

            % Singleton Hall condition, and how often it already decides
            noStranded = (min(cardA) >= 1);
            nNoStr     = nNoStr + noStranded;
            nFeasNoStr = nFeasNoStr + (noStranded && feasible);

            % ---- when it fails, which of the two resources is binding?
            % We re-test feasibility with one constraint at a time removed.
            % If dropping the budget makes it feasible, money is what binds;
            % if dropping the range does, autonomy is. If neither is enough
            % on its own, the two interact and only a joint relaxation helps.
            if ~feasible
                [~, mR] = hopcroft_karp(double(Rmat));      % ignore price
                [~, mA] = hopcroft_karp(double(delta));     % ignore range
                okR = (mR == N);   okA = (mA == N);
                if      okR && ~okA,  bA = bA + 1;   % affordability binds
                elseif ~okR &&  okA,  bR = bR + 1;   % reachability binds
                elseif  okR &&  okA,  bB = bB + 1;   % only their intersection
                else,                 bB = bB + 1;   % both bind separately
                end
            end
        end

        pInfeas(a,b)    = nInf   / REPS;
        pSuffCert(a,b)  = nSuff  / REPS;
        pPrefixOK(a,b)  = nPre   / REPS;
        minAccept(a,b)  = accMin;
        meanAccept(a,b) = accSum / REPS;
        pEmptySet(a,b)  = emptySum / REPS;
        defMean(a,b)    = defSum / REPS;
        pNoStranded(a,b) = nNoStr / REPS;
        if nNoStr > 0
            pFeasNoStranded(a,b) = nFeasNoStr / nNoStr;
        end
        if nInf > 0
            bindReach(a,b)  = bR / nInf;
            bindAfford(a,b) = bA / nInf;
            bindBoth(a,b)   = bB / nInf;
        end
    end
    fprintf('  Rmax = %.1f km done (%.0f s elapsed)\n', Rmax, toc(tStart));
end

%% ==================================================================
%  PART 2: console summary at the nominal budget band
%  ==================================================================

[~, bNom] = min(abs(rhoBarGrid - rhoNominal));

fprintf('\n--- At the nominal willingness to pay, rho in %.2f-%.2f $/kWh ---\n', ...
        rhoBarGrid(bNom)-halfBand, rhoBarGrid(bNom)+halfBand);
fprintf(['  Rmax   feasible   min|A|   mean|A|   suff.cert   prefix   ' ...
         'no stranded   feas|no stranded\n']);
for a = 1:nR
    fprintf('  %4.1f    %6.1f%%   %6d   %7.0f   %8.1f%%   %6.1f%%   %9.1f%%   %14.1f%%\n', ...
            RmaxGrid(a), 100*(1-pInfeas(a,bNom)), minAccept(a,bNom), ...
            meanAccept(a,bNom), 100*pSuffCert(a,bNom), 100*pPrefixOK(a,bNom), ...
            100*pNoStranded(a,bNom), 100*pFeasNoStranded(a,bNom));
end

% The smallest range at which every layout of the cell was feasible.
idx = find(pInfeas(:,bNom) == 0, 1, 'first');
if isempty(idx)
    fprintf('\n  No range in the grid made every layout feasible.\n');
else
    fprintf(['\n  Smallest range with 100%% feasibility over %d layouts: ' ...
             '%.1f km\n'], REPS, RmaxGrid(idx));
end

% How often each bound is informative, pooled over the whole grid.
totInf   = sum(pInfeas(:)) * REPS;
totFeas  = numel(pInfeas) * REPS - totInf;
certFeas = sum(pSuffCert(:)) * REPS;
missInf  = sum(pPrefixOK(:) .* pInfeas(:)) * REPS;   % infeasible but passing
fprintf('\n--- Tightness of the two bounds, over the whole grid ---\n');
fprintf('  feasible instances certified by the sufficient condition : %.1f%%\n', ...
        100 * certFeas / max(totFeas,1));
fprintf('  infeasible instances detected by the prefix condition    : %.1f%%\n', ...
        100 * (totInf - missInf) / max(totInf,1));
fprintf('  of the infeasible ones: range binds %.0f%%, budget binds %.0f%%, ', ...
        100*mean(bindReach(pInfeas>0)), 100*mean(bindAfford(pInfeas>0)));
fprintf('interaction %.0f%%\n', 100*mean(bindBoth(pInfeas>0)));

%% ==================================================================
%  PART 3: figure
%  ==================================================================

fig = figure('Color','w','Position',[100 100 1000 380]);

subplot(1,2,1);
imagesc(rhoBarGrid, RmaxGrid, 100*(1-pInfeas));
set(gca,'YDir','normal');
colormap(gca, parula);  caxis([0 100]);
cb = colorbar;  cb.Label.String = 'feasible layouts [%]';
xlabel('centre of the willingness to pay band [$/kWh]');
ylabel('driving range R_{max} [km]');
title(sprintf('%s: where the feasible region ends', cityTag));
hold on;
% The frontier: for each budget band, the smallest range with no failure.
frontier = nan(1, nRho);
for b = 1:nRho
    k = find(pInfeas(:,b) == 0, 1, 'first');
    if ~isempty(k),  frontier(b) = RmaxGrid(k);  end
end
plot(rhoBarGrid, frontier, 'o-', 'LineWidth', 1.8, 'Color', [0.85 0.2 0.1], ...
     'MarkerFaceColor', 'w');
legend({'smallest feasible range'}, 'Location','northeast', 'Box','off');
hold off;

subplot(1,2,2);
plot(RmaxGrid, 100*(1-pInfeas(:,bNom)),  'o-', 'LineWidth', 2.0); hold on;
plot(RmaxGrid, 100*pNoStranded(:,bNom),  '^-', 'LineWidth', 1.4);
plot(RmaxGrid, 100*pPrefixOK(:,bNom),    'd:', 'LineWidth', 1.4);
plot(RmaxGrid, 100*pSuffCert(:,bNom),    's--','LineWidth', 1.4);
grid on;  ylim([-3 103]);
xlabel('driving range R_{max} [km]');
ylabel('share of layouts [%]');
title(sprintf('At \\rho \\in [%.2f, %.2f] $/kWh', ...
      rhoBarGrid(bNom)-halfBand, rhoBarGrid(bNom)+halfBand));
legend({'feasible (exact)', 'no stranded EV', 'passing (14)', ...
        'certified by (12)'}, 'Location','southeast', 'Box','off');
hold off;

if ~isfolder(figdir),  mkdir(figdir);  end
outPdf = fullfile(figdir, sprintf('feasibility_sweep_%s.pdf', cityTag));
exportgraphics(fig, outPdf, 'ContentType', 'vector');
fprintf('\n  figure written to %s\n', outPdf);

%% ==================================================================
%  PART 4: paste-ready LaTeX table
%  ==================================================================

fprintf('\n%% ---------- copy from here ----------\n');
fprintf('\\begin{table}[t]\n\\centering\n');
fprintf(['\\caption{Feasibility of the %s network as the driving range ' ...
         'varies, at a willingness to pay uniform on $[%.2f,%.2f]$~\\$/kWh, ' ...
         'over $%d$ random layouts per row. Stranded vehicles are not ' ...
         'repaired, so infeasibility is observed rather than assumed.}\n'], ...
        cityTag, rhoBarGrid(bNom)-halfBand, rhoBarGrid(bNom)+halfBand, REPS);
fprintf('\\label{tab:feasibility_sweep_%s}\n\\small\n', lower(cityTag));
fprintf('\\begin{tabular}{ccccccc}\n\\toprule\n');
fprintf(['$R_{\\max}$ [km] & Feasible [\\%%] & $\\min_j|\\mathcal{A}^{(j)}|$ & ' ...
         '$\\overline{|\\mathcal{A}^{(j)}|}$ & No stranded EV [\\%%] & ' ...
         'Passing \\eqref{eq:feas_condition} [\\%%] & ' ...
         'Certified by \\eqref{eq:budget_sufficient} [\\%%] \\\\\n\\midrule\n']);
for a = 1:nR
    fprintf('$%.1f$ & $%.0f$ & $%d$ & $%.0f$ & $%.0f$ & $%.0f$ & $%.0f$ \\\\\n', ...
            RmaxGrid(a), 100*(1-pInfeas(a,bNom)), minAccept(a,bNom), ...
            meanAccept(a,bNom), 100*pNoStranded(a,bNom), ...
            100*pPrefixOK(a,bNom), 100*pSuffCert(a,bNom));
end
fprintf('\\bottomrule\n\\end{tabular}\n\\end{table}\n');
fprintf('%% ---------- to here ----------\n\n');

%% ==================================================================
%  OUTPUT
%  ==================================================================

results = struct('city', cityTag, 'M', M, 'N', N, 'REPS', REPS, ...
                 'RmaxGrid', RmaxGrid, 'rhoBarGrid', rhoBarGrid, ...
                 'halfBand', halfBand, ...
                 'pInfeasible', pInfeas, 'pSuffCertified', pSuffCert, ...
                 'pPrefixHolds', pPrefixOK, 'minAcceptable', minAccept, ...
                 'meanAcceptable', meanAccept, 'pEmptyAcceptable', pEmptySet, ...
                 'meanDeficiency', defMean, 'bindReach', bindReach, ...
                 'bindAfford', bindAfford, 'bindBoth', bindBoth, ...
                 'pNoStranded', pNoStranded, ...
                 'pFeasibleGivenNoStranded', pFeasNoStranded);

save(sprintf('feasibility_sweep_%s.mat', cityTag), 'results');
fprintf('  results saved to feasibility_sweep_%s.mat\n\n', cityTag);
end


%% ==================================================================
%  LOCAL FUNCTIONS
%  ==================================================================

function ok = prefix_condition(A, ratio)
% PREFIX_CONDITION  The necessary condition of Theorem 2.1.
%
% Sort the vehicles by increasing affordability ratio B_j/Q_j, which here is
% the drawn threshold "ratio", and check that the first j of them have at
% least j distinct acceptable ports between them. The union is accumulated
% incrementally, so the whole check costs one pass over the columns rather
% than N set unions.

    [~, ord] = sort(ratio, 'ascend');
    seen = false(size(A,1), 1);
    ok   = true;
    for j = 1:numel(ord)
        seen = seen | A(:, ord(j));
        if nnz(seen) < j
            ok = false;
            return;
        end
    end
end


function [evLat, evLon] = sample_ev_positions(hullPoly, centLat, centLon, N, sigmaKm)
% SAMPLE_EV_POSITIONS  Identical to the one in realworld_analysis, repeated
% here so that this script can be run on its own.

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

    latP = d2r(pLat(:));    lonP = d2r(pLon(:));
    latV = d2r(vLat(:)');   lonV = d2r(vLon(:)');

    dlat = latP - latV;
    dlon = lonP - lonV;
    a    = sin(dlat/2).^2 + (cos(latP) .* cos(latV)) .* sin(dlon/2).^2;
    D    = Rearth * 2 .* atan2(sqrt(a), sqrt(1-a));
end

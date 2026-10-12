function alpha = objective_weights(profile)
% OBJECTIVE_WEIGHTS  The two free coefficients of the objective.
%
%   alpha = objective_weights()          % calibrated if available
%   alpha = objective_weights(profile)   % 'literature', 'low', 'high'
%
% The cost of one assignment is
%
%   c_ij = alpha.dist  * D_ij                 [$/km]  x [km]
%        + alpha.price * C_i * Q_j            [ - ]   x [$]
%        + alpha.time  * (W_i + 60*Q_j/P_i)   [$/min] x [min]
%
% so c_ij comes out in dollars. Only two of the three are free, because
% multiplying the objective by a positive constant does not move the
% minimiser; fixing alpha_2 = 1 is what puts the criterion in dollars, and
% it says nothing more than that a dollar costs a dollar. Fixing alpha_3
% instead would assert that a minute is worth one dollar, which is a claim
% about the value of time and not a change of units.
%
% Values come from calibrated_alpha.mat when it is on the path, from the
% transport-economics literature otherwise. The function says out loud which
% one it used, once per session, so that a wrong working directory shows up
% immediately instead of silently changing every result.


    persistent announced          % so the message is printed once per session

    if nargin < 1 || isempty(profile);  profile = 'default';  end
    profile = lower(profile);

    % ---- the literature values -------------------------------------------
    % alpha_1: energy at roughly 0.18 kWh/km and 0.40 $/kWh gives about
    % 0.07 $/km; tyres, maintenance and depreciation add roughly 0.10 $/km;
    % the driver's own time, at an urban speed of about 20 km/h, adds three
    % minutes per kilometre. The three together land slightly above 1 $/km.
    % alpha_3: of the order of the value of time recommended by US DOT
    % guidance for local personal travel.
    litDist = 1.00;      % [$/km]
    litTime = 0.35;      % [$/min]

    alpha.price = 1.00;  % [-] the normalisation that puts the objective in $

    switch profile
        case 'literature'
            alpha.dist = litDist;  alpha.time = litTime;
            alpha.source = 'literature';

        case 'low'
            alpha.dist = litDist;  alpha.time = 0.15;
            alpha.source = 'literature, low value of time';

        case 'high'
            alpha.dist = litDist;  alpha.time = 0.70;
            alpha.source = 'literature, high value of time';

        case 'default'
            % Prefer the calibration when it exists. The try/catch keeps a
            % corrupted or outdated .mat from breaking every experiment: the
            % fallback is always a usable pair of coefficients.
            loaded = false;
            if exist('calibrated_alpha.mat', 'file') == 2
                try
                    S = load('calibrated_alpha.mat', 'alphaStar');
                    if isfield(S, 'alphaStar') && ...
                       all(isfield(S.alphaStar, {'dist','time'}))
                        alpha.dist = S.alphaStar.dist;
                        alpha.time = S.alphaStar.time;
                        alpha.source = 'calibrated (calibration_pareto)';
                        loaded = true;
                    end
                catch ME
                    warning('calibrated_alpha.mat could not be read (%s).', ME.message);
                end
            end
            if ~loaded
                alpha.dist = litDist;  alpha.time = litTime;
                alpha.source = 'literature (no calibration found)';
            end

        otherwise
            error(['Unknown profile "%s". Use default, literature, low ' ...
                   'or high.'], profile);
    end

    alpha.profile = profile;

    if isempty(announced)
        fprintf(['Objective coefficients: alpha_1 = %.3f $/km, alpha_3 = ' ...
                 '%.3f $/min  [%s]\n'], alpha.dist, alpha.time, alpha.source);
        announced = true;
    end
end

function [c, terms] = build_objective_matrix(D, W, C, P, Q, alpha)
% BUILD_OBJECTIVE_MATRIX  Assemble the M-by-N edge-cost matrix.
%
%   [c, terms] = build_objective_matrix(D, W, C, P, Q, alpha)
%
%   c_ij = alpha.dist*D_ij + alpha.price*C_i*Q_j + alpha.time*(W_i + 60*Q_j/P_i)
%
% Two details that are easy to get wrong, and that we did get wrong in an
% earlier version.
%
%   The energy term is the EXPENDITURE C_i*Q_j and not the tariff C_i. The
%   budget constraint already compares C_i*Q_j against B_j, so using the
%   rate in the objective and the expenditure in the constraint would put
%   the same quantity in two different units inside one model.
%
%   Q_j/P_i is in HOURS (kWh over kW) while W_i is in minutes. Hence the 60.
%   Without it the charging term is understated sixtyfold and nothing
%   complains.
%
% Neither choice changes the structure: Q and P are data, so c is still a
% constant matrix and the feasibility theory applies verbatim.


    [M, N] = size(D);

    % ---- catch calls written for the old four-weight signature -----------
    % The old form was build_objective_matrix(D, W, C, P, alpha) with alpha a
    % 1x4 vector. Silently reinterpreting it would produce plausible-looking
    % but meaningless numbers, so it is better to stop with an explicit note.
    if nargin == 5 && isnumeric(Q) && numel(Q) == 4
        error(['build_objective_matrix now takes (D, W, C, P, Q, alpha).\n' ...
               'The four weights are now monetary coefficients returned as a ' ...
               'STRUCT by\nobjective_weights(), not a 1x4 vector. Call it as\n\n' ...
               '    alpha = objective_weights();\n' ...
               '    c = build_objective_matrix(D, W, C, P, Qdemand, alpha);\n']);
    end
    if nargin < 6
        error('Six inputs are required: D, W, C, P, Q, alpha.');
    end

    % ---- shape checks: cheap, and they catch broadcasting mistakes -------
    W = W(:);  C = C(:);  P = P(:);
    assert(numel(W) == M, 'W must have M entries');
    assert(numel(C) == M, 'C must have M entries');
    assert(numel(P) == M, 'P must have M entries');
    assert(all(P > 0),    'P must be strictly positive: Q/P would not be defined');
    assert(isstruct(alpha) && all(isfield(alpha, {'dist','time'})), ...
           'alpha must be a struct with fields dist and time');
    if ~isfield(alpha, 'price'),  alpha.price = 1;  end   % unit of the objective

    % Q may be a scalar (all EVs ask for the same energy, as in the case
    % studies) or one value per vehicle. It is normalised to a 1 x N row so
    % that it broadcasts along the columns, which are indexed by the EVs.
    if isscalar(Q)
        Qrow = repmat(Q, 1, N);
    else
        Qrow = reshape(Q, 1, []);
        assert(numel(Qrow) == N, 'Q must be a scalar or have N entries');
    end
    assert(all(Qrow > 0), 'Q must be strictly positive for every vehicle');

    % ---- the four components ---------------------------------------------
    % repmat rather than implicit expansion, for compatibility with MATLAB
    % releases older than R2016b, consistently with the rest of the codebase.
    % Time out of service, in minutes: the queue plus the charge itself.
    Tout    = repmat(W, 1, N) + 60 * (1 ./ P) * Qrow;           % [min] M x N

    travel  = alpha.dist  * D;                                  % alpha_1
    spend   = alpha.price * (C * Qrow);                         % alpha_2
    outserv = alpha.time  * Tout;                               % alpha_3

    c = travel + spend + outserv;

    if nargout > 1
        terms.travel  = travel;                 % distance, monetised
        terms.outserv = outserv;                % time out of service, monetised
        terms.spend   = spend;                  % energy actually paid for
        terms.queue   = alpha.time * repmat(W, 1, N);           % breakdown only
        terms.charge  = alpha.time * 60 * (1 ./ P) * Qrow;      % breakdown only
    end
end

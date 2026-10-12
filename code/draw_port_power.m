function P = draw_port_power(M)
% DRAW_PORT_POWER  Connector powers with a realistic mix.
%
%   P = draw_port_power(M)
%
% Levels are 7, 19 and 50 kW, drawn with the proportions of the installed
% base of the Washington network rather than uniformly. This matters more
% than it looks: a uniform draw puts a third of the ports at 7 kW, where a
% 40 kWh charge takes almost six hours, and the charging-time term then
% dominates everything else in the objective. With the real mix the picture
% is completely different.


    levels  = [7, 19, 50];          % Level 1, Level 2, DC fast   [kW]
    weights = [2, 988, 63];         % installed base, Washington DC

    P = datasample(levels, M, 'Weights', weights)';
end

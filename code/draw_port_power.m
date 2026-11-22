function P = draw_port_power(M)
% Class-proxy power scenarios, weighted by the Washington port inventory.
% Rates are assumed physical inputs; the inventory records class counts.
    levels = [1.4, 7.2, 50];        % US Level 1, Level 2, DC fast [kW]
    weights = [2, 988, 63];
    P = datasample(levels,M,'Weights',weights)';
end

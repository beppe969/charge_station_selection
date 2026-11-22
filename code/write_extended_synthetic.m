function write_extended_synthetic(calibration, study)
% Write the appendix tables from calibration_pareto and structural_analysis.
    out = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'results', 'extended');
    if ~isfolder(out)
        mkdir(out);
    end

    grid = calibration.table;
    writetable(grid, fullfile(out, 'calibration_grid.csv'));
    [~, shortest] = min(grid.meanDist);
    [~, cheapest] = min(grid.meanSpend);
    [~, fastest] = min(grid.meanTime);
    points = grid([shortest; cheapest; fastest; calibration.kneeIdx], :);
    labels = {'Shortest distance', 'Lowest bill', 'Shortest station time', 'Normalized compromise'};
    rows = cell(height(points), 1);
    for i = 1:height(points)
        rows{i} = sprintf('%s & %.3f & %.3f & %.3f & %.2f & %.1f', labels{i}, points{i,:});
    end
    write_rows(fullfile(out, 'calibration_rows.tex'), rows);

    s = study.comparison;
    assert(all(s.valid) && all(s.servAcc(:) == 1), 'The policy table assumes full service.');
    names = {'Batch cost', 'Nearest', 'Cheapest of three'};
    policies = cell(numel(names), 1);
    rows = cell(numel(names), 1);
    for i = 1:numel(names)
        n = size(s.objAcc, 1);
        policies{i} = table((1:n)', repmat(names(i),n,1), s.objAcc(:,i)/s.N, ...
            s.distAcc(:,i), s.costAcc(:,i), s.waitAcc(:,i), s.servAcc(:,i), ...
            'VariableNames', {'instance','policy','cost_per_driver','km','tariff_per_kWh','wait_minutes','served_fraction'});
        rows{i} = sprintf('%s & %.2f & %.2f & %.3f & %.3f & %.2f', names{i}, ...
            s.meanObj(i)/s.N, s.stdObj(i)/s.N, s.meanDist(i), s.meanCost(i), s.meanWait(i));
    end
    writetable(vertcat(policies{:}), fullfile(out, 'synthetic_policies.csv'));
    write_rows(fullfile(out, 'synthetic_policy_rows.tex'), rows);

    ports = study.M_list(:);
    demand = study.N_list(:);
    copies = study.M_rep(:);
    independent = series_table('Independent ports', study.N_base + zeros(size(ports)), ports, study.perEV_M);
    replicated = series_table('Copies of ports', study.N_base + zeros(size(copies)), copies, study.perEV_R);
    fleet = series_table('More requests', demand, study.M_base + zeros(size(demand)), study.perEV_N);
    writetable([independent;replicated;fleet], fullfile(out, 'synthetic_structure.csv'));
    groups = {independent, replicated, fleet};
    row_indices = {find(ismember(ports, [500, 700, 1000])), ...
                   find(ismember(copies, [500, 700, 1000])), find(ismember(demand, [50, 250, 500]))};
    rows = cell(numel(groups), 1);
    for i = 1:numel(groups)
        t = groups{i}(row_indices{i}, :);
        assert(height(t) == 3);
        if i < 3
            varied = t.ports;
        else
            varied = t.requests;
        end
        rows{i} = sprintf('%s & %d & %.2f & %d & %.2f & %d & %.2f', t.design{1}, ...
            varied(1), t.cost_per_driver(1), varied(2), t.cost_per_driver(2), varied(3), t.cost_per_driver(3));
    end
    write_rows(fullfile(out, 'synthetic_structure_rows.tex'), rows);
end

function t = series_table(name, requests, ports, costs)
    names = repmat({name},numel(ports),1);
    t = table(names, requests, ports, costs(:), ...
        'VariableNames', {'design','requests','ports','cost_per_driver'});
end

function write_rows(path, rows)
    fid = fopen(path, 'w');
    assert(fid ~= -1, 'Could not open %s', path);
    cleanup = onCleanup(@() fclose(fid));
    for i = 1:numel(rows)
        fprintf(fid, '%s %s\n', rows{i}, [char(92) char(92)]);
    end
end

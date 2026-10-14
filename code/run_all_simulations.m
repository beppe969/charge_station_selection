function summary = run_all_simulations(RUN_CALIBRATION, figDir)
% RUN_ALL_SIMULATIONS  Reproduce every experiment of the paper, in order.
%
% Luca Ambrosino, Politecnico di Torino.
% "A Matching-Theoretic Approach to Optimal Charging Station Selection for
% Electric Vehicles".
%
%   run_all_simulations                 % skip the calibration, use the shipped weights
%   run_all_simulations(true)           % re-run the calibration from scratch
%   run_all_simulations(false, 'C:\temp\EVfigures')
%
% The whole thing takes a couple of hours. Most of it goes into time_scaling,
% which solves a mixed-integer program at eight sizes, and into the three real
% networks, which solve a hundred assignment problems each. If you only want
% one result, run that script alone: the header of each says which section of
% the paper it produces, and the table in README.tex lists the mapping.
%
% Order matters in exactly one place. calibration_pareto writes
% calibrated_alpha.mat and every later script reads the two coefficients from
% it, so the calibration has to come first if you run it at all. A copy of
% that file ships with the package, which is why the default is to skip it
% and still land on the numbers printed in the paper.
%
% Each script is called inside try/catch, so one failure does not abort the
% rest of the run; the summary at the end says what passed and what did not.

    if nargin < 1 || isempty(RUN_CALIBRATION),  RUN_CALIBRATION = false;  end
    if nargin < 2,                              figDir = '';              end

    %% ------------------------------------------------ where things live
    % Resolve the data folder relative to THIS file rather than to pwd, so
    % the driver works no matter which directory it is launched from.
    here    = fileparts(mfilename('fullpath'));
    dataDir = fullfile(here, '..', 'data');
    if ~isfolder(dataDir)
        error(['Cannot find the data folder at\n    %s\n' ...
               'The package expects code/ and data/ side by side.'], dataDir);
    end

    DC        = fullfile(dataDir, 'CS_location_DC_complete.csv');
    CHICAGO   = fullfile(dataDir, 'CS_location_Chicago.xlsx');
    MANHATTAN = fullfile(dataDir, 'CS_location_Manhattan.xlsx');
    for f = {DC, CHICAGO, MANHATTAN}
        if ~isfile(f{1})
            error('Missing data file:\n    %s', f{1});
        end
    end

    %% ------------------------------------------------ dependency check
    % Everything listed here ships with the package, so a miss means the
    % folder was flattened or only part of it was copied.
    needed = {'build_objective_matrix', 'objective_weights', ...
              'hungarian_assignment_binary', 'hungarian_assignment_sparse', ...
              'munkres', 'hopcroft_karp', 'draw_port_power', ...
              'greedy_nearest_assignment', 'greedy_cheapest_k_nearest', ...
              'greedy_cost_assignment', 'pareto_knee_from_table'};
    missing = needed(cellfun(@(f) exist(f, 'file') ~= 2, needed));
    if ~isempty(missing)
        error(['These files are not on the path: %s.\nThey are part of the ' ...
               'package: add the code/ folder to the MATLAB path and re-run.'], ...
               strjoin(missing, ', '));
    end

    %% ------------------------------------------------ output folder
    % By default ./Figures, but an explicit folder can be passed instead.
    % Useful when the project sits inside OneDrive or Dropbox, which now and
    % then delays or refuses a write.
    if isempty(figDir),  figDir = fullfile(pwd, 'Figures');  end
    if ~isfolder(figDir)
        [made, msg] = mkdir(figDir);
        if ~made
            error(['Cannot create the output folder\n    %s\nReason: %s\n\n' ...
                   'Pass a writable folder explicitly, e.g.\n' ...
                   '    run_all_simulations(false, ''C:\\temp\\EVfigures'')'], ...
                   figDir, msg);
        end
    end
    fprintf('Figures will be written to:\n    %s\n', figDir);

    % Advisory writability probe. Cloud-synced folders give false negatives,
    % so we warn and carry on instead of refusing to run.
    probe = fullfile(figDir, sprintf('wtest_%d.tmp', randi(1e6)));
    fid = fopen(probe, 'w');
    if fid == -1
        warning('runAll:probeFailed', ...
            ['Could not create a test file in\n    %s\n' ...
             'Often a false alarm on OneDrive/Dropbox. The run continues; ' ...
             'if the figures end up missing, rerun with a local folder.'], figDir);
    else
        fclose(fid);  delete(probe);
    end

    %% ------------------------------------------------ build the queue
    % One entry per experiment: the label printed while it runs, the section
    % of the paper it produces, and a no-argument handle that runs it.
    q = struct('name', {}, 'section', {}, 'fun', {});

    if RUN_CALIBRATION
        q = add(q, 'calibration_pareto', 'Sec. 3.2 (Table 1, Fig. 3)', ...
                @() calibration_pareto(figDir));
    else
        if exist('calibrated_alpha.mat', 'file') ~= 2
            warning('runAll:noCalibration', ...
                ['calibrated_alpha.mat is not on the path, so the scripts ' ...
                 'will fall back to the literature weights and the numbers ' ...
                 'will not match the paper. Run with RUN_CALIBRATION = true, ' ...
                 'or put the shipped .mat file on the path.']);
        end
        fprintf('Skipping the calibration; using the shipped calibrated_alpha.mat.\n');
    end

    q = add(q, 'structural_analysis', 'Sec. 3.3.1, 3.3.3 (Figs. 4-5)', ...
            @() structural_analysis(figDir));
    q = add(q, 'synthetic_comparison', 'Sec. 3.3.2 (Table 2)', ...
            @() synthetic_comparison(figDir));
    q = add(q, 'toy_equivalence', 'Sec. 3.4.1 (Fig. 6)', ...
            @() toy_equivalence(figDir));
    q = add(q, 'time_scaling', 'Sec. 3.4.2 (Fig. 7)', ...
            @() time_scaling(figDir));

    q = add(q, 'realworld_analysis (Washington DC)', 'Sec. 4.1 (Tables 3-4, Figs. 8-9)', ...
            @() realworld_analysis(DC, figDir));
    q = add(q, 'realworld_analysis (Chicago)', 'Sec. 4.1 (Table 4, Fig. 10)', ...
            @() realworld_analysis(CHICAGO, figDir));
    q = add(q, 'realworld_analysis (Manhattan)', 'Sec. 4.1 (Table 4, Fig. 11)', ...
            @() realworld_analysis(MANHATTAN, figDir));

    q = add(q, 'feasibility_sweep', 'Sec. 4.2 (Table 5, Fig. 12)', ...
            @() feasibility_sweep(DC, figDir));
    q = add(q, 'load_sweep', 'Sec. 4.3 (Table 6, Fig. 13)', ...
            @() load_sweep(DC, figDir));
    q = add(q, 'sensitivity_alpha', 'Sec. 4.4 (Fig. 14)', ...
            @() sensitivity_alpha(DC, figDir));

    n       = numel(q);
    status  = repmat({'FAILED'}, n, 1);
    elapsed = zeros(n, 1);

    %% ------------------------------------------------ run them
    for k = 1:n
        fprintf('\n===============================================\n');
        fprintf(' %s  (%d/%d)\n', q(k).name, k, n);
        fprintf(' %s\n', q(k).section);
        fprintf('===============================================\n');

        t0 = tic;
        try
            q(k).fun();
            status{k} = 'OK';
        catch ME
            fprintf(2, '\n*** %s FAILED ***\n', q(k).name);
            fprintf(2, 'Message : %s\n', ME.message);
            if ~isempty(ME.identifier)
                fprintf(2, 'ID      : %s\n', ME.identifier);
            end
            for s = 1:numel(ME.stack)
                fprintf(2, '  at %s (line %d)\n', ME.stack(s).name, ME.stack(s).line);
            end
            fprintf(2, '\n');
        end
        elapsed(k) = toc(t0);
        fprintf('--> %s: %s (%.1f s)\n', q(k).name, status{k}, elapsed(k));

        close all;   % the figures are already saved; keep memory bounded
    end

    %% ------------------------------------------------ summary
    fprintf('\n===============================================\n');
    fprintf(' Summary\n');
    fprintf('===============================================\n');
    for k = 1:n
        fprintf('  %-38s %-7s %8.1f s\n', q(k).name, status{k}, elapsed(k));
    end
    fprintf('  %-38s %-7s %8.1f s\n', 'TOTAL', '', sum(elapsed));

    nFailed = sum(strcmp(status, 'FAILED'));
    if nFailed == 0
        fprintf('\nAll done. The figures are in\n    %s\n', figDir);
    else
        fprintf(2, '\n%d of %d experiments failed -- see the messages above.\n', ...
                nFailed, n);
        fprintf(['To debug one, call it directly so the error surfaces with ' ...
                 'its full stack trace,\ne.g.  >> structural_analysis(''Figures'')\n']);
    end

    if nargout > 0
        summary = struct('name', {q.name}, 'status', status', ...
                         'seconds', num2cell(elapsed'));
    end
end


% =========================================================================
function q = add(q, name, section, fun)
    q(end+1) = struct('name', name, 'section', section, 'fun', fun);
end

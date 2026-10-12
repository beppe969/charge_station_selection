function run_all_simulations(RUN_CALIBRATION, figDir)
% RUN_ALL_SIMULATIONS  Reproduce every experiment in the paper, in order.
%
% Luca Ambrosino, Politecnico di Torino.
% "A Matching-Theoretic Approach to Optimal Charging Station Selection for
% Electric Vehicles".
%
% Run this from the code/ folder with the data/ folder next to it. It takes
% a couple of hours end to end, most of it in time_scaling (the MILP) and in
% the three real networks. If you only want to check one result, the header
% of each script says which section it produces.
%
% Order matters in one place only: calibration_pareto writes
% calibrated_alpha.mat, and everything after it reads the coefficients from
% there. A shipped copy of that file is included, so you can skip step 1 and
% still get the numbers in the paper.


    if nargin < 1 || isempty(RUN_CALIBRATION)
        RUN_CALIBRATION = false;
    end
    if nargin < 2
        figDir = '';        % resolved to ./Figures further down
    end

    %% ------------------------------------------------ dependency check
    % Fail fast and explain, instead of dying halfway with a cryptic error.
    if exist('check_dependencies', 'file') == 2
        if ~check_dependencies(true)
            error(['Missing dependencies -- see the report above. ' ...
                   'Fix them and re-run.']);
        end
    else
        warning('check_dependencies.m not found; skipping the pre-flight check.');
    end

    %% ------------------------------------------------ solver setup
    % Wrapped in try/catch so a missing Mosek does not abort the whole run.
    try
        cvx_solver mosek
        cvx_solver_settings('MSK_IPAR_OPTIMIZER', 'MSK_OPTIMIZER_FREE_SIMPLEX')
    catch ME
        warning('Could not select Mosek (%s). Falling back to the CVX default.', ...
                ME.message);
    end

    %% ------------------------------------------------ output folder
    % Where the figures go. By default ./Figures, but an explicit folder
    % can be passed as the second argument -- handy when the project sits
    % inside OneDrive/Dropbox, which sometimes refuses or delays writes.
    if isempty(figDir)
        figDir = fullfile(pwd, 'Figures');
    end
    if ~exist(figDir, 'dir')
        [made, msg] = mkdir(figDir);
        if ~made
            error(['Cannot create the output folder\n    %s\nReason: %s\n\n' ...
                   'Pass a writable folder explicitly, e.g.\n' ...
                   '    run_all_simulations(false, ''C:\\temp\\EVfigures'')'], ...
                   figDir, msg);
        end
    end

    % Publish the choice so that save_figure picks it up. appdata on the
    % root handle survives "clear", so it holds for the whole run.
    setappdata(0, 'EVPAPER_FIGDIR', figDir);
    fprintf('Figures will be written to:\n    %s\n', figDir);

    % A quick, NON-FATAL writability check. It is only advisory: probes
    % can give false negatives on cloud-synced folders, so we warn and
    % carry on rather than refusing to run.
    probe = fullfile(figDir, sprintf('wtest_%d.tmp', randi(1e6)));
    fid = fopen(probe, 'w');
    if fid == -1
        warning('runAll:probeFailed', ...
            ['Could not create a test file in\n    %s\n' ...
             'This is often a false alarm on OneDrive/Dropbox folders, so ' ...
             'the run will continue.\nIf the figures do end up missing, ' ...
             'rerun with a local folder:\n' ...
             '    run_all_simulations(false, ''C:\\temp\\EVfigures'')'], figDir);
    else
        fclose(fid);  delete(probe);
    end

    %% ------------------------------------------------ build the queue
    scripts = {};
    if RUN_CALIBRATION
        % sim_calibration_alpha is obsolete: the objective is monetized, so
        % there are no abstract weights left to calibrate on a grid. What
        % replaces it is a sensitivity study on a real network.
        % scripts{end+1} = 'sim_calibration_alpha';
        scripts{end+1} = 'sensitivity_alpha';          % Section 3.2
    end
    scripts{end+1} = 'sim_structural_properties';     % Section 3.3
    scripts{end+1} = 'sim_toy_equivalence';           % Section 3.4
    scripts{end+1} = 'sim_time_scaling';              % Section 3.5
    scripts{end+1} = 'sim_realworld_greedy';          % Section 3.6
    scripts{end+1} = 'sim_fairness_boxplot';          % Section 3.7

    nScripts = numel(scripts);
    status   = cell(nScripts, 1);
    elapsed  = zeros(nScripts, 1);

    %% ------------------------------------------------ run them
    for k = 1:nScripts
        name = scripts{k};
        fprintf('\n===============================================\n');
        fprintf(' Running %s  (%d/%d)\n', name, k, nScripts);
        fprintf('===============================================\n');

        t0 = tic;
        try
            run_one(name);
            status{k} = 'OK';
        catch ME
            status{k} = 'FAILED';
            % Report the REAL error, with the exact file and line, instead
            % of swallowing it. This is what was missing before.
            fprintf(2, '\n*** %s FAILED ***\n', name);
            fprintf(2, 'Message : %s\n', ME.message);
            if ~isempty(ME.identifier)
                fprintf(2, 'ID      : %s\n', ME.identifier);
            end
            for s = 1:numel(ME.stack)
                fprintf(2, '  at %s (line %d)\n', ...
                        ME.stack(s).name, ME.stack(s).line);
            end
            fprintf(2, '\n');
        end
        elapsed(k) = toc(t0);
        fprintf('--> %s: %s (%.1f s)\n', name, status{k}, elapsed(k));

        close all;   % figures are already saved; keep memory bounded
    end

    %% ------------------------------------------------ summary
    fprintf('\n===============================================\n');
    fprintf(' Summary\n');
    fprintf('===============================================\n');
    for k = 1:nScripts
        fprintf('  %-28s %-7s %8.1f s\n', scripts{k}, status{k}, elapsed(k));
    end

    nFailed = sum(strcmp(status, 'FAILED'));
    if nFailed == 0
        fprintf('\nAll done. Figures are in ./Figures (vector PDF).\n');
    else
        fprintf(2, '\n%d of %d scripts failed -- see the messages above.\n', ...
                nFailed, nScripts);
        fprintf(['To debug a single one, run it directly at the command line,\n' ...
                 'e.g.  >> sim_structural_properties\n' ...
                 'so that the error surfaces with its full stack trace.\n']);
    end
end


% =========================================================================
function run_one(scriptName)
% Execute a simulation script inside an isolated function workspace.
%
% Anything the script creates -- or clears -- lives here and cannot touch
% the caller's loop variables. evalin/run semantics make the script body
% execute in THIS function's scope, which is exactly the isolation we want.
    run(scriptName);
end

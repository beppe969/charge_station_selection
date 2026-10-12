function [ParetoSet, KneeSolution, isDominated, paretoIdx, kneeIdx] = ...
    pareto_knee_from_table(T, objNames, makePlot)
% PARETO_KNEE_FROM_TABLE
% Extract the Pareto set from a MATLAB table and select a representative
% knee-point solution. Optionally, generate a scatter plot showing dominated
% and non-dominated configurations.
%
% INPUTS:
%   T         - MATLAB table containing one row per configuration.
%
%   objNames  - Cell array of objective column names to be minimized, e.g.
%               {'meanTime','meanCost','meanDist','meanWait'}
%
%   makePlot  - Optional logical flag. If true, generate a scatter plot in
%               the (cost, distance) plane, colored by waiting time, and
%               highlight the Pareto set and the knee point.
%               Default: true.
%
% OUTPUTS:
%   ParetoSet    - Subtable of non-dominated configurations.
%   KneeSolution - One-row table corresponding to the selected knee point.
%   isDominated  - Logical vector over the rows of T.
%   paretoIdx    - Indices of Pareto-optimal rows in T.
%   kneeIdx      - Index of the selected knee point in T.
%
% NOTES:
%   1) All objectives in objNames are assumed to be minimized.
%   2) The knee point is selected as the Pareto solution with minimum
%      Euclidean distance from the normalized ideal point.
%   3) The plot assumes that T contains the variables meanCost, meanDist,
%      and meanWait. These are used only for visualization, not for the
%      Pareto computation unless explicitly included in objNames.
%
% EXAMPLE:
%   [ParetoSet, KneeSolution] = pareto_knee_from_table( ...
%       T, {'meanTime','meanCost','meanDist','meanWait'}, true);

    if nargin < 3 || isempty(makePlot)
        makePlot = true;
    end

    %--------------------------------------------------------------
    % Step 1: Build objective matrix
    %--------------------------------------------------------------
    nObj = numel(objNames);
    X = zeros(height(T), nObj);

    for k = 1:nObj
        if ~ismember(objNames{k}, T.Properties.VariableNames)
            error('Column "%s" not found in table T.', objNames{k});
        end
        X(:,k) = T.(objNames{k});
    end

    %--------------------------------------------------------------
    % Step 2: Compute dominated / non-dominated points
    %--------------------------------------------------------------
    n = size(X,1);
    isDominated = false(n,1);

    for i = 1:n
        for j = 1:n
            if j ~= i
                % j dominates i if:
                %   X(j,:) <= X(i,:) for all objectives, and
                %   X(j,:) <  X(i,:) for at least one objective
                if all(X(j,:) <= X(i,:)) && any(X(j,:) < X(i,:))
                    isDominated(i) = true;
                    break;
                end
            end
        end
    end

    paretoIdx = find(~isDominated);
    ParetoSet = T(paretoIdx, :);

    %--------------------------------------------------------------
    % Step 3: Select the knee point from the Pareto set
    %--------------------------------------------------------------
    XP = zeros(height(ParetoSet), nObj);
    for k = 1:nObj
        XP(:,k) = ParetoSet.(objNames{k});
    end

    % Min-max normalization over Pareto points
    xmin = min(XP, [], 1);
    xmax = max(XP, [], 1);
    Xn = (XP - xmin) ./ (xmax - xmin + eps);

    % Distance from the ideal point [0, 0, ..., 0]
    distIdeal = sqrt(sum(Xn.^2, 2));

    % Knee point = Pareto point closest to the normalized ideal point
    [~, localKnee] = min(distIdeal);

    KneeSolution = ParetoSet(localKnee, :);
    kneeIdx = paretoIdx(localKnee);

    %--------------------------------------------------------------
    % Step 4: Visualization (optional)
    %--------------------------------------------------------------
    if makePlot
        requiredVars = {'meanCost','meanDist','meanWait'};
        for k = 1:numel(requiredVars)
            if ~ismember(requiredVars{k}, T.Properties.VariableNames)
                error('Plot requested, but column "%s" is missing from T.', requiredVars{k});
            end
        end

        figure;
        hold on;
        grid on;
        box on;

        % Dominated configurations
        scatter(T.meanCost(isDominated), ...
                T.meanDist(isDominated), ...
                35, T.meanWait(isDominated), ...
                'filled', ...
                'MarkerFaceAlpha', 0.30, ...
                'MarkerEdgeAlpha', 0.30);

        % Pareto-optimal configurations
        scatter(T.meanCost(~isDominated), ...
                T.meanDist(~isDominated), ...
                80, T.meanWait(~isDominated), ...
                'o', ...
                'filled', ...
                'MarkerEdgeColor', 'k', ...
                'LineWidth', 0.8);

        % Selected knee point
        scatter(T.meanCost(kneeIdx), ...
                T.meanDist(kneeIdx), ...
                220, T.meanWait(kneeIdx), ...
                'p', ...
                'filled', ...
                'MarkerEdgeColor', 'k', ...
                'LineWidth', 1.5);

        xlabel('Average Cost [$/kWh]');
        ylabel('Average Travel Distance [min]');
        title('Pareto set in the Cost-Distance plane, colored by Waiting Time');

        cb = colorbar;
        cb.Label.String = 'Average Waiting Time [min]';
        colormap turbo;

        legend({'Dominated configurations', ...
                'Pareto-optimal configurations', ...
                'Selected knee point'}, ...
                'Location', 'best');

        hold off;

        % Save the figure as PDF
        % print(gcf, 'Pareto.pdf', '-dpdf', '-bestfit')
    end
end
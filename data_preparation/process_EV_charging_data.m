function filtered_data = process_EV_charging_data(filename, selected_state)
%PROCESS_EV_CHARGING_DATA Reads a CSV and processes EV station data for a given state.
%   INPUTS:
%       filename - string, name of the CSV file to read (e.g., 'USA_EV_CS.csv')
%       selected_state - string, 2-letter abbreviation of the state to filter (e.g., 'NY')
%   OUTPUT:
%       filtered_data - filtered table with selected columns and cleaned EVSE values

    % Read the CSV file and import data
    data = readtable(filename);

    % --- PIE CHART OF STATES ---
    %{
    % Count the number of occurrences for each unique state
    [stateNames, ~, idx] = unique(data.State);
    stateCounts = accumarray(idx, 1);

    % Create full labels with state name and count
    fullLabels = strcat(stateNames, " (", string(stateCounts), ")");
    
    % Show labels only for states with more than 1000 entries
    showLabels = stateCounts > 1000;
    displayLabels = repmat("", size(fullLabels));
    displayLabels(showLabels) = fullLabels(showLabels);

    % Choose a custom colormap
    colors = parula(numel(stateCounts));

    % Create the pie chart
    figure;
    pie(stateCounts, displayLabels);
    colormap(colors)
    title('Distribution of Charging Stations by State', 'FontSize', 14);
    set(findall(gcf,'Type','text'), 'FontSize', 11, 'FontWeight', 'bold');
    axis equal;
    %}

    % --- FILTER DATA FOR SELECTED STATE + PUBLIC ACCESS ---
    filtered_data = data(strcmp(data.State, selected_state) & strcmp(data.AccessCode, 'public'), :);

    % Keep only selected columns
    filtered_data = filtered_data(:, {
        'StreetAddress', ...
        'City', ...
        'State', ...
        'ZIP', ...
        'EVLevel1EVSENum', ...
        'EVLevel2EVSENum', ...
        'EVDCFastCount'
    });

    % Display number of missing values for each column
    disp("Missing value count per column:");
    summary(filtered_data)

    % Replace NaN with 0 in EV charger columns
    filtered_data.EVLevel1EVSENum(isnan(filtered_data.EVLevel1EVSENum)) = 0;
    filtered_data.EVLevel2EVSENum(isnan(filtered_data.EVLevel2EVSENum)) = 0;
    filtered_data.EVDCFastCount(isnan(filtered_data.EVDCFastCount)) = 0;

    % Check for invalid values
    cols = {'EVLevel1EVSENum', 'EVLevel2EVSENum', 'EVDCFastCount'};
    for i = 1:length(cols)
        col = filtered_data.(cols{i});
        if any(col < 0)
            warning("Negative values found in column %s", cols{i});
        end
        if any(mod(col, 1) ~= 0)
            warning("Non-integer values found in column %s", cols{i});
        end
    end

    % Compute basic statistics
    stats = table( ...
        mean(filtered_data.EVLevel1EVSENum), ...
        max(filtered_data.EVLevel1EVSENum), ...
        min(filtered_data.EVLevel1EVSENum), ...
        mean(filtered_data.EVLevel2EVSENum), ...
        max(filtered_data.EVLevel2EVSENum), ...
        min(filtered_data.EVLevel2EVSENum), ...
        mean(filtered_data.EVDCFastCount), ...
        max(filtered_data.EVDCFastCount), ...
        min(filtered_data.EVDCFastCount), ...
        'VariableNames', { ...
            'Mean_L1', 'Max_L1', 'Min_L1', ...
            'Mean_L2', 'Max_L2', 'Min_L2', ...
            'Mean_DC', 'Max_DC', 'Min_DC' ...
        });
    
    disp("Statistics of EV charger counts:");
    disp(stats);

    % Remove rows where all charger counts are zero
    total_chargers = filtered_data.EVLevel1EVSENum + ...
                     filtered_data.EVLevel2EVSENum + ...
                     filtered_data.EVDCFastCount;
    filtered_data(total_chargers == 0, :) = [];

end

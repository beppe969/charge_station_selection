clc
clear all
close all

%%% Code to translate addresses in coordinates

%% Selection: state and city
stateCode = 'NY';              % two-letter state code
cityName  = "Manhattan";        % city to keep; use "" to keep the whole state

% cityName may also be a list, which is what New York needs, since the
% dataset stores the boroughs as separate cities:
% cityName = ["New York","Brooklyn","Queens","Bronx","Staten Island", ...
%             "Manhattan","Long Island City","Flushing","Astoria","Jamaica"];

%% Use the function "process_EV_charging_data" to clean dataset
filtered_data = process_EV_charging_data('USA_EV_CS.csv', stateCode);

% Keep only the requested city (or cities). The comparison is
% case-insensitive and ignores stray blanks, because the City field of the
% national dataset is not perfectly normalised.
if strlength(join(cityName, "")) > 0
    cityCol = strtrim(string(filtered_data.City));
    keep    = ismember(lower(cityCol), lower(cityName));
    filtered_data = filtered_data(keep, :);

    assert(~isempty(filtered_data), ...
        'No station found for "%s" in state %s. Check the spelling.', ...
        join(cityName, ", "), stateCode);
    fprintf('%s, %s: %d charging stations selected.\n', ...
            join(cityName, ", "), stateCode, height(filtered_data));
else
    fprintf('%s: %d charging stations selected (whole state).\n', ...
            stateCode, height(filtered_data));
end

% NOTE: the line "filtered_data = filtered_data(1:3, :);" that used to be
% here truncated the table to three rows. It was a test and must stay out,
% otherwise only three addresses get geocoded.

%% Convert address to coordinates using OpenCage

apiKey = '5aadd82555394db792545f3291fc69f5';  % Replace with your OpenCage API key
n = height(filtered_data);  % Number of addresses
latitudes = NaN(n,1); % Initialize latitude
longitudes = NaN(n,1); % Initialize longitude

% Create waitbar
h = waitbar(0, 'Geocoding in progress...');

for i = 1:n
    % Construct full address
    address = sprintf('%s, %s, %s %s', ...
        filtered_data.StreetAddress{i}, ...
        filtered_data.City{i}, ...
        filtered_data.State{i}, ...
        string(filtered_data.ZIP(i)));

    % Encode the address for URL
    addressEncoded = urlencode(address);

    % Send request to OpenCage API
    url = ['https://api.opencagedata.com/geocode/v1/json?q=' ...
           addressEncoded '&key=' apiKey '&limit=1'];
    try
        response = webread(url);
        if ~isempty(response.results)
            latitudes(i) = response.results(1).geometry.lat;
            longitudes(i) = response.results(1).geometry.lng;
        end
    catch ME
        warning('Error retrieving geocode for index %d: %s', i, ME.message);
    end

    % Update waitbar
    waitbar(i/n, h, sprintf('Processing %d of %d addresses...', i, n));

    pause(1);  % Pause to avoid hitting API rate limit
end

% Close waitbar
close(h);

% Add coordinates to table
filtered_data.Latitude = latitudes;
filtered_data.Longitude = longitudes;

%% Save the dataset with coordinates

% Copy the filtered table to preserve the original data
export_table = filtered_data;

% Identify numeric variables in the table
numericVars = varfun(@isnumeric, export_table, 'OutputFormat', 'uniform');

% Loop through all numeric columns to convert them to string with comma as decimal separator
for i = find(numericVars)
    % Convert numbers to strings and replace dot with comma for decimal separation (Excel EU format)
    export_table.(i) = strrep(cellstr(num2str(export_table.(i), '%.6f')), '.', ',');
end

% Build a filename that identifies the selection: the city when one was
% requested, the state otherwise. Blanks are removed so that the name is
% safe to use ('New York' -> 'NewYork').
if strlength(join(cityName, "")) > 0
    tag = char(regexprep(cityName(1), '\s+', ''));
    if numel(cityName) > 1
        tag = [tag 'Area'];    % several cities merged, e.g. the NYC boroughs
    end
else
    tag = upper(stateCode);
end
filename = ['CS_location_' tag '.csv'];

% Write the table to a CSV file using semicolon delimiter and text format
writetable(export_table, filename, 'Delimiter', ';', 'FileType', 'text');


%% Check that the coordinates have reasonable numbers and filter to be sure
% The last number is the percentile: if we use 5, we'll keep data between
% the 5% and 95%, sorting both coordinates.
filtered_output = plot_and_filter_coordinates(filtered_data, stateCode, 5);

%% Compute the total number of charging ports in each station

% List of columns you want to convert
colsToFix = {'EVLevel1EVSENum', 'EVLevel2EVSENum', 'EVDCFastCount'};

for i = 1:length(colsToFix)
    colName = colsToFix{i};

    % Step 1: Replace ',' with '.' to use the correct decimal separator
    cleanedStrings = strrep(CS_location_DC.(colName), ',', '.');

    % Step 2: Convert to double
    numbers = str2double(cleanedStrings);

    % Step 3: Round and convert to integer
    CS_location_DC.(colName) = int32(round(numbers));
end

% Compute the total number of charging ports per each CS
col8 = CS_location_DC{:, 5} + CS_location_DC{:, 6} + CS_location_DC{:, 7};

% New column
CS_location_DC = [CS_location_DC(:,1:7) table(col8, 'VariableNames', {'NumPorts'}) CS_location_DC(:,8:end)];


%% Simulate EVs and plot the map

% Read the CSV file and import data
CS_location_DC = readtable('CS_location_DC_complete.csv');

toNum = @(z) str2double(strrep(string(z), ',', '.'));  % handles char/cellstr/string

% centroid (compute before calling other functions)
centroidLat = mean(toNum(CS_location_DC.Latitude), 'omitnan');
centroidLon = mean(toNum(CS_location_DC.Longitude), 'omitnan');

N = 10; % Number of EVs N
% Use the function to generate N random points (EVs location)
% and to compute their distances within the charging ports
% [distMatrix, points] = computeDistancesFromRandomPoints(CS_location_DC, N);
% NOTE THAT for now, the function does not consider multiple ports in the
% same location. So each charging station in the dataset is considered as a
% single chargign port.
[distMatrix, points] = computeDistancesFromRandomPointsBiased( ...
    CS_location_DC, N, centroidLat, centroidLon, ...
    'SigmaKm', 2.0, ...        % smaller => tighter around centroid
    'Shrink', 0.6, ...         % 0<shrink<=1, smaller => tighter concave hull (1 = convex hull)
    'Plot', true);              % set false to skip plotting


%% Plot with the real GeoMap in the background

figure % new figure
gx = geoaxes;                      % create a geographic axes
geobasemap(gx,'streets');          % try 'streets-dark','satellite','topographic', ...
hold(gx,'on')

% stationLat, stationLon : column vectors (charging stations, blue)
% vehLat,     vehLon     : column vectors (vehicles, red)
% save latitude and longitude
stationLat = toNum(CS_location_DC.Latitude(:));
stationLon = toNum(CS_location_DC.Longitude(:));
vehLat = toNum(points(:,1));
vehLon = toNum(points(:,2));
ok = ~isnan(vehLat) & ~isnan(vehLon);

% nice view limits (tight box around all points with a small margin)
latAll = [stationLat; vehLat];
lonAll = [stationLon; vehLon];
pad = 0.005;                       % ~0.5 km; tweak as you like
geolimits(gx, [min(latAll)-pad, max(latAll)+pad], [min(lonAll)-pad, max(lonAll)+pad])


% plot Charging Ports
geoscatter(gx, stationLat, stationLon, 15, 'b', 'filled', ...
    'MarkerFaceAlpha', 0.7, 'MarkerEdgeAlpha', 0.7);

% add CAR ICONS ON A GEOAXES
stampCarIcons(gx, vehLat, vehLon, 'car_red.png', 'WidthKm',0.5, 'ScaleX',0.5, 'ScaleY',0.6);
% if you want instead simple red dots for vehicles, uncomment:
% geoscatter(gx, vehLat(ok), vehLon(ok), 15, 'r', 'filled');
% if you want instead simple red cross for vehicles, uncomment:
% geoscatter(gx, vehLat(ok), vehLon(ok), 15, 'r', 'X', 'filled', ...
%    'MarkerFaceAlpha',0.9,'MarkerEdgeColor','r');

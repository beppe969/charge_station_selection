function T = Coordinates_Finder(stateCode, cityNames, outputFile)
% Geocode public charging-station addresses with the OpenCage API.
% Coordinates_Finder('IL',"Chicago",'Chicago-geocoded.csv')
% Coordinates_Finder('NY',"New York",'Manhattan-geocoded.csv')
% Give an empty city string to select all public records for the state.
    if nargin < 1, stateCode = 'NY'; end
    if nargin < 2
        if strcmpi(strtrim(stateCode),'NY')
            cityNames = "New York";  % source city label used by the bundled Manhattan file
        else
            cityNames = "";
        end
    end
    cityNames = strtrim(string(cityNames));
    cityNames = cityNames(strlength(cityNames) > 0);
    if nargin < 3
        if isempty(cityNames), label = string(stateCode); else, label = join(cityNames,"-"); end
        outputFile = char("CS_location_" + regexprep(label,'\s+','') + "_geocoded.csv");
    end
    if isfile(outputFile)
        error('geocoding:outputExists','Choose a new output file. %s already exists.',outputFile);
    end
    apiKey = getenv('OPENCAGE_API_KEY');
    if isempty(apiKey)
        error('geocoding:missingKey','Set OPENCAGE_API_KEY before requesting fresh geocodes.');
    end
    packageDir = fileparts(fileparts(mfilename('fullpath')));
    T = process_EV_charging_data(fullfile(packageDir,'data','USA_EV_CS.csv'),upper(strtrim(stateCode)));
    if ~isempty(cityNames)
        T = T(ismember(lower(strtrim(string(T.City))),lower(cityNames)),:);
    end
    if isempty(T)
        error('geocoding:noStations','No public stations with positive port counts match the selection.');
    end
    T.NumPorts = T.EVLevel1EVSENum + T.EVLevel2EVSENum + T.EVDCFastCount;
    T.Latitude = nan(height(T),1);
    T.Longitude = nan(height(T),1);
    endpoint = 'https://api.opencagedata.com/geocode/v1/json';
    for i = 1:height(T)
        address = join([string(T.StreetAddress(i)), string(T.City(i)), ...
                        string(T.State(i)), string(T.ZIP(i))],", ");
        try
            response = webread(endpoint,'q',char(address),'key',apiKey,'limit','1');
        catch ME
            % Network exceptions may include the request URL and access key.
            error('geocoding:requestFailed','OpenCage request failed at row %d (%s).',i,ME.identifier);
        end
        if ~isempty(response.results)
            T.Latitude(i) = response.results(1).geometry.lat;
            T.Longitude(i) = response.results(1).geometry.lng;
        end
        pause(1);
    end
    writetable(T,outputFile,'Delimiter',';');
    fprintf('Geocoded %d of %d selected rows. Wrote %s\n',nnz(isfinite(T.Latitude)),height(T),outputFile);
end

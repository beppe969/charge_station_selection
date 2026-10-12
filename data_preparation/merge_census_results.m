function T = merge_census_results(xlsxFile, censusResultFile)
% MERGE_CENSUS_RESULTS  Fill Latitude/Longitude from a Census Geocoder result.
%
% Code edited by Luca Ambrosino.
% Optimal Charging Station Selection for EVs.
%
%   T = MERGE_CENSUS_RESULTS(xlsxFile, censusResultFile)
%
% Takes the file returned by the US Census Bureau batch geocoder and writes
% its coordinates into the corresponding city spreadsheet, producing a file
% with exactly the format of CS_location_DC_complete.
%
% WORKFLOW
%   1. Upload census_batch_<City>.csv at
%        https://geocoding.geo.census.gov/geocoder/locations/addressbatch
%      choosing benchmark "Public_AR_Current". The service is free and needs
%      no API key.
%   2. Download the result, typically named GeocodeResults.csv.
%   3. Run, for example
%        merge_census_results('CS_location_Chicago.xlsx', 'GeocodeResults.csv')
%
% THE COORDINATE ORDER IS THE TRAP. The Census result reports the matched
% point as "longitude,latitude", in that order, which is the opposite of
% the convention used almost everywhere else. Swapping them silently puts
% Chicago in the Indian Ocean, so this function splits the field explicitly
% and assigns longitude first.
%
% The result file has NO header and its columns are:
%   1 unique ID        2 input address     3 match indicator
%   4 match type       5 matched address   6 "lon,lat"
%   7 TIGER line ID    8 side of the street
% Rows whose match indicator is not "Match" are left as NaN rather than
% guessed, and are listed at the end of the run.
%
% INPUTS
%   xlsxFile          the city spreadsheet to fill
%   censusResultFile  the CSV returned by the Census geocoder
%
% OUTPUT
%   T                 the completed table, also written back to xlsxFile

    assert(isfile(xlsxFile),         'Cannot find %s', xlsxFile);
    assert(isfile(censusResultFile), 'Cannot find %s', censusResultFile);

    T = readtable(xlsxFile);
    nRows = height(T);

    if ~ismember('Latitude',  T.Properties.VariableNames);  T.Latitude  = nan(nRows,1);  end
    if ~ismember('Longitude', T.Properties.VariableNames);  T.Longitude = nan(nRows,1);  end
    T.Latitude  = nan(nRows,1);
    T.Longitude = nan(nRows,1);

    %% ------------------------------------------------- read the result
    % The file has no header and a variable number of populated columns,
    % so it is read as plain text and split by hand.
    opts = detectImportOptions(censusResultFile, 'FileType', 'text', ...
                               'Delimiter', ',', 'ReadVariableNames', false);
    opts = setvartype(opts, 'char');
    R = readtable(censusResultFile, opts);

    nMatched = 0;
    unmatched = {};

    for k = 1:height(R)
        idStr = strtrim(R{k,1}{1});
        id = str2double(idStr);
        if ~isfinite(id) || id < 1 || id > nRows
            continue;                       % malformed line, skip it
        end

        status = '';
        if width(R) >= 3;  status = strtrim(R{k,3}{1});  end
        if ~strcmpi(status, 'Match')
            unmatched{end+1} = sprintf('row %4d: %s', id, strtrim(R{k,2}{1})); %#ok<AGROW>
            continue;
        end

        coordStr = '';
        if width(R) >= 6;  coordStr = strtrim(R{k,6}{1});  end
        parts = strsplit(coordStr, ',');
        if numel(parts) ~= 2
            unmatched{end+1} = sprintf('row %4d: empty coordinates', id); %#ok<AGROW>
            continue;
        end

        % Census order is LONGITUDE first, then LATITUDE
        lon = str2double(parts{1});
        lat = str2double(parts{2});
        if ~isfinite(lat) || ~isfinite(lon)
            unmatched{end+1} = sprintf('row %4d: unparsable coordinates', id); %#ok<AGROW>
            continue;
        end

        T.Latitude(id)  = lat;
        T.Longitude(id) = lon;
        nMatched = nMatched + 1;
    end

    %% ------------------------------------------------- sanity check
    % A coarse plausibility test on the continental United States. It costs
    % nothing and catches the classic lon/lat inversion immediately.
    ok = isfinite(T.Latitude);
    if any(ok)
        badBox = ok & ( T.Latitude  < 24 | T.Latitude  > 50 | ...
                        T.Longitude < -125 | T.Longitude > -66 );
        if any(badBox)
            warning(['%d rows fall outside the continental US bounding box. ' ...
                     'If ALL of them do, latitude and longitude have been ' ...
                     'swapped somewhere.'], nnz(badBox));
        end
    end

    writetable(T, xlsxFile);

    fprintf('\nMatched %d of %d rows (%.1f%%). File written: %s\n', ...
            nMatched, nRows, 100*nMatched/nRows, xlsxFile);
    if ~isempty(unmatched)
        fprintf('%d addresses were not matched:\n', numel(unmatched));
        for k = 1:min(10, numel(unmatched))
            fprintf('   %s\n', unmatched{k});
        end
        if numel(unmatched) > 10
            fprintf('   ... and %d more.\n', numel(unmatched)-10);
        end
        fprintf(['These keep NaN coordinates. Drop them before the ' ...
                 'simulations, or geocode the few of them by hand.\n']);
    end
end

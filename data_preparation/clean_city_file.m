function T = clean_city_file(xlsxFile, maxKm, doPlot)
% CLEAN_CITY_FILE  Remove unusable and misplaced stations from a city file.
%
% Code edited by Luca Ambrosino.
% Optimal Charging Station Selection for EVs.
%
%   T = CLEAN_CITY_FILE(xlsxFile)
%   T = CLEAN_CITY_FILE(xlsxFile, maxKm)
%   T = CLEAN_CITY_FILE(xlsxFile, maxKm, doPlot)
%
% Two kinds of row have to go before a city file can be used in the
% simulations, and they have different causes.
%
%   NO COORDINATES. The geocoder could not resolve the address, so latitude
%   and longitude are NaN. These rows are simply unusable.
%
%   MISPLACED STATIONS. The row has perfectly good coordinates, hundreds of
%   kilometres away from the city. These are NOT geocoding failures: they
%   are errors in the national dataset, where a station is labelled with the
%   wrong city while its street address and ZIP belong somewhere else. In
%   the New York file, for instance, three rows labelled "New York" carried
%   ZIP codes 14456 and 12065, that is Geneva and Clifton Park, two and
%   three hundred kilometres upstate. The geocoder placed them correctly;
%   the label was wrong. Left in, they would inflate the extent of the
%   network and distort every distance in the model.
%
% The second kind is detected by distance from the MEDIAN position, not the
% mean: the median is insensitive to the very outliers we are looking for,
% whereas a mean would be dragged towards them and could hide them.
%
% INPUTS
%   xlsxFile  city spreadsheet, already geocoded
%   maxKm     (optional) distance from the city centre beyond which a
%             station is considered misplaced. Default 30 km, which suits a
%             compact city. Sprawling networks need more: Los Angeles or
%             Dallas realistically extend 50 to 60 km, so use maxKm = 60
%             there, and check the report before accepting the result.
%   doPlot    (optional, default true) show the kept and discarded points
%
% OUTPUT
%   T         the cleaned table, also written back to xlsxFile
%
% The function prints every discarded station with its address and ZIP, so
% that the decision can be inspected rather than trusted blindly.

    if nargin < 2 || isempty(maxKm);   maxKm  = 30;    end
    if nargin < 3 || isempty(doPlot);  doPlot = true;  end
    assert(isfile(xlsxFile), 'Cannot find %s', xlsxFile);

    T0 = readtable(xlsxFile);
    n0 = height(T0);

    lat = double(T0.Latitude);
    lon = double(T0.Longitude);

    %% ---------------------------------------- rows without coordinates
    hasCoord = isfinite(lat) & isfinite(lon);
    nNoCoord = nnz(~hasCoord);

    %% ---------------------------------------- misplaced stations
    % Robust centre of the network
    mlat = median(lat(hasCoord));
    mlon = median(lon(hasCoord));

    % Local flat-earth approximation: good to a fraction of a per cent over
    % a city, and far cheaper than Haversine for a screening test.
    dKm = nan(n0,1);
    dKm(hasCoord) = hypot( (lat(hasCoord)-mlat) * 110.574, ...
                           (lon(hasCoord)-mlon) * 111.320 * cosd(mlat) );

    isFar = hasCoord & (dKm > maxKm);
    keep  = hasCoord & ~isFar;

    %% ---------------------------------------- report
    fprintf('\n=== %s ===\n', xlsxFile);
    fprintf('  rows in file            : %d\n', n0);
    fprintf('  removed, no coordinates : %d\n', nNoCoord);
    fprintf('  removed, beyond %3.0f km  : %d\n', maxKm, nnz(isFar));

    if any(isFar)
        fprintf('\n  Discarded as misplaced:\n');
        idx = find(isFar);
        [~, ord] = sort(dKm(idx), 'descend');
        for k = ord'
            i = idx(k);
            fprintf('   %7.1f km  |  %-32s  ZIP %s\n', dKm(i), ...
                    string(T0.StreetAddress(i)), string(T0.ZIP(i)));
        end
        fprintf(['\n  Check the ZIP codes: if they belong to another town, the ' ...
                 'national dataset\n  mislabelled the city and the removal is ' ...
                 'correct.\n']);
    end

    T = T0(keep, :);
    fprintf('\n  KEPT                    : %d stations, %d ports\n', ...
            height(T), sum(double(T.NumPorts)));
    if height(T) > 0
        la = double(T.Latitude);  lo = double(T.Longitude);
        fprintf('  extent                  : %.1f km N-S x %.1f km E-W\n', ...
                (max(la)-min(la))*110.574, ...
                (max(lo)-min(lo))*111.320*cosd(mlat));
    end

    %% ---------------------------------------- optional check plot
    if doPlot && any(hasCoord)
        figure('Position', [100 100 640 620]);
        hold on; grid on; box on;
        scatter(lon(keep),  lat(keep),  22, [0.00 0.35 0.75], 'filled', ...
                'MarkerFaceAlpha', 0.75, 'DisplayName', 'kept');
        if any(isFar)
            scatter(lon(isFar), lat(isFar), 70, [0.85 0.20 0.15], 'x', ...
                    'LineWidth', 2, 'DisplayName', 'discarded');
        end
        plot(mlon, mlat, 'kp', 'MarkerSize', 16, 'MarkerFaceColor', 'y', ...
             'DisplayName', 'median centre');
        xlabel('Longitude'); ylabel('Latitude');
        title(sprintf('%s: %d kept, %d discarded', xlsxFile, height(T), nnz(isFar)), ...
              'Interpreter', 'none');
        legend('Location', 'best'); axis equal;
        set(gca, 'FontSize', 11);
    end

    writetable(T, xlsxFile);
    fprintf('  File overwritten with the cleaned table.\n\n');
end

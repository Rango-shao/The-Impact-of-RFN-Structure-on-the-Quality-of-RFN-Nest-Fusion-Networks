% script for analyse the fused images

% Define folder paths
ir_greyFolder = 'F:\ImageFusion\new_test_image\ir_grey\';
vis_greyFolder = 'F:\ImageFusion\new_test_image\vis_grey\';
alphaFolder = 'F:\ImageFusion\code\RFN-Nest-S\rfn-D1\outputs\alpha_1e4_21\';

% Get lists of all image files in the folders
irFiles = dir(fullfile(ir_greyFolder, '*.jpg')); % Assumes images are in jpg format
visFiles = dir(fullfile(vis_greyFolder, '*.jpg')); % Assumes images are in jpg format
alphaFiles = dir(fullfile(alphaFolder, '*.jpg')); % Assumes images are in jpg format

% Extract the numerical part of the filenames
numIR = zeros(length(irFiles), 1);
for k = 1:length(irFiles)
    matches = regexp(irFiles(k).name, '\d+', 'match'); % Find all digit matches
    if ~isempty(matches)
        numIR(k) = str2double(matches{1}); % Take the first match
    end
end

numVIS = zeros(length(visFiles), 1);
for k = 1:length(visFiles)
    matches = regexp(visFiles(k).name, '\d+', 'match'); % Find all digit matches
    if ~isempty(matches)
        numVIS(k) = str2double(matches{1}); % Take the first match
    end
end

numAlpha = zeros(length(alphaFiles), 1);
for k = 1:length(alphaFiles)
    matches = regexp(alphaFiles(k).name, '\d+', 'match'); % Find all digit matches
    if ~isempty(matches)
        numAlpha(k) = str2double(matches{1}); % Take the first match
    end
end

% Combine the numerical parts and sort them
[sortedNums, sortedIdx] = sort(numIR); % Sort based on IR images

% Reorder the file lists based on the sorted indices
irFiles = irFiles(sortedIdx);
visFiles = visFiles(sortedIdx);
alphaFiles = alphaFiles(sortedIdx);

% Initialize a table to store all the metrics
metricsTable = table();

% Loop through each image and calculate the metrics
for k = 1:length(irFiles)
    % Read the images
    fileName_source_l = fullfile(ir_greyFolder, irFiles(k).name);
    fileName_source_r = fullfile(vis_greyFolder, visFiles(k).name);
    fileName_fused = fullfile(alphaFolder, alphaFiles(k).name);

    fprintf('Processing image %d...\n', k);

    % Read the images
    fused_image = imread(fileName_fused);
    sourceTestImage1 = imread(fileName_source_l);
    sourceTestImage2 = imread(fileName_source_r);

    tic;
    metrics = analysis_Reference(fused_image, sourceTestImage1, sourceTestImage2);
    toc;

    % Store the results in the table
    metricsTable = [metricsTable; {sortedNums(k), irFiles(k).name, visFiles(k).name, alphaFiles(k).name, ...
        metrics.EN, metrics.SD, metrics.MI, metrics.Nabf, metrics.SCD, metrics.MS_SSIM}];
end

% Define the variable names for the table
varNames = {'ImageNumber', 'IR_Image', 'VIS_Image', 'Fused_Image', 'EN', 'SD', 'MI', 'Nabf', 'SCD', 'MS_SSIM'};
metricsTable.Properties.VariableNames = varNames;

% Display the table
disp(metricsTable);

% Optionally: Save the table to an Excel file
writetable(metricsTable, 'metricsResults_D1.xlsx');
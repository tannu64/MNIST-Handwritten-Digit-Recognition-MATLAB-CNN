function [images, labels] = load_mnist_data(dataType)
%LOAD_MNIST_DATA Load MNIST dataset for handwritten digit recognition
%
% Syntax:
%   [images, labels] = load_mnist_data(dataType)
%
% Inputs:
%   dataType - String, either 'train' or 'test' to specify dataset type
%
% Outputs:
%   images - 4D array of size [28, 28, 1, N] containing digit images
%   labels - Categorical array of length N containing digit labels (0-9)
%
% Description:
%   This function attempts to load the MNIST dataset. If MNIST is not
%   available, it falls back to MATLAB's built-in digit dataset.
%
% Example:
%   [trainImages, trainLabels] = load_mnist_data('train');
%   [testImages, testLabels] = load_mnist_data('test');

    % Validate input
    if ~ismember(dataType, {'train', 'test'})
        error('dataType must be either ''train'' or ''test''');
    end
    
    try
        % Try to load MNIST data using Deep Learning Toolbox
        if strcmp(dataType, 'train')
            % Load training data (60,000 images)
            fprintf('Loading MNIST training data...\n');
            
            % Check if digitTrain4DArrayData exists (newer MATLAB versions)
            if exist('digitTrain4DArrayData', 'file')
                [images, labels] = digitTrain4DArrayData;
            else
                % Alternative method for older MATLAB versions
                ds = imageDatastore('mnist_train', 'IncludeSubfolders', true, ...
                    'LabelSource', 'foldernames');
                [images, labels] = load_from_datastore(ds, 60000);
            end
            
        else % test
            % Load test data (10,000 images)
            fprintf('Loading MNIST test data...\n');
            
            % Check if digitTest4DArrayData exists (newer MATLAB versions)
            if exist('digitTest4DArrayData', 'file')
                [images, labels] = digitTest4DArrayData;
            else
                % Alternative method for older MATLAB versions
                ds = imageDatastore('mnist_test', 'IncludeSubfolders', true, ...
                    'LabelSource', 'foldernames');
                [images, labels] = load_from_datastore(ds, 10000);
            end
        end
        
        % Ensure images are in correct format [28, 28, 1, N]
        if size(images, 3) ~= 1
            images = rgb2gray(images);
        end
        
        % Ensure labels are categorical
        if ~iscategorical(labels)
            labels = categorical(labels);
        end
        
        fprintf('Successfully loaded %d images of size [%d, %d, %d]\n', ...
            size(images, 4), size(images, 1), size(images, 2), size(images, 3));
            
    catch ME
        % If MNIST loading fails, display warning and use alternative
        warning('MNIST data not found. Error: %s', ME.message);
        fprintf('Falling back to alternative digit dataset...\n');
        
        % Use alternative loading method
        [images, labels] = load_alternative_digit_data(dataType);
    end
end

function [images, labels] = load_from_datastore(ds, maxSamples)
%LOAD_FROM_DATASTORE Helper function to load data from image datastore
%
% Inputs:
%   ds - ImageDatastore object
%   maxSamples - Maximum number of samples to load
%
% Outputs:
%   images - 4D array of images
%   labels - Categorical array of labels

    % Initialize arrays
    numFiles = min(length(ds.Files), maxSamples);
    images = zeros(28, 28, 1, numFiles, 'uint8');
    labels = categorical.empty(numFiles, 0);
    
    % Load images in batches for memory efficiency
    batchSize = 1000;
    idx = 1;
    
    fprintf('Loading %d images in batches of %d...\n', numFiles, batchSize);
    
    while hasdata(ds) && idx <= numFiles
        % Determine batch end index
        endIdx = min(idx + batchSize - 1, numFiles);
        currentBatchSize = endIdx - idx + 1;
        
        % Read batch of images
        for i = 1:currentBatchSize
            if hasdata(ds)
                [img, info] = read(ds);
                
                % Resize to 28x28 if necessary
                if ~isequal(size(img), [28, 28]) && ~isequal(size(img), [28, 28, 1])
                    img = imresize(img, [28, 28]);
                end
                
                % Convert to grayscale if necessary
                if size(img, 3) > 1
                    img = rgb2gray(img);
                end
                
                % Store image and label
                images(:, :, 1, idx) = img;
                labels(idx) = info.Label;
                idx = idx + 1;
            end
        end
        
        % Display progress
        fprintf('Loaded %d/%d images (%.1f%%)\n', idx-1, numFiles, ...
            (idx-1)/numFiles*100);
    end
    
    % Trim arrays to actual loaded size
    images = images(:, :, :, 1:idx-1);
    labels = labels(1:idx-1);
end

function [images, labels] = load_alternative_digit_data(dataType)
%LOAD_ALTERNATIVE_DIGIT_DATA Load alternative digit dataset when MNIST unavailable
%
% This function creates a synthetic dataset or loads from other sources
% when the standard MNIST dataset is not available.

    fprintf('Creating alternative digit dataset...\n');
    
    if strcmp(dataType, 'train')
        numSamples = 6000;  % Reduced size for alternative dataset
    else
        numSamples = 1000;  % Test set size
    end
    
    % Generate synthetic digit-like data or use built-in patterns
    images = zeros(28, 28, 1, numSamples, 'uint8');
    labels = categorical.empty(numSamples, 0);
    
    for i = 1:numSamples
        % Generate random digit (0-9)
        digit = randi([0, 9]);
        
        % Create simple digit pattern
        img = create_simple_digit_pattern(digit);
        
        % Add some noise for realism
        noise = randn(28, 28) * 10;
        img = uint8(max(0, min(255, double(img) + noise)));
        
        images(:, :, 1, i) = img;
        labels(i) = categorical(digit);
    end
    
    fprintf('Generated %d synthetic digit images\n', numSamples);
end

function img = create_simple_digit_pattern(digit)
%CREATE_SIMPLE_DIGIT_PATTERN Create a simple pattern for a given digit
%
% This function creates basic digit patterns for demonstration purposes
% when MNIST data is not available.

    % Initialize 28x28 image
    img = zeros(28, 28, 'uint8');
    
    % Create simple patterns based on digit
    switch digit
        case 0
            % Draw circle for 0
            [X, Y] = meshgrid(1:28, 1:28);
            center = 14;
            radius = 8;
            circle = ((X - center).^2 + (Y - center).^2) <= radius^2 & ...
                     ((X - center).^2 + (Y - center).^2) >= (radius-3)^2;
            img(circle) = 255;
            
        case 1
            % Draw vertical line for 1
            img(8:20, 13:15) = 255;
            
        case 2
            % Draw Z-like pattern for 2
            img(8:10, 8:20) = 255;    % Top horizontal
            img(10:15, 12:17) = 255;  % Diagonal
            img(18:20, 8:20) = 255;   % Bottom horizontal
            
        case 3
            % Draw two horizontal lines for 3
            img(8:10, 10:18) = 255;   % Top
            img(13:15, 10:18) = 255;  % Middle
            img(18:20, 10:18) = 255;  % Bottom
            
        case 4
            % Draw L-like pattern for 4
            img(8:15, 10:12) = 255;   % Vertical left
            img(13:15, 10:18) = 255;  % Horizontal
            img(8:20, 16:18) = 255;   % Vertical right
            
        case 5
            % Draw S-like pattern for 5
            img(8:10, 8:18) = 255;    % Top
            img(10:13, 8:10) = 255;   % Left vertical
            img(13:15, 8:18) = 255;   % Middle
            img(15:18, 16:18) = 255;  % Right vertical
            img(18:20, 8:18) = 255;   % Bottom
            
        case 6
            % Draw 6 pattern
            img(8:20, 8:10) = 255;    % Left vertical
            img(8:10, 8:18) = 255;    % Top
            img(13:15, 8:18) = 255;   % Middle
            img(18:20, 8:18) = 255;   % Bottom
            img(15:18, 16:18) = 255;  % Bottom right vertical
            
        case 7
            % Draw 7 pattern
            img(8:10, 8:18) = 255;    % Top horizontal
            img(10:20, 15:17) = 255;  % Diagonal
            
        case 8
            % Draw 8 pattern (two rectangles)
            img(8:12, 10:18) = 255;   % Top rectangle outline
            img(8:12, 12:16) = 0;     % Top rectangle inside
            img(16:20, 10:18) = 255;  % Bottom rectangle outline
            img(16:20, 12:16) = 0;    % Bottom rectangle inside
            img(12:16, 10:18) = 255;  % Middle connection
            
        case 9
            % Draw 9 pattern
            img(8:12, 10:18) = 255;   % Top rectangle
            img(8:18, 16:18) = 255;   % Right vertical
            img(12:14, 10:18) = 255;  % Middle horizontal
    end
end

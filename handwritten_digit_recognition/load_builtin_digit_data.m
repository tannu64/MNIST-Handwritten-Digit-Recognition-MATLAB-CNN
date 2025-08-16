function [trainImages, trainLabels, testImages, testLabels] = load_builtin_digit_data()
%LOAD_BUILTIN_DIGIT_DATA Load MATLAB's built-in digit dataset
%
% Syntax:
%   [trainImages, trainLabels, testImages, testLabels] = load_builtin_digit_data()
%
% Outputs:
%   trainImages - 4D array [28, 28, 1, N_train] of training images
%   trainLabels - Categorical array of training labels
%   testImages  - 4D array [28, 28, 1, N_test] of test images  
%   testLabels  - Categorical array of test labels
%
% Description:
%   This function loads MATLAB's built-in digit dataset and splits it
%   into training and test sets. This serves as a fallback when MNIST
%   dataset is not available.

    fprintf('Loading MATLAB built-in digit dataset...\n');
    
    try
        % Try to load digitTrain4DArrayData (MATLAB R2019b and later)
        if exist('digitTrain4DArrayData', 'file') == 2
            [trainImages, trainLabels] = digitTrain4DArrayData;
            fprintf('Loaded training data: %d images\n', size(trainImages, 4));
        else
            % Create training data from synthetic patterns
            fprintf('digitTrain4DArrayData not found, creating synthetic training data...\n');
            [trainImages, trainLabels] = create_synthetic_training_data();
        end
        
        % Try to load digitTest4DArrayData
        if exist('digitTest4DArrayData', 'file') == 2
            [testImages, testLabels] = digitTest4DArrayData;
            fprintf('Loaded test data: %d images\n', size(testImages, 4));
        else
            % Create test data from synthetic patterns
            fprintf('digitTest4DArrayData not found, creating synthetic test data...\n');
            [testImages, testLabels] = create_synthetic_test_data();
        end
        
    catch ME
        % If built-in datasets fail, create fully synthetic dataset
        warning('Built-in digit datasets not available. Creating synthetic dataset...');
        fprintf('Error details: %s\n', ME.message);
        
        [trainImages, trainLabels] = create_synthetic_training_data();
        [testImages, testLabels] = create_synthetic_test_data();
    end
    
    % Ensure correct data format
    trainImages = ensure_correct_format(trainImages);
    testImages = ensure_correct_format(testImages);
    
    % Ensure labels are categorical
    if ~iscategorical(trainLabels)
        trainLabels = categorical(trainLabels);
    end
    if ~iscategorical(testLabels)
        testLabels = categorical(testLabels);
    end
    
    fprintf('Final dataset summary:\n');
    fprintf('  Training: %d images of size [%d, %d, %d]\n', ...
        size(trainImages, 4), size(trainImages, 1), size(trainImages, 2), size(trainImages, 3));
    fprintf('  Test: %d images of size [%d, %d, %d]\n', ...
        size(testImages, 4), size(testImages, 1), size(testImages, 2), size(testImages, 3));
end

function [images, labels] = create_synthetic_training_data()
%CREATE_SYNTHETIC_TRAINING_DATA Create synthetic training dataset
%
% Creates a synthetic training dataset with multiple variations of each digit

    numSamplesPerDigit = 600;  % 600 samples per digit (0-9)
    totalSamples = numSamplesPerDigit * 10;
    
    fprintf('Creating synthetic training data: %d samples per digit...\n', numSamplesPerDigit);
    
    % Initialize arrays
    images = zeros(28, 28, 1, totalSamples, 'uint8');
    labels = categorical.empty(totalSamples, 0);
    
    idx = 1;
    for digit = 0:9
        fprintf('  Generating digit %d samples...\n', digit);
        
        for sample = 1:numSamplesPerDigit
            % Create base digit pattern
            img = create_digit_pattern_with_variations(digit);
            
            % Store image and label
            images(:, :, 1, idx) = img;
            labels(idx) = categorical(digit);
            idx = idx + 1;
        end
    end
    
    % Shuffle the data
    shuffleIdx = randperm(totalSamples);
    images = images(:, :, :, shuffleIdx);
    labels = labels(shuffleIdx);
    
    fprintf('Synthetic training data created: %d total samples\n', totalSamples);
end

function [images, labels] = create_synthetic_test_data()
%CREATE_SYNTHETIC_TEST_DATA Create synthetic test dataset
%
% Creates a synthetic test dataset with variations different from training

    numSamplesPerDigit = 100;  % 100 samples per digit (0-9)
    totalSamples = numSamplesPerDigit * 10;
    
    fprintf('Creating synthetic test data: %d samples per digit...\n', numSamplesPerDigit);
    
    % Initialize arrays
    images = zeros(28, 28, 1, totalSamples, 'uint8');
    labels = categorical.empty(totalSamples, 0);
    
    idx = 1;
    for digit = 0:9
        for sample = 1:numSamplesPerDigit
            % Create base digit pattern with different variations than training
            img = create_digit_pattern_with_variations(digit, true);
            
            % Store image and label
            images(:, :, 1, idx) = img;
            labels(idx) = categorical(digit);
            idx = idx + 1;
        end
    end
    
    % Shuffle the data
    shuffleIdx = randperm(totalSamples);
    images = images(:, :, :, shuffleIdx);
    labels = labels(shuffleIdx);
    
    fprintf('Synthetic test data created: %d total samples\n', totalSamples);
end

function img = create_digit_pattern_with_variations(digit, isTestData)
%CREATE_DIGIT_PATTERN_WITH_VARIATIONS Create digit pattern with random variations
%
% Inputs:
%   digit - Integer from 0 to 9
%   isTestData - Boolean, if true applies different variations for test set

    if nargin < 2
        isTestData = false;
    end
    
    % Create base pattern
    img = create_base_digit_pattern(digit);
    
    % Apply random variations
    img = apply_random_variations(img, isTestData);
    
    % Ensure uint8 format and proper range
    img = uint8(max(0, min(255, img)));
end

function img = create_base_digit_pattern(digit)
%CREATE_BASE_DIGIT_PATTERN Create basic pattern for each digit
%
% This creates more sophisticated patterns than the simple version

    img = zeros(28, 28);
    
    switch digit
        case 0
            % Create oval/circle for 0
            [X, Y] = meshgrid(1:28, 1:28);
            centerX = 14 + randn() * 2;  % Add slight randomness
            centerY = 14 + randn() * 2;
            radiusX = 6 + randn() * 1;
            radiusY = 8 + randn() * 1;
            
            oval = ((X - centerX).^2 / radiusX^2 + (Y - centerY).^2 / radiusY^2) <= 1 & ...
                   ((X - centerX).^2 / (radiusX-2)^2 + (Y - centerY).^2 / (radiusY-2)^2) >= 1;
            img(oval) = 200 + randn() * 20;
            
        case 1
            % Create vertical line with slight variations
            centerX = 14 + randn() * 2;
            startY = 6 + randi(3);
            endY = 22 + randi(3);
            width = 2 + randi(2);
            
            x1 = max(1, round(centerX - width/2));
            x2 = min(28, round(centerX + width/2));
            y1 = max(1, startY);
            y2 = min(28, endY);
            
            img(y1:y2, x1:x2) = 200 + randn() * 20;
            
            % Add small top serif
            if rand() > 0.5
                img(y1:y1+1, x1-1:x2+1) = 180 + randn() * 20;
            end
            
        case 2
            % Create S-like curve for 2
            create_curved_2(img);
            
        case 3
            % Create 3 with curved edges
            create_curved_3(img);
            
        case 4
            % Create 4 with angled parts
            create_angled_4(img);
            
        case 5
            % Create 5 with proper curves
            create_curved_5(img);
            
        case 6
            % Create 6 with circular bottom
            create_curved_6(img);
            
        case 7
            % Create 7 with diagonal
            create_diagonal_7(img);
            
        case 8
            % Create 8 with two circles
            create_double_circle_8(img);
            
        case 9
            % Create 9 with circular top
            create_curved_9(img);
    end
    
    % Helper functions for complex digit patterns would be defined here
    % For brevity, using simplified implementations
    
    function create_curved_2(img)
        % Simplified 2 pattern
        img(7:9, 6:20) = 200;     % Top horizontal
        img(9:15, 15:20) = 200;   % Right vertical
        img(13:15, 6:20) = 200;   % Middle diagonal/horizontal
        img(15:20, 6:10) = 200;   % Left vertical
        img(18:20, 6:20) = 200;   % Bottom horizontal
    end
    
    function create_curved_3(img)
        % Simplified 3 pattern
        img(7:9, 8:18) = 200;     % Top horizontal
        img(13:15, 8:16) = 200;   % Middle horizontal
        img(18:20, 8:18) = 200;   % Bottom horizontal
        img(9:13, 16:18) = 200;   % Top right vertical
        img(15:18, 16:18) = 200;  % Bottom right vertical
    end
    
    function create_angled_4(img)
        % Simplified 4 pattern
        img(7:14, 8:10) = 200;    % Left vertical
        img(12:14, 8:18) = 200;   % Horizontal crossbar
        img(7:20, 16:18) = 200;   % Right vertical
    end
    
    function create_curved_5(img)
        % Simplified 5 pattern
        img(7:9, 6:18) = 200;     % Top horizontal
        img(9:13, 6:8) = 200;     % Left vertical
        img(13:15, 6:16) = 200;   % Middle horizontal
        img(15:18, 14:16) = 200;  % Right vertical
        img(18:20, 6:16) = 200;   % Bottom horizontal
    end
    
    function create_curved_6(img)
        % Simplified 6 pattern
        img(7:20, 6:8) = 200;     % Left vertical
        img(7:9, 6:16) = 200;     % Top horizontal
        img(13:15, 6:18) = 200;   % Middle horizontal
        img(18:20, 6:18) = 200;   % Bottom horizontal
        img(15:18, 16:18) = 200;  % Bottom right vertical
    end
    
    function create_diagonal_7(img)
        % Simplified 7 pattern
        img(7:9, 6:18) = 200;     % Top horizontal
        % Create diagonal line
        for i = 1:13
            y = 9 + i;
            x = 18 - i;
            if y <= 28 && x >= 1
                img(y, x:x+1) = 200;
            end
        end
    end
    
    function create_double_circle_8(img)
        % Simplified 8 pattern - two stacked rectangles
        img(7:13, 8:18) = 200;    % Top rectangle outline
        img(9:11, 10:16) = 0;     % Top rectangle hollow
        img(15:21, 8:18) = 200;   % Bottom rectangle outline
        img(17:19, 10:16) = 0;    % Bottom rectangle hollow
    end
    
    function create_curved_9(img)
        % Simplified 9 pattern
        img(7:13, 8:18) = 200;    % Top rectangle
        img(9:11, 10:16) = 0;     % Make it hollow
        img(7:20, 16:18) = 200;   % Right vertical
        img(13:15, 8:18) = 200;   % Middle horizontal
    end
end

function img = apply_random_variations(img, isTestData)
%APPLY_RANDOM_VARIATIONS Apply random transformations to make data more realistic
%
% Applies various transformations like rotation, scaling, noise, etc.

    % Apply different variation intensities for train vs test
    if isTestData
        variationIntensity = 0.7;  % Less variation for test data
    else
        variationIntensity = 1.0;  % Full variation for training data
    end
    
    % 1. Add Gaussian noise
    noiseLevel = 15 * variationIntensity;
    noise = randn(size(img)) * noiseLevel;
    img = img + noise;
    
    % 2. Apply small random rotation
    angle = (randn() * 10) * variationIntensity;  % ±10 degrees max
    if abs(angle) > 1
        img = imrotate(img, angle, 'bilinear', 'crop');
    end
    
    % 3. Apply small random translation
    shiftX = round(randn() * 2 * variationIntensity);
    shiftY = round(randn() * 2 * variationIntensity);
    if shiftX ~= 0 || shiftY ~= 0
        img = imtranslate(img, [shiftX, shiftY]);
    end
    
    % 4. Apply slight scaling variation
    scaleFactor = 1 + (randn() * 0.1 * variationIntensity);
    if abs(scaleFactor - 1) > 0.05
        img = imresize(img, scaleFactor);
        % Crop or pad to maintain 28x28 size
        [h, w] = size(img);
        if h ~= 28 || w ~= 28
            % Crop from center or pad with zeros
            if h > 28 || w > 28
                startR = max(1, round((h-28)/2) + 1);
                startC = max(1, round((w-28)/2) + 1);
                img = img(startR:startR+27, startC:startC+27);
            else
                padR = (28 - h) / 2;
                padC = (28 - w) / 2;
                img = padarray(img, [floor(padR), floor(padC)], 0, 'both');
                if size(img, 1) < 28
                    img = padarray(img, [1, 0], 0, 'post');
                end
                if size(img, 2) < 28
                    img = padarray(img, [0, 1], 0, 'post');
                end
                img = img(1:28, 1:28);
            end
        end
    end
    
    % 5. Add random intensity variation
    intensityVar = 1 + (randn() * 0.2 * variationIntensity);
    img = img * intensityVar;
    
    % 6. Apply random blur occasionally
    if rand() < 0.3 * variationIntensity
        blurSigma = 0.5 + rand() * 0.5;
        img = imgaussfilt(img, blurSigma);
    end
end

function images = ensure_correct_format(images)
%ENSURE_CORRECT_FORMAT Ensure images are in correct 4D format
%
% Converts images to [height, width, channels, num_images] format

    % Get current dimensions
    dims = size(images);
    
    if length(dims) == 2
        % Single image [H, W] -> [H, W, 1, 1]
        images = reshape(images, dims(1), dims(2), 1, 1);
    elseif length(dims) == 3
        if dims(3) <= 3
            % Multiple color images [H, W, C] -> [H, W, C, 1]
            images = reshape(images, dims(1), dims(2), dims(3), 1);
        else
            % Multiple grayscale images [H, W, N] -> [H, W, 1, N]
            images = reshape(images, dims(1), dims(2), 1, dims(3));
        end
    elseif length(dims) == 4
        % Already in correct format [H, W, C, N]
        % Check if channels need to be 1 for grayscale
        if dims(3) > 1
            % Convert to grayscale if needed
            grayImages = zeros(dims(1), dims(2), 1, dims(4), 'like', images);
            for i = 1:dims(4)
                if size(images, 3) == 3
                    grayImages(:, :, 1, i) = rgb2gray(images(:, :, :, i));
                else
                    grayImages(:, :, 1, i) = images(:, :, 1, i);
                end
            end
            images = grayImages;
        end
    end
    
    % Ensure correct data type
    if ~isa(images, 'uint8')
        % Normalize to 0-255 range if needed
        if max(images(:)) <= 1
            images = images * 255;
        end
        images = uint8(images);
    end
end

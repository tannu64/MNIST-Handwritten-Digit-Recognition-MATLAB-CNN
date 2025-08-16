function interactive_digit_testing(net)
%INTERACTIVE_DIGIT_TESTING Interactive digit testing interface
%
% Syntax:
%   interactive_digit_testing(net)
%
% Inputs:
%   net - Trained CNN network for digit classification
%
% Description:
%   Provides an interactive interface for testing the trained CNN model
%   with user-provided images or generated test cases. Features include:
%   - Load and test custom images
%   - Generate synthetic test digits
%   - Real-time prediction display
%   - Confidence scores for all classes
%   - Image preprocessing visualization

    if nargin < 1 || isempty(net)
        error('A trained network must be provided');
    end
    
    fprintf('=== INTERACTIVE DIGIT TESTING INTERFACE ===\n');
    fprintf('This interface allows you to test the trained CNN model\n');
    fprintf('with various input images and see real-time predictions.\n\n');
    
    % Main interactive loop
    while true
        fprintf('Available Options:\n');
        fprintf('  1. Test with random synthetic digit\n');
        fprintf('  2. Test with custom image file\n');
        fprintf('  3. Test with noisy digit\n');
        fprintf('  4. Batch test multiple images\n');
        fprintf('  5. Show prediction confidence analysis\n');
        fprintf('  6. Exit interactive testing\n\n');
        
        choice = input('Enter your choice (1-6): ');
        
        switch choice
            case 1
                test_synthetic_digit(net);
            case 2
                test_custom_image(net);
            case 3
                test_noisy_digit(net);
            case 4
                batch_test_images(net);
            case 5
                confidence_analysis(net);
            case 6
                fprintf('Exiting interactive testing. Thank you!\n');
                break;
            otherwise
                fprintf('Invalid choice. Please enter a number between 1-6.\n\n');
        end
        
        fprintf('\n');
    end
end

function test_synthetic_digit(net)
%TEST_SYNTHETIC_DIGIT Generate and test a synthetic digit
%
% Creates a synthetic digit and shows the prediction process

    fprintf('\n--- Testing Synthetic Digit ---\n');
    
    % Generate random digit
    targetDigit = randi([0, 9]);
    fprintf('Generating synthetic digit: %d\n', targetDigit);
    
    % Create synthetic image
    syntheticImg = create_synthetic_digit_image(targetDigit);
    
    % Preprocess for network
    processedImg = preprocess_single_image(syntheticImg);
    
    % Make prediction
    [predictedLabel, scores] = classify(net, processedImg);
    
    % Display results
    display_prediction_results(syntheticImg, targetDigit, predictedLabel, scores, 'Synthetic Digit Test');
end

function test_custom_image(net)
%TEST_CUSTOM_IMAGE Test with user-provided image file
%
% Allows user to load and test their own digit image

    fprintf('\n--- Testing Custom Image ---\n');
    
    % Get image file from user
    [filename, pathname] = uigetfile(...
        {'*.png;*.jpg;*.jpeg;*.bmp;*.tiff', 'Image Files (*.png,*.jpg,*.jpeg,*.bmp,*.tiff)'; ...
         '*.*', 'All Files (*.*)'}, ...
        'Select an image file containing a digit');
    
    if isequal(filename, 0)
        fprintf('No file selected.\n');
        return;
    end
    
    try
        % Load and preprocess image
        fullPath = fullfile(pathname, filename);
        fprintf('Loading image: %s\n', filename);
        
        originalImg = imread(fullPath);
        processedImg = preprocess_custom_image(originalImg);
        
        % Make prediction
        [predictedLabel, scores] = classify(net, processedImg);
        
        % Display results
        display_prediction_results(processedImg(:,:,1), 'Custom', predictedLabel, scores, ...
            sprintf('Custom Image: %s', filename));
        
    catch ME
        fprintf('Error loading or processing image: %s\n', ME.message);
    end
end

function test_noisy_digit(net)
%TEST_NOISY_DIGIT Test with noise-corrupted digit
%
% Creates a digit with various types of noise to test robustness

    fprintf('\n--- Testing Noisy Digit ---\n');
    
    % Generate base digit
    targetDigit = randi([0, 9]);
    fprintf('Generating noisy digit: %d\n', targetDigit);
    
    % Create clean synthetic image
    cleanImg = create_synthetic_digit_image(targetDigit);
    
    % Add noise
    noiseType = randi([1, 3]);
    switch noiseType
        case 1
            noisyImg = add_gaussian_noise(cleanImg);
            noiseDesc = 'Gaussian Noise';
        case 2
            noisyImg = add_salt_pepper_noise(cleanImg);
            noiseDesc = 'Salt & Pepper Noise';
        case 3
            noisyImg = add_blur_noise(cleanImg);
            noiseDesc = 'Blur Noise';
    end
    
    % Preprocess for network
    processedImg = preprocess_single_image(noisyImg);
    
    % Make prediction
    [predictedLabel, scores] = classify(net, processedImg);
    
    % Display results
    display_prediction_results(noisyImg, targetDigit, predictedLabel, scores, ...
        sprintf('Noisy Digit Test (%s)', noiseDesc));
end

function batch_test_images(net)
%BATCH_TEST_IMAGES Test multiple images in batch
%
% Generates and tests multiple images to show batch prediction capabilities

    fprintf('\n--- Batch Testing ---\n');
    
    numImages = input('Enter number of images to test (1-20): ');
    if isempty(numImages) || numImages < 1 || numImages > 20
        numImages = 5;
        fprintf('Using default: %d images\n', numImages);
    end
    
    fprintf('Generating and testing %d images...\n', numImages);
    
    % Generate batch of images
    images = zeros(28, 28, 1, numImages, 'single');
    trueLabels = zeros(numImages, 1);
    
    for i = 1:numImages
        digit = randi([0, 9]);
        img = create_synthetic_digit_image(digit);
        images(:, :, 1, i) = preprocess_single_image(img);
        trueLabels(i) = digit;
    end
    
    % Make batch predictions
    predictedLabels = classify(net, images);
    
    % Calculate accuracy
    accuracy = sum(double(predictedLabels) == trueLabels) / numImages;
    
    % Display results
    fprintf('\nBatch Test Results:\n');
    fprintf('  Images tested: %d\n', numImages);
    fprintf('  Correct predictions: %d\n', sum(double(predictedLabels) == trueLabels));
    fprintf('  Accuracy: %.1f%%\n', accuracy * 100);
    
    % Show individual results
    fprintf('\nIndividual Results:\n');
    for i = 1:numImages
        isCorrect = (double(predictedLabels(i)) == trueLabels(i));
        status = ternary(isCorrect, '✓', '✗');
        fprintf('  Image %d: True=%d, Pred=%d %s\n', i, trueLabels(i), ...
            double(predictedLabels(i)), status);
    end
    
    % Visualize some results
    visualize_batch_results(images, trueLabels, predictedLabels, min(9, numImages));
end

function confidence_analysis(net)
%CONFIDENCE_ANALYSIS Analyze prediction confidence for different scenarios
%
% Tests various challenging scenarios to analyze model confidence

    fprintf('\n--- Confidence Analysis ---\n');
    
    scenarios = {
        'Clear digit', @() create_synthetic_digit_image(randi([0,9]));
        'Noisy digit', @() add_gaussian_noise(create_synthetic_digit_image(randi([0,9])));
        'Blurred digit', @() add_blur_noise(create_synthetic_digit_image(randi([0,9])));
        'Rotated digit', @() rotate_digit(create_synthetic_digit_image(randi([0,9])));
    };
    
    fprintf('Testing confidence in different scenarios...\n');
    
    figure('Name', 'Confidence Analysis', 'Position', [200, 100, 1200, 800]);
    
    for i = 1:length(scenarios)
        scenarioName = scenarios{i, 1};
        imageGenerator = scenarios{i, 2};
        
        % Generate test image
        testImg = imageGenerator();
        processedImg = preprocess_single_image(testImg);
        
        % Get prediction with scores
        [predictedLabel, scores] = classify(net, processedImg);
        
        % Convert scores to probabilities (softmax is already applied)
        probabilities = double(scores);
        
        % Display in subplot
        subplot(2, length(scenarios), i);
        imshow(testImg, []);
        title(sprintf('%s\nPred: %s', scenarioName, string(predictedLabel)));
        
        subplot(2, length(scenarios), i + length(scenarios));
        bar(0:9, probabilities);
        title(sprintf('Confidence Distribution\nMax: %.2f%%', max(probabilities) * 100));
        xlabel('Digit Class');
        ylabel('Confidence');
        ylim([0, 1]);
        
        % Highlight predicted class
        hold on;
        predDigit = double(predictedLabel);
        bar(predDigit, probabilities(predDigit + 1), 'r');
        hold off;
    end
    
    sgtitle('Prediction Confidence Analysis', 'FontSize', 16, 'FontWeight', 'bold');
end

function img = create_synthetic_digit_image(digit)
%CREATE_SYNTHETIC_DIGIT_IMAGE Create a synthetic digit image
%
% Helper function to generate synthetic digit patterns

    img = zeros(28, 28, 'single');
    
    % Create simple patterns for each digit
    switch digit
        case 0
            % Circle
            [X, Y] = meshgrid(1:28, 1:28);
            center = 14;
            radius = 8;
            circle = ((X - center).^2 + (Y - center).^2) <= radius^2 & ...
                     ((X - center).^2 + (Y - center).^2) >= (radius-3)^2;
            img(circle) = 1;
            
        case 1
            % Vertical line
            img(6:22, 12:16) = 1;
            
        case 2
            % Z pattern
            img(6:8, 6:22) = 1;   % Top
            img(8:20, 18:22) = 1; % Right diagonal approximation
            img(20:22, 6:22) = 1; % Bottom
            
        case 3
            % Three horizontal lines
            img(6:8, 8:20) = 1;   % Top
            img(13:15, 8:18) = 1; % Middle
            img(20:22, 8:20) = 1; % Bottom
            
        case 4
            % Four pattern
            img(6:14, 8:10) = 1;  % Left vertical
            img(12:14, 8:20) = 1; % Horizontal
            img(6:22, 18:20) = 1; % Right vertical
            
        case 5
            % Five pattern
            img(6:8, 6:20) = 1;   % Top
            img(8:14, 6:8) = 1;   % Left vertical
            img(12:14, 6:18) = 1; % Middle
            img(14:20, 16:18) = 1; % Right vertical
            img(20:22, 6:18) = 1; % Bottom
            
        case 6
            % Six pattern
            img(6:22, 6:8) = 1;   % Left vertical
            img(6:8, 6:18) = 1;   % Top
            img(12:14, 6:18) = 1; % Middle
            img(20:22, 6:18) = 1; % Bottom
            img(14:20, 16:18) = 1; % Bottom right
            
        case 7
            % Seven pattern
            img(6:8, 6:20) = 1;   % Top
            img(8:22, 16:18) = 1; % Diagonal
            
        case 8
            % Eight pattern
            img(6:11, 8:18) = 1;  % Top rectangle
            img(17:22, 8:18) = 1; % Bottom rectangle
            img(11:17, 6:8) = 1;  % Left connector
            img(11:17, 18:20) = 1; % Right connector
            
        case 9
            % Nine pattern
            img(6:14, 6:8) = 1;   % Left vertical
            img(6:8, 6:18) = 1;   % Top
            img(12:14, 6:18) = 1; % Middle
            img(6:22, 16:18) = 1; % Right vertical
    end
    
    % Add slight gaussian blur for more realistic appearance
    img = imgaussfilt(img, 0.5);
end

function processedImg = preprocess_single_image(img)
%PREPROCESS_SINGLE_IMAGE Preprocess single image for network input
%
% Ensures image is in correct format for the CNN

    % Ensure image is grayscale
    if size(img, 3) > 1
        img = rgb2gray(img);
    end
    
    % Resize to 28x28 if necessary
    if ~isequal(size(img), [28, 28])
        img = imresize(img, [28, 28]);
    end
    
    % Convert to single precision
    processedImg = single(img);
    
    % Normalize to [0, 1] range
    if max(processedImg(:)) > 1
        processedImg = processedImg / 255;
    end
    
    % Ensure 4D format [H, W, C, N]
    if ndims(processedImg) < 4
        processedImg = reshape(processedImg, 28, 28, 1, 1);
    end
end

function processedImg = preprocess_custom_image(originalImg)
%PREPROCESS_CUSTOM_IMAGE Preprocess user-provided image
%
% Handles various image formats and preprocessing steps

    % Convert to grayscale if needed
    if size(originalImg, 3) > 1
        img = rgb2gray(originalImg);
    else
        img = originalImg;
    end
    
    % Resize to 28x28
    img = imresize(img, [28, 28]);
    
    % Convert to double for processing
    img = double(img);
    
    % Invert if background is dark (common for digit images)
    if mean(img(:)) < 128
        img = 255 - img;
    end
    
    % Normalize to [0, 1]
    img = img / 255;
    
    % Convert to single precision 4D array
    processedImg = single(reshape(img, 28, 28, 1, 1));
end

function noisyImg = add_gaussian_noise(img)
%ADD_GAUSSIAN_NOISE Add Gaussian noise to image
    noise = randn(size(img)) * 0.1;
    noisyImg = img + noise;
    noisyImg = max(0, min(1, noisyImg));
end

function noisyImg = add_salt_pepper_noise(img)
%ADD_SALT_PEPPER_NOISE Add salt and pepper noise to image
    noisyImg = img;
    noiseLevel = 0.05;
    
    % Salt noise (white pixels)
    saltMask = rand(size(img)) < noiseLevel/2;
    noisyImg(saltMask) = 1;
    
    % Pepper noise (black pixels)
    pepperMask = rand(size(img)) < noiseLevel/2;
    noisyImg(pepperMask) = 0;
end

function blurredImg = add_blur_noise(img)
%ADD_BLUR_NOISE Add blur to image
    sigma = 1 + rand() * 1;  % Random blur amount
    blurredImg = imgaussfilt(img, sigma);
end

function rotatedImg = rotate_digit(img)
%ROTATE_DIGIT Rotate digit by random angle
    angle = (rand() - 0.5) * 60;  % ±30 degrees
    rotatedImg = imrotate(img, angle, 'bilinear', 'crop');
end

function display_prediction_results(img, trueLabel, predictedLabel, scores, titleStr)
%DISPLAY_PREDICTION_RESULTS Display comprehensive prediction results
%
% Shows image, prediction, and confidence scores

    figure('Name', titleStr, 'Position', [300, 200, 800, 400]);
    
    % Display image
    subplot(1, 2, 1);
    imshow(img, []);
    if isnumeric(trueLabel)
        title(sprintf('Input Image\nTrue Label: %d', trueLabel), 'FontSize', 12);
    else
        title(sprintf('Input Image\nTrue Label: %s', trueLabel), 'FontSize', 12);
    end
    
    % Display prediction confidence
    subplot(1, 2, 2);
    probabilities = double(scores);
    bars = bar(0:9, probabilities);
    
    % Color the predicted class differently
    predDigit = double(predictedLabel);
    bars.FaceColor = 'flat';
    bars.CData = repmat([0.3, 0.6, 0.9], 10, 1);  % Default blue
    bars.CData(predDigit + 1, :) = [0.9, 0.3, 0.3];  % Red for prediction
    
    title(sprintf('Prediction: %s (%.1f%% confidence)', ...
        string(predictedLabel), max(probabilities) * 100), 'FontSize', 12);
    xlabel('Digit Class');
    ylabel('Confidence');
    ylim([0, 1]);
    grid on;
    
    % Add text annotations
    for i = 1:10
        if probabilities(i) > 0.05  % Only show significant probabilities
            text(i-1, probabilities(i) + 0.02, sprintf('%.1f%%', probabilities(i) * 100), ...
                'HorizontalAlignment', 'center', 'FontSize', 8);
        end
    end
    
    sgtitle(titleStr, 'FontSize', 14, 'FontWeight', 'bold');
end

function visualize_batch_results(images, trueLabels, predictedLabels, numToShow)
%VISUALIZE_BATCH_RESULTS Visualize batch prediction results
%
% Shows grid of images with predictions

    figure('Name', 'Batch Test Results', 'Position', [100, 150, 900, 600]);
    
    gridSize = ceil(sqrt(numToShow));
    
    for i = 1:numToShow
        subplot(gridSize, gridSize, i);
        
        img = images(:, :, 1, i);
        imshow(img, []);
        
        isCorrect = (double(predictedLabels(i)) == trueLabels(i));
        if isCorrect
            borderColor = [0, 1, 0];  % Green
            status = '✓';
        else
            borderColor = [1, 0, 0];  % Red
            status = '✗';
        end
        
        title(sprintf('%s T:%d P:%d', status, trueLabels(i), double(predictedLabels(i))), ...
            'Color', borderColor, 'FontWeight', 'bold');
        
        % Add colored border
        hold on;
        rectangle('Position', [0.5, 0.5, 28, 28], 'EdgeColor', borderColor, 'LineWidth', 2);
        hold off;
        
        set(gca, 'XTick', [], 'YTick', []);
    end
    
    sgtitle('Batch Prediction Results', 'FontSize', 16, 'FontWeight', 'bold');
end

function result = ternary(condition, trueValue, falseValue)
%TERNARY Ternary operator implementation
    if condition
        result = trueValue;
    else
        result = falseValue;
    end
end

function visualize_predictions(images, trueLabels, predictedLabels, numSamples)
%VISUALIZE_PREDICTIONS Display sample predictions with true vs predicted labels
%
% Syntax:
%   visualize_predictions(images, trueLabels, predictedLabels, numSamples)
%
% Inputs:
%   images          - 4D array [H, W, C, N] of digit images
%   trueLabels      - Categorical array of true digit labels
%   predictedLabels - Categorical array of predicted digit labels
%   numSamples      - Number of samples to display (optional, default: 16)
%
% Description:
%   Creates a visualization showing sample predictions with both correct
%   and incorrect classifications highlighted. Correct predictions are
%   shown with green borders, incorrect ones with red borders.

    if nargin < 4
        numSamples = 16;  % Default to 4x4 grid
    end
    
    % Validate inputs
    if size(images, 4) < numSamples
        numSamples = size(images, 4);
    end
    
    if length(trueLabels) ~= length(predictedLabels)
        error('True and predicted labels must have the same length');
    end
    
    if size(images, 4) ~= length(trueLabels)
        error('Number of images must match number of labels');
    end
    
    % Determine grid size for display (will be recalculated after actualSamples is known)
    gridRows = ceil(sqrt(numSamples));
    gridCols = ceil(numSamples / gridRows);
    
    % Validate array bounds first
    totalAvailable = min([size(images, 4), length(trueLabels), length(predictedLabels)]);
    
    % Debug information
    fprintf('Debug Info:\n');
    fprintf('  Images size: %dx%dx%dx%d\n', size(images));
    fprintf('  True labels length: %d\n', length(trueLabels));
    fprintf('  Predicted labels length: %d\n', length(predictedLabels));
    fprintf('  Total available: %d\n', totalAvailable);
    
    if totalAvailable == 0
        fprintf('ERROR: No samples available for visualization.\n');
        fprintf('Please check that all input arrays have the same number of samples.\n');
        return;
    end
    
    % Adjust numSamples if necessary
    if numSamples > totalAvailable
        numSamples = totalAvailable;
        fprintf('Reducing sample count to %d (total available)\n', numSamples);
    end
    
    % Find both correct and incorrect predictions for balanced display
    % Ensure both arrays are the same size and type
    if length(trueLabels) ~= length(predictedLabels)
        error('True labels and predicted labels must have the same length');
    end
    
    % Convert to same format if needed
    if iscategorical(trueLabels) && iscategorical(predictedLabels)
        % Both categorical - direct comparison should work
        correctPredictions = trueLabels == predictedLabels;
    else
        % Convert to strings for comparison
        correctPredictions = string(trueLabels) == string(predictedLabels);
    end
    
    correctIndices = find(correctPredictions);
    incorrectIndices = find(~correctPredictions);
    
    % Debug the prediction matching
    fprintf('  Correct predictions: %d\n', length(correctIndices));
    fprintf('  Incorrect predictions: %d\n', length(incorrectIndices));
    fprintf('  Total predictions: %d\n', length(correctPredictions));
    
    % Additional debug - show some sample comparisons
    if length(trueLabels) >= 5
        fprintf('  Sample comparisons (first 5):\n');
        for i = 1:5
            fprintf('    True: %s, Pred: %s, Match: %s\n', ...
                string(trueLabels(i)), string(predictedLabels(i)), ...
                string(correctPredictions(i)));
        end
    end
    
    % Simplified approach: just select random indices if the complex logic fails
    if isempty(correctIndices) && isempty(incorrectIndices)
        fprintf('  Warning: No correct/incorrect classification found. Using random selection.\n');
        selectedIndices = randperm(totalAvailable, min(numSamples, totalAvailable))';
    else
        % Select a mix of correct and incorrect predictions
        numCorrect = min(ceil(numSamples * 0.7), length(correctIndices));  % 70% correct
        numIncorrect = min(numSamples - numCorrect, length(incorrectIndices));
        
        selectedIndices = [];
        
        % Add correct predictions
        if numCorrect > 0 && ~isempty(correctIndices)
            numCorrectToSelect = min(numCorrect, length(correctIndices));
            selectedCorrect = correctIndices(randperm(length(correctIndices), numCorrectToSelect));
            selectedIndices = [selectedIndices; selectedCorrect];
        end
        
        % Add incorrect predictions
        if numIncorrect > 0 && ~isempty(incorrectIndices)
            numIncorrectToSelect = min(numIncorrect, length(incorrectIndices));
            selectedIncorrect = incorrectIndices(randperm(length(incorrectIndices), numIncorrectToSelect));
            selectedIndices = [selectedIndices; selectedIncorrect];
        end
        
        % If we don't have enough samples, fill with random selections
        if length(selectedIndices) < numSamples
            remainingNeeded = numSamples - length(selectedIndices);
            allIndices = 1:totalAvailable;
            availableIndices = setdiff(allIndices, selectedIndices);
            
            if length(availableIndices) >= remainingNeeded
                additionalIndices = availableIndices(randperm(length(availableIndices), remainingNeeded));
                selectedIndices = [selectedIndices; additionalIndices'];
            else
                selectedIndices = [selectedIndices; availableIndices'];
            end
        end
    end
    
    % Trim to exact number needed and ensure within bounds
    selectedIndices = selectedIndices(1:min(numSamples, length(selectedIndices)));
    selectedIndices = selectedIndices(selectedIndices <= totalAvailable);  % Extra safety check
    actualSamples = length(selectedIndices);
    
    % Final fallback: if still no samples, force random selection
    if actualSamples == 0
        fprintf('  Warning: No samples selected by normal logic. Using fallback random selection.\n');
        selectedIndices = randperm(totalAvailable, min(16, totalAvailable))';
        actualSamples = length(selectedIndices);
        
        if actualSamples == 0
            fprintf('ERROR: Still no valid samples. Cannot visualize.\n');
            return;
        end
        
        fprintf('  Fallback: Selected %d random samples\n', actualSamples);
    end
    
    % Shuffle the selected indices for random display order
    selectedIndices = selectedIndices(randperm(length(selectedIndices)));
    
    % Recalculate grid size based on actual samples
    gridRows = ceil(sqrt(actualSamples));
    gridCols = ceil(actualSamples / gridRows);
    
    fprintf('Displaying %d prediction samples (%d correct, %d incorrect)...\n', ...
        actualSamples, sum(correctPredictions(selectedIndices)), ...
        sum(~correctPredictions(selectedIndices)));
    
    % Create figure
    figure('Name', 'Prediction Results', 'Position', [250, 100, 1000, 800]);
    
    % Track statistics for summary
    correctCount = 0;
    incorrectCount = 0;
    
    for i = 1:actualSamples
        idx = selectedIndices(i);
        
        % Extra safety check for array bounds
        if idx < 1 || idx > size(images, 4) || idx > length(trueLabels) || idx > length(predictedLabels)
            fprintf('Warning: Skipping invalid index %d\n', idx);
            continue;
        end
        
        % Extract image and labels
        img = images(:, :, 1, idx);
        trueLabel = trueLabels(idx);
        predLabel = predictedLabels(idx);
        isCorrect = (trueLabel == predLabel);
        
        % Update counters
        if isCorrect
            correctCount = correctCount + 1;
        else
            incorrectCount = incorrectCount + 1;
        end
        
        % Create subplot
        subplot(gridRows, gridCols, i);
        
        % Display image
        imshow(img, []);
        
        % Create title with color coding
        if isCorrect
            titleColor = [0, 0.7, 0];  % Green for correct
            titleStr = sprintf('✓ True: %s, Pred: %s', string(trueLabel), string(predLabel));
        else
            titleColor = [0.8, 0, 0];  % Red for incorrect
            titleStr = sprintf('✗ True: %s, Pred: %s', string(trueLabel), string(predLabel));
        end
        
        % Add title with appropriate color
        title(titleStr, 'FontSize', 9, 'Color', titleColor, 'FontWeight', 'bold');
        
        % Add colored border to image
        hold on;
        if isCorrect
            % Green border for correct predictions
            rectangle('Position', [0.5, 0.5, size(img, 2), size(img, 1)], ...
                     'EdgeColor', [0, 0.7, 0], 'LineWidth', 2);
        else
            % Red border for incorrect predictions
            rectangle('Position', [0.5, 0.5, size(img, 2), size(img, 1)], ...
                     'EdgeColor', [0.8, 0, 0], 'LineWidth', 2);
        end
        hold off;
        
        % Remove axis ticks
        set(gca, 'XTick', [], 'YTick', []);
    end
    
    % Add main title with summary statistics
    accuracy = correctCount / actualSamples * 100;
    mainTitle = sprintf('Prediction Results (%.1f%% accuracy in sample)', accuracy);
    sgtitle(mainTitle, 'FontSize', 16, 'FontWeight', 'bold');
    
    % Add legend
    subplot(gridRows, gridCols, actualSamples);
    hold on;
    
    % Create invisible plots for legend
    h1 = plot(NaN, NaN, 's', 'Color', [0, 0.7, 0], 'MarkerSize', 10, 'LineWidth', 2);
    h2 = plot(NaN, NaN, 's', 'Color', [0.8, 0, 0], 'MarkerSize', 10, 'LineWidth', 2);
    
    legend([h1, h2], {'Correct Prediction', 'Incorrect Prediction'}, ...
           'Location', 'best', 'FontSize', 10);
    axis off;
    
    % Display summary statistics
    fprintf('\nPrediction Visualization Summary:\n');
    fprintf('  Correct predictions: %d/%d (%.1f%%)\n', ...
        correctCount, actualSamples, correctCount/actualSamples*100);
    fprintf('  Incorrect predictions: %d/%d (%.1f%%)\n', ...
        incorrectCount, actualSamples, incorrectCount/actualSamples*100);
    
    % Analyze incorrect predictions
    if incorrectCount > 0
        analyze_incorrect_predictions(selectedIndices, trueLabels, predictedLabels, correctPredictions);
    end
    
    fprintf('Prediction visualization complete.\n');
end

function analyze_incorrect_predictions(selectedIndices, trueLabels, predictedLabels, correctPredictions)
%ANALYZE_INCORRECT_PREDICTIONS Analyze patterns in incorrect predictions
%
% Provides insights into which digits are most commonly misclassified

    % Find incorrect predictions in the selected samples
    incorrectInSample = selectedIndices(~correctPredictions(selectedIndices));
    
    if isempty(incorrectInSample)
        return;
    end
    
    fprintf('\nIncorrect Prediction Analysis:\n');
    
    % Count misclassification patterns
    misclassificationPatterns = containers.Map();
    
    for i = 1:length(incorrectInSample)
        idx = incorrectInSample(i);
        trueLabel = string(trueLabels(idx));
        predLabel = string(predictedLabels(idx));
        
        pattern = sprintf('%s -> %s', trueLabel, predLabel);
        
        if isKey(misclassificationPatterns, pattern)
            misclassificationPatterns(pattern) = misclassificationPatterns(pattern) + 1;
        else
            misclassificationPatterns(pattern) = 1;
        end
    end
    
    % Display patterns
    patterns = keys(misclassificationPatterns);
    counts = values(misclassificationPatterns);
    
    [sortedCounts, sortIdx] = sort([counts{:}], 'descend');
    sortedPatterns = patterns(sortIdx);
    
    fprintf('  Misclassification patterns in sample:\n');
    for i = 1:length(sortedPatterns)
        fprintf('    %s: %d time(s)\n', sortedPatterns{i}, sortedCounts(i));
    end
end

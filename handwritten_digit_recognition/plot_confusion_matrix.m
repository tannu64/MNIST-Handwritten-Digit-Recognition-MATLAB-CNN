function confMat = plot_confusion_matrix(trueLabels, predictedLabels, titleStr)
%PLOT_CONFUSION_MATRIX Create and display confusion matrix for digit classification
%
% Syntax:
%   confMat = plot_confusion_matrix(trueLabels, predictedLabels, titleStr)
%
% Inputs:
%   trueLabels      - Categorical array of true class labels
%   predictedLabels - Categorical array of predicted class labels
%   titleStr        - String for plot title (optional)
%
% Outputs:
%   confMat - Confusion matrix (10x10 for digits 0-9)
%
% Description:
%   Creates a comprehensive confusion matrix visualization showing:
%   - True vs predicted classifications for each digit
%   - Per-class accuracy and error rates
%   - Color-coded heatmap for easy interpretation
%   - Detailed statistics and annotations

    if nargin < 3
        titleStr = 'Confusion Matrix';
    end
    
    % Validate inputs
    if length(trueLabels) ~= length(predictedLabels)
        error('True and predicted labels must have the same length');
    end
    
    % Convert to categorical if needed
    if ~iscategorical(trueLabels)
        trueLabels = categorical(trueLabels);
    end
    if ~iscategorical(predictedLabels)
        predictedLabels = categorical(predictedLabels);
    end
    
    % Get all possible classes (should be 0-9 for digits)
    allClasses = categories(trueLabels);
    expectedClasses = categorical(0:9);
    
    % Ensure we have all digit classes, add missing ones if necessary
    missingClasses = setdiff(expectedClasses, categorical(allClasses));
    if ~isempty(missingClasses)
        fprintf('Warning: Some digit classes missing from labels\n');
        allClasses = cellstr(string(0:9));
    else
        allClasses = cellstr(string(allClasses));
    end
    
    % Sort classes numerically
    allClasses = sort(str2double(allClasses));
    allClasses = cellstr(string(allClasses));
    
    fprintf('Creating confusion matrix...\n');
    
    % Create confusion matrix
    confMat = confusionmat(trueLabels, predictedLabels);
    
    % Ensure confusion matrix is 10x10 (for digits 0-9)
    if size(confMat, 1) < 10 || size(confMat, 2) < 10
        % Pad with zeros if some classes are missing
        newConfMat = zeros(10, 10);
        existingClasses = str2double(allClasses);
        for i = 1:length(existingClasses)
            for j = 1:length(existingClasses)
                newConfMat(existingClasses(i)+1, existingClasses(j)+1) = confMat(i, j);
            end
        end
        confMat = newConfMat;
        allClasses = cellstr(string(0:9));
    end
    
    % Create figure for confusion matrix
    figure('Name', titleStr, 'Position', [100, 100, 800, 700]);
    
    % Create heatmap
    h = heatmap(allClasses, allClasses, confMat);
    h.Title = titleStr;
    h.XLabel = 'Predicted Class';
    h.YLabel = 'True Class';
    h.ColorbarVisible = 'on';
    h.GridVisible = 'off';
    
    % Customize appearance
    h.FontSize = 12;
    h.CellLabelFormat = '%d';
    
    % Use a colormap that shows errors more prominently
    try
        colormap(h, 'Blues');
    catch
        % Fallback for older MATLAB versions
        colormap(h, 'default');
    end
    
    % Calculate and display per-class statistics
    numClasses = size(confMat, 1);
    classAccuracies = zeros(numClasses, 1);
    classPrecisions = zeros(numClasses, 1);
    classRecalls = zeros(numClasses, 1);
    classF1Scores = zeros(numClasses, 1);
    
    fprintf('\nPer-Class Performance:\n');
    fprintf('Class | Accuracy | Precision | Recall | F1-Score | Support\n');
    fprintf('------|----------|-----------|--------|----------|--------\n');
    
    for i = 1:numClasses
        % True positives, false positives, false negatives
        tp = confMat(i, i);
        fp = sum(confMat(:, i)) - tp;
        fn = sum(confMat(i, :)) - tp;
        tn = sum(confMat(:)) - tp - fp - fn;
        
        % Calculate metrics
        classAccuracies(i) = (tp + tn) / sum(confMat(:));
        classPrecisions(i) = tp / (tp + fp + eps);  % eps to avoid division by zero
        classRecalls(i) = tp / (tp + fn + eps);
        classF1Scores(i) = 2 * (classPrecisions(i) * classRecalls(i)) / ...
                          (classPrecisions(i) + classRecalls(i) + eps);
        
        support = sum(confMat(i, :));
        
        fprintf('  %s   |  %.4f  |   %.4f  | %.4f |  %.4f  |  %4d\n', ...
            allClasses{i}, classAccuracies(i), classPrecisions(i), ...
            classRecalls(i), classF1Scores(i), support);
    end
    
    % Calculate overall metrics
    overallAccuracy = trace(confMat) / sum(confMat(:));
    avgPrecision = mean(classPrecisions);
    avgRecall = mean(classRecalls);
    avgF1Score = mean(classF1Scores);
    
    fprintf('\nOverall Performance:\n');
    fprintf('  Overall Accuracy: %.4f (%.2f%%)\n', overallAccuracy, overallAccuracy * 100);
    fprintf('  Average Precision: %.4f\n', avgPrecision);
    fprintf('  Average Recall: %.4f\n', avgRecall);
    fprintf('  Average F1-Score: %.4f\n', avgF1Score);
    
    % Add text annotations showing common misclassifications
    add_misclassification_analysis(confMat, allClasses);
    
    % Create additional visualization for per-class performance
    create_performance_bar_chart(classAccuracies, classPrecisions, classRecalls, classF1Scores, allClasses);
    
    fprintf('Confusion matrix visualization completed.\n');
end

function add_misclassification_analysis(confMat, classNames)
%ADD_MISCLASSIFICATION_ANALYSIS Identify and report common misclassifications
%
% Analyzes the confusion matrix to find the most common misclassification patterns

    fprintf('\nMisclassification Analysis:\n');
    
    % Find the most common misclassifications (off-diagonal elements)
    [numClasses, ~] = size(confMat);
    misclassifications = [];
    
    for i = 1:numClasses
        for j = 1:numClasses
            if i ~= j && confMat(i, j) > 0
                misclassifications = [misclassifications; confMat(i, j), i, j];
            end
        end
    end
    
    if ~isempty(misclassifications)
        % Sort by frequency
        [~, sortIdx] = sort(misclassifications(:, 1), 'descend');
        misclassifications = misclassifications(sortIdx, :);
        
        % Display top 5 misclassifications
        fprintf('Top misclassifications:\n');
        numToShow = min(5, size(misclassifications, 1));
        
        for i = 1:numToShow
            count = misclassifications(i, 1);
            trueClass = misclassifications(i, 2);
            predClass = misclassifications(i, 3);
            
            percentage = (count / sum(confMat(trueClass, :))) * 100;
            
            fprintf('  %s misclassified as %s: %d times (%.1f%% of true %s)\n', ...
                classNames{trueClass}, classNames{predClass}, count, percentage, classNames{trueClass});
        end
    else
        fprintf('  No misclassifications found (perfect classification!)\n');
    end
end

function create_performance_bar_chart(accuracies, precisions, recalls, f1scores, classNames)
%CREATE_PERFORMANCE_BAR_CHART Create bar chart showing per-class performance metrics
%
% Creates a grouped bar chart showing accuracy, precision, recall, and F1-score for each class

    figure('Name', 'Per-Class Performance Metrics', 'Position', [200, 150, 1000, 600]);
    
    % Prepare data for grouped bar chart
    performanceData = [accuracies, precisions, recalls, f1scores];
    
    % Create grouped bar chart
    bar(performanceData);
    
    % Customize chart
    title('Per-Class Performance Metrics');
    xlabel('Digit Class');
    ylabel('Score');
    legend({'Accuracy', 'Precision', 'Recall', 'F1-Score'}, 'Location', 'best');
    
    % Set x-axis labels
    set(gca, 'XTickLabel', classNames);
    
    % Add grid for better readability
    grid on;
    
    % Set y-axis limits
    ylim([0, 1.1]);
    
    % Add value annotations on bars
    for i = 1:length(classNames)
        for j = 1:4
            if performanceData(i, j) > 0.05  % Only show labels for values > 5%
                text(i + (j-2.5)*0.2, performanceData(i, j) + 0.02, ...
                    sprintf('%.3f', performanceData(i, j)), ...
                    'HorizontalAlignment', 'center', 'FontSize', 8);
            end
        end
    end
    
    % Improve appearance
    set(gca, 'FontSize', 12);
    box on;
end

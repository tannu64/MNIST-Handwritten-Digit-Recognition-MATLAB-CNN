function accuracy = calculate_accuracy(predictedLabels, trueLabels)
%CALCULATE_ACCURACY Calculate classification accuracy for digit recognition
%
% Syntax:
%   accuracy = calculate_accuracy(predictedLabels, trueLabels)
%
% Inputs:
%   predictedLabels - Categorical array of predicted class labels
%   trueLabels      - Categorical array of true class labels
%
% Outputs:
%   accuracy - Scalar value between 0 and 1 representing classification accuracy
%
% Description:
%   Calculates the overall classification accuracy by comparing predicted
%   labels with true labels. Handles both categorical and numeric label formats.
%
% Example:
%   predicted = categorical([1, 2, 3, 1, 2]);
%   true = categorical([1, 2, 2, 1, 2]);
%   acc = calculate_accuracy(predicted, true);  % Returns 0.8 (80%)

    % Validate inputs
    if length(predictedLabels) ~= length(trueLabels)
        error('Predicted and true labels must have the same length');
    end
    
    if isempty(predictedLabels) || isempty(trueLabels)
        error('Labels cannot be empty');
    end
    
    % Convert to categorical if needed
    if ~iscategorical(predictedLabels)
        predictedLabels = categorical(predictedLabels);
    end
    
    if ~iscategorical(trueLabels)
        trueLabels = categorical(trueLabels);
    end
    
    % Calculate accuracy
    correctPredictions = (predictedLabels == trueLabels);
    numCorrect = sum(correctPredictions);
    totalPredictions = length(trueLabels);
    
    accuracy = numCorrect / totalPredictions;
    
    % Display detailed accuracy information
    fprintf('Accuracy Calculation:\n');
    fprintf('  Correct predictions: %d\n', numCorrect);
    fprintf('  Total predictions: %d\n', totalPredictions);
    fprintf('  Accuracy: %.4f (%.2f%%)\n', accuracy, accuracy * 100);
end

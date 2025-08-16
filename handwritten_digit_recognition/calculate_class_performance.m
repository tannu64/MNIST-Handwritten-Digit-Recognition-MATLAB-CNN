function classPerformance = calculate_class_performance(trueLabels, predictedLabels)
%CALCULATE_CLASS_PERFORMANCE Calculate per-class performance metrics
%
% Syntax:
%   classPerformance = calculate_class_performance(trueLabels, predictedLabels)
%
% Inputs:
%   trueLabels      - Categorical array of true class labels
%   predictedLabels - Categorical array of predicted class labels
%
% Outputs:
%   classPerformance - Array of per-class accuracy values (length 10 for digits 0-9)
%
% Description:
%   Calculates accuracy for each digit class (0-9) individually.
%   Returns an array where index i+1 corresponds to digit i's accuracy.

    % Initialize performance array for digits 0-9
    classPerformance = zeros(10, 1);
    
    % Convert labels to categorical if needed
    if ~iscategorical(trueLabels)
        trueLabels = categorical(trueLabels);
    end
    if ~iscategorical(predictedLabels)
        predictedLabels = categorical(predictedLabels);
    end
    
    fprintf('Calculating per-class performance...\n');
    
    % Calculate accuracy for each digit (0-9)
    for digit = 0:9
        % Find indices where true label is current digit
        digitIndices = (trueLabels == categorical(digit));
        
        if sum(digitIndices) > 0
            % Calculate accuracy for this digit
            correctPredictions = (predictedLabels(digitIndices) == categorical(digit));
            classPerformance(digit + 1) = sum(correctPredictions) / sum(digitIndices);
        else
            % No samples for this digit
            classPerformance(digit + 1) = NaN;
            fprintf('  Warning: No samples found for digit %d\n', digit);
        end
    end
    
    % Display results
    fprintf('\nPer-class accuracy:\n');
    for digit = 0:9
        if ~isnan(classPerformance(digit + 1))
            fprintf('  Digit %d: %.4f (%.2f%%)\n', digit, ...
                classPerformance(digit + 1), classPerformance(digit + 1) * 100);
        else
            fprintf('  Digit %d: No samples\n', digit);
        end
    end
end

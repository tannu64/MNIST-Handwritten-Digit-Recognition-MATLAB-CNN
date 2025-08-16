function [bestClass, worstClass] = find_best_worst_classes(classPerformance)
%FIND_BEST_WORST_CLASSES Find the best and worst performing digit classes
%
% Syntax:
%   [bestClass, worstClass] = find_best_worst_classes(classPerformance)
%
% Inputs:
%   classPerformance - Array of per-class accuracy values (length 10 for digits 0-9)
%
% Outputs:
%   bestClass  - Digit (0-9) with highest accuracy
%   worstClass - Digit (0-9) with lowest accuracy
%
% Description:
%   Identifies the best and worst performing digit classes based on
%   their individual accuracy scores. Ignores NaN values.

    % Validate input
    if length(classPerformance) ~= 10
        error('Class performance array must have exactly 10 elements (digits 0-9)');
    end
    
    % Handle NaN values
    validIndices = ~isnan(classPerformance);
    
    if ~any(validIndices)
        error('No valid performance data available');
    end
    
    % Find best performing class
    [~, bestIdx] = max(classPerformance);
    bestClass = bestIdx - 1;  % Convert to digit (0-9)
    
    % Find worst performing class
    [~, worstIdx] = min(classPerformance);
    worstClass = worstIdx - 1;  % Convert to digit (0-9)
    
    % Display results
    fprintf('Performance extremes:\n');
    fprintf('  Best performing digit: %d (%.2f%% accuracy)\n', ...
        bestClass, classPerformance(bestIdx) * 100);
    fprintf('  Worst performing digit: %d (%.2f%% accuracy)\n', ...
        worstClass, classPerformance(worstIdx) * 100);
end

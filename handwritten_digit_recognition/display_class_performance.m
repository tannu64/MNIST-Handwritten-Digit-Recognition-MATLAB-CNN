function display_class_performance(classPerformance)
%DISPLAY_CLASS_PERFORMANCE Display detailed per-class performance analysis
%
% Syntax:
%   display_class_performance(classPerformance)
%
% Inputs:
%   classPerformance - Array of per-class accuracy values (length 10 for digits 0-9)
%
% Description:
%   Creates comprehensive visualizations and analysis of per-class performance
%   including bar charts, statistics, and performance insights.

    fprintf('\n=== DETAILED CLASS PERFORMANCE ANALYSIS ===\n');
    
    % Validate input
    if length(classPerformance) ~= 10
        error('Class performance array must have exactly 10 elements (digits 0-9)');
    end
    
    % Remove NaN values for statistics calculation
    validPerformance = classPerformance(~isnan(classPerformance));
    
    if isempty(validPerformance)
        fprintf('No valid performance data available.\n');
        return;
    end
    
    % Calculate summary statistics
    meanAccuracy = mean(validPerformance);
    stdAccuracy = std(validPerformance);
    minAccuracy = min(validPerformance);
    maxAccuracy = max(validPerformance);
    
    % Find best and worst performing classes
    [~, bestClassIdx] = max(classPerformance);
    [~, worstClassIdx] = min(classPerformance);
    
    % Display summary statistics
    fprintf('\nSummary Statistics:\n');
    fprintf('  Mean Accuracy: %.4f (%.2f%%)\n', meanAccuracy, meanAccuracy * 100);
    fprintf('  Standard Deviation: %.4f\n', stdAccuracy);
    fprintf('  Min Accuracy: %.4f (%.2f%%) - Digit %d\n', ...
        minAccuracy, minAccuracy * 100, worstClassIdx - 1);
    fprintf('  Max Accuracy: %.4f (%.2f%%) - Digit %d\n', ...
        maxAccuracy, maxAccuracy * 100, bestClassIdx - 1);
    fprintf('  Accuracy Range: %.4f\n', maxAccuracy - minAccuracy);
    
    % Performance categorization
    highPerformers = find(classPerformance >= 0.95);  % >= 95%
    goodPerformers = find(classPerformance >= 0.90 & classPerformance < 0.95);  % 90-95%
    averagePerformers = find(classPerformance >= 0.80 & classPerformance < 0.90);  % 80-90%
    poorPerformers = find(classPerformance < 0.80);  % < 80%
    
    fprintf('\nPerformance Categories:\n');
    
    if ~isempty(highPerformers)
        fprintf('  High Performers (≥95%%): Digits %s\n', ...
            sprintf('%d ', highPerformers - 1));
    end
    
    if ~isempty(goodPerformers)
        fprintf('  Good Performers (90-95%%): Digits %s\n', ...
            sprintf('%d ', goodPerformers - 1));
    end
    
    if ~isempty(averagePerformers)
        fprintf('  Average Performers (80-90%%): Digits %s\n', ...
            sprintf('%d ', averagePerformers - 1));
    end
    
    if ~isempty(poorPerformers)
        fprintf('  Poor Performers (<80%%): Digits %s\n', ...
            sprintf('%d ', poorPerformers - 1));
        fprintf('    → Consider additional training or data augmentation for these digits\n');
    end
    
    % Create visualization
    create_class_performance_visualization(classPerformance);
    
    % Provide recommendations
    provide_performance_recommendations(classPerformance, meanAccuracy, stdAccuracy);
    
    fprintf('\n=== END CLASS PERFORMANCE ANALYSIS ===\n');
end

function create_class_performance_visualization(classPerformance)
%CREATE_CLASS_PERFORMANCE_VISUALIZATION Create bar chart of per-class accuracy
%
% Creates a detailed bar chart showing accuracy for each digit class

    figure('Name', 'Per-Class Accuracy', 'Position', [300, 200, 900, 600]);
    
    % Create bar chart
    digits = 0:9;
    validData = ~isnan(classPerformance);
    
    % Create bar plot
    bars = bar(digits, classPerformance * 100, 'FaceColor', 'flat');
    
    % Color bars based on performance level
    for i = 1:length(classPerformance)
        if isnan(classPerformance(i))
            bars.CData(i, :) = [0.7, 0.7, 0.7];  % Gray for missing data
        elseif classPerformance(i) >= 0.95
            bars.CData(i, :) = [0, 0.8, 0];  % Green for high performance
        elseif classPerformance(i) >= 0.90
            bars.CData(i, :) = [0.5, 0.8, 0];  % Light green for good performance
        elseif classPerformance(i) >= 0.80
            bars.CData(i, :) = [1, 0.8, 0];  % Orange for average performance
        else
            bars.CData(i, :) = [1, 0.2, 0];  % Red for poor performance
        end
    end
    
    % Customize chart
    title('Classification Accuracy by Digit Class', 'FontSize', 16, 'FontWeight', 'bold');
    xlabel('Digit Class', 'FontSize', 14);
    ylabel('Accuracy (%)', 'FontSize', 14);
    
    % Set axis properties
    set(gca, 'XTick', digits);
    set(gca, 'XTickLabel', string(digits));
    ylim([0, 105]);
    grid on;
    grid minor;
    
    % Add value labels on bars
    for i = 1:length(digits)
        if ~isnan(classPerformance(i))
            text(digits(i), classPerformance(i) * 100 + 2, ...
                sprintf('%.1f%%', classPerformance(i) * 100), ...
                'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);
        else
            text(digits(i), 5, 'N/A', ...
                'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);
        end
    end
    
    % Add horizontal reference lines
    meanAccuracy = mean(classPerformance(~isnan(classPerformance))) * 100;
    line([0, 9], [meanAccuracy, meanAccuracy], 'Color', 'black', 'LineStyle', '--', 'LineWidth', 2);
    text(8.5, meanAccuracy + 3, sprintf('Mean: %.1f%%', meanAccuracy), ...
        'FontSize', 10, 'FontWeight', 'bold');
    
    % Add 90% reference line
    line([0, 9], [90, 90], 'Color', 'red', 'LineStyle', ':', 'LineWidth', 1.5);
    text(8.5, 92, '90% Target', 'FontSize', 9, 'Color', 'red');
    
    % Add legend
    legendElements = {
        'High (≥95%)', [0, 0.8, 0];
        'Good (90-95%)', [0.5, 0.8, 0];
        'Average (80-90%)', [1, 0.8, 0];
        'Poor (<80%)', [1, 0.2, 0];
        'No Data', [0.7, 0.7, 0.7]
    };
    
    % Create simple text legend instead of using rectangle objects
    legendText = {};
    
    for i = 1:size(legendElements, 1)
        % Check if this performance level exists in the data
        switch i
            case 1  % High
                exists = any(classPerformance >= 0.95 & ~isnan(classPerformance));
            case 2  % Good
                exists = any(classPerformance >= 0.90 & classPerformance < 0.95);
            case 3  % Average
                exists = any(classPerformance >= 0.80 & classPerformance < 0.90);
            case 4  % Poor
                exists = any(classPerformance < 0.80 & ~isnan(classPerformance));
            case 5  % No Data
                exists = any(isnan(classPerformance));
        end
        
        if exists
            legendText{end+1} = legendElements{i, 1};
        end
    end
    
    % Add text annotation instead of legend
    if ~isempty(legendText)
        annotation('textbox', [0.02, 0.85, 0.3, 0.1], 'String', ...
            ['Performance Levels:' newline strjoin(legendText, newline)], ...
            'FontSize', 9, 'BackgroundColor', 'white', 'EdgeColor', 'black');
    end
    
    % Improve overall appearance
    set(gca, 'FontSize', 12);
    box on;
end

function provide_performance_recommendations(classPerformance, meanAccuracy, stdAccuracy)
%PROVIDE_PERFORMANCE_RECOMMENDATIONS Provide actionable recommendations
%
% Analyzes performance patterns and suggests improvements

    fprintf('\nPerformance Recommendations:\n');
    
    % Overall performance assessment
    if meanAccuracy >= 0.95
        fprintf('  ✓ Excellent overall performance (%.2f%% mean accuracy)\n', meanAccuracy * 100);
    elseif meanAccuracy >= 0.90
        fprintf('  ✓ Good overall performance (%.2f%% mean accuracy)\n', meanAccuracy * 100);
    elseif meanAccuracy >= 0.80
        fprintf('  ⚠ Acceptable performance (%.2f%% mean accuracy)\n', meanAccuracy * 100);
        fprintf('    → Consider model improvements or more training data\n');
    else
        fprintf('  ✗ Poor overall performance (%.2f%% mean accuracy)\n', meanAccuracy * 100);
        fprintf('    → Significant model improvements needed\n');
    end
    
    % Consistency assessment
    if stdAccuracy <= 0.05
        fprintf('  ✓ Consistent performance across classes (σ = %.3f)\n', stdAccuracy);
    elseif stdAccuracy <= 0.10
        fprintf('  ⚠ Moderate performance variation (σ = %.3f)\n', stdAccuracy);
        fprintf('    → Some classes may need attention\n');
    else
        fprintf('  ✗ High performance variation (σ = %.3f)\n', stdAccuracy);
        fprintf('    → Class imbalance or model bias detected\n');
    end
    
    % Specific recommendations for poor performers
    poorPerformers = find(classPerformance < 0.80 & ~isnan(classPerformance));
    if ~isempty(poorPerformers)
        fprintf('\nSpecific Recommendations for Poor Performers:\n');
        for i = 1:length(poorPerformers)
            digit = poorPerformers(i) - 1;
            accuracy = classPerformance(poorPerformers(i));
            fprintf('  Digit %d (%.1f%% accuracy):\n', digit, accuracy * 100);
            
            % Digit-specific recommendations
            switch digit
                case {6, 9}
                    fprintf('    → These digits are often confused with each other\n');
                    fprintf('    → Consider data augmentation with rotation variations\n');
                case {3, 8}
                    fprintf('    → Complex shapes that may need more training data\n');
                    fprintf('    → Consider increasing model capacity\n');
                case {4, 7}
                    fprintf('    → Digits with diagonal strokes can be challenging\n');
                    fprintf('    → Ensure training data includes various writing styles\n');
                case {1, 7}
                    fprintf('    → Simple shapes that might be under-represented\n');
                    fprintf('    → Check for class imbalance in training data\n');
                otherwise
                    fprintf('    → Consider collecting more training examples\n');
                    fprintf('    → Review data quality and preprocessing\n');
            end
        end
    end
    
    % General improvement suggestions
    if meanAccuracy < 0.95
        fprintf('\nGeneral Improvement Suggestions:\n');
        fprintf('  1. Increase training epochs or adjust learning rate\n');
        fprintf('  2. Add data augmentation (rotation, translation, scaling)\n');
        fprintf('  3. Try different network architectures\n');
        fprintf('  4. Collect more training data for poorly performing classes\n');
        fprintf('  5. Apply class balancing techniques\n');
        fprintf('  6. Use ensemble methods for final predictions\n');
    end
end

function plot_training_progress(trainingHistory)
%PLOT_TRAINING_PROGRESS Plot training and validation progress curves
%
% Syntax:
%   plot_training_progress(trainingHistory)
%
% Inputs:
%   trainingHistory - Training history structure from trainNetwork
%
% Description:
%   Creates comprehensive plots showing training progress including:
%   - Training and validation loss over epochs
%   - Training and validation accuracy over epochs
%   - Learning rate schedule
%   - Training time per epoch

    if isempty(trainingHistory)
        fprintf('No training history available for plotting.\n');
        return;
    end
    
    fprintf('Creating training progress visualization...\n');
    
    % Extract data from training history
    try
        % Get available fields
        fields = fieldnames(trainingHistory);
        
        % Common field names in training history
        epochField = '';
        lossField = '';
        accuracyField = '';
        valLossField = '';
        valAccuracyField = '';
        learnRateField = '';
        
        % Find appropriate field names
        for i = 1:length(fields)
            fieldName = lower(fields{i});
            if contains(fieldName, 'epoch')
                epochField = fields{i};
            elseif contains(fieldName, 'loss') && ~contains(fieldName, 'val')
                lossField = fields{i};
            elseif contains(fieldName, 'accuracy') && ~contains(fieldName, 'val')
                accuracyField = fields{i};
            elseif contains(fieldName, 'val') && contains(fieldName, 'loss')
                valLossField = fields{i};
            elseif contains(fieldName, 'val') && contains(fieldName, 'accuracy')
                valAccuracyField = fields{i};
            elseif contains(fieldName, 'learn') || contains(fieldName, 'rate')
                learnRateField = fields{i};
            end
        end
        
        % Create figure with subplots
        figure('Name', 'Training Progress', 'Position', [100, 50, 1200, 900]);
        
        % Plot 1: Loss curves
        subplot(2, 2, 1);
        hold on;
        
        if ~isempty(lossField) && isfield(trainingHistory, lossField)
            epochs = 1:length(trainingHistory.(lossField));
            plot(epochs, trainingHistory.(lossField), 'b-', 'LineWidth', 2, 'DisplayName', 'Training Loss');
        end
        
        if ~isempty(valLossField) && isfield(trainingHistory, valLossField)
            valEpochs = 1:length(trainingHistory.(valLossField));
            plot(valEpochs, trainingHistory.(valLossField), 'r-', 'LineWidth', 2, 'DisplayName', 'Validation Loss');
        end
        
        title('Training and Validation Loss', 'FontSize', 14, 'FontWeight', 'bold');
        xlabel('Epoch');
        ylabel('Loss');
        legend('show');
        grid on;
        hold off;
        
        % Plot 2: Accuracy curves
        subplot(2, 2, 2);
        hold on;
        
        if ~isempty(accuracyField) && isfield(trainingHistory, accuracyField)
            epochs = 1:length(trainingHistory.(accuracyField));
            plot(epochs, trainingHistory.(accuracyField), 'b-', 'LineWidth', 2, 'DisplayName', 'Training Accuracy');
        end
        
        if ~isempty(valAccuracyField) && isfield(trainingHistory, valAccuracyField)
            valEpochs = 1:length(trainingHistory.(valAccuracyField));
            plot(valEpochs, trainingHistory.(valAccuracyField), 'r-', 'LineWidth', 2, 'DisplayName', 'Validation Accuracy');
        end
        
        title('Training and Validation Accuracy', 'FontSize', 14, 'FontWeight', 'bold');
        xlabel('Epoch');
        ylabel('Accuracy');
        legend('show');
        grid on;
        ylim([0, 1]);
        hold off;
        
        % Plot 3: Learning rate schedule
        subplot(2, 2, 3);
        if ~isempty(learnRateField) && isfield(trainingHistory, learnRateField)
            epochs = 1:length(trainingHistory.(learnRateField));
            semilogy(epochs, trainingHistory.(learnRateField), 'g-', 'LineWidth', 2);
            title('Learning Rate Schedule', 'FontSize', 14, 'FontWeight', 'bold');
            xlabel('Epoch');
            ylabel('Learning Rate (log scale)');
            grid on;
        else
            % Show default learning rate schedule if data not available
            epochs = 1:20;  % Assume 20 epochs
            lr = create_default_lr_schedule(epochs);
            semilogy(epochs, lr, 'g-', 'LineWidth', 2);
            title('Learning Rate Schedule (Estimated)', 'FontSize', 14, 'FontWeight', 'bold');
            xlabel('Epoch');
            ylabel('Learning Rate (log scale)');
            grid on;
        end
        
        % Plot 4: Training summary and statistics
        subplot(2, 2, 4);
        axis off;
        
        % Display training summary
        summaryText = create_training_summary(trainingHistory, lossField, accuracyField, valLossField, valAccuracyField);
        text(0.1, 0.9, summaryText, 'FontSize', 10, 'VerticalAlignment', 'top', 'FontName', 'FixedWidth');
        title('Training Summary', 'FontSize', 14, 'FontWeight', 'bold');
        
        % Adjust layout
        sgtitle('CNN Training Progress Analysis', 'FontSize', 16, 'FontWeight', 'bold');
        
    catch ME
        warning('Error creating training progress plot: %s', ME.message);
        fprintf('Available fields in training history: %s\n', strjoin(fields, ', '));
        create_simple_progress_plot(trainingHistory);
    end
    
    fprintf('Training progress visualization completed.\n');
end

function lr = create_default_lr_schedule(epochs)
%CREATE_DEFAULT_LR_SCHEDULE Create default learning rate schedule
%
% Creates a typical learning rate schedule for visualization when actual data is not available

    initialLR = 0.001;
    lr = ones(size(epochs)) * initialLR;
    
    % Apply step decay (drops by 0.5 every 5 epochs)
    for i = 1:length(epochs)
        epoch = epochs(i);
        if epoch > 10
            lr(i) = initialLR * 0.5;
        end
        if epoch > 15
            lr(i) = initialLR * 0.25;
        end
    end
end

function summaryText = create_training_summary(trainingHistory, lossField, accuracyField, valLossField, valAccuracyField)
%CREATE_TRAINING_SUMMARY Create text summary of training results
%
% Creates formatted text summary of key training metrics

    summaryLines = {};
    
    summaryLines{end+1} = 'TRAINING SUMMARY';
    summaryLines{end+1} = '================';
    summaryLines{end+1} = '';
    
    % Training metrics
    if ~isempty(lossField) && isfield(trainingHistory, lossField)
        finalLoss = trainingHistory.(lossField)(end);
        summaryLines{end+1} = sprintf('Final Training Loss: %.4f', finalLoss);
    end
    
    if ~isempty(accuracyField) && isfield(trainingHistory, accuracyField)
        finalAccuracy = trainingHistory.(accuracyField)(end);
        summaryLines{end+1} = sprintf('Final Training Accuracy: %.2f%%', finalAccuracy * 100);
    end
    
    summaryLines{end+1} = '';
    
    % Validation metrics
    if ~isempty(valLossField) && isfield(trainingHistory, valLossField)
        finalValLoss = trainingHistory.(valLossField)(end);
        summaryLines{end+1} = sprintf('Final Validation Loss: %.4f', finalValLoss);
    end
    
    if ~isempty(valAccuracyField) && isfield(trainingHistory, valAccuracyField)
        finalValAccuracy = trainingHistory.(valAccuracyField)(end);
        summaryLines{end+1} = sprintf('Final Validation Accuracy: %.2f%%', finalValAccuracy * 100);
    end
    
    summaryLines{end+1} = '';
    
    % Training analysis
    if ~isempty(accuracyField) && isfield(trainingHistory, accuracyField)
        accuracy = trainingHistory.(accuracyField);
        maxAccuracy = max(accuracy);
        [~, maxEpoch] = max(accuracy);
        summaryLines{end+1} = sprintf('Best Training Accuracy: %.2f%%', maxAccuracy * 100);
        summaryLines{end+1} = sprintf('Achieved at Epoch: %d', maxEpoch);
    end
    
    summaryLines{end+1} = '';
    
    % Overfitting analysis
    if ~isempty(lossField) && ~isempty(valLossField) && ...
       isfield(trainingHistory, lossField) && isfield(trainingHistory, valLossField)
        
        trainLoss = trainingHistory.(lossField)(end);
        valLoss = trainingHistory.(valLossField)(end);
        
        if valLoss > trainLoss * 1.2
            summaryLines{end+1} = 'Overfitting Detected!';
            summaryLines{end+1} = 'Consider:';
            summaryLines{end+1} = '- Reducing model complexity';
            summaryLines{end+1} = '- Adding regularization';
            summaryLines{end+1} = '- Early stopping';
        else
            summaryLines{end+1} = 'No significant overfitting';
        end
    end
    
    % Join all lines
    summaryText = strjoin(summaryLines, '\n');
end

function create_simple_progress_plot(trainingHistory)
%CREATE_SIMPLE_PROGRESS_PLOT Create simplified progress plot when detailed data unavailable
%
% Fallback visualization when standard field names are not found

    figure('Name', 'Training Progress (Simplified)', 'Position', [150, 100, 800, 600]);
    
    % Try to plot any available numeric data
    fields = fieldnames(trainingHistory);
    numericFields = {};
    
    for i = 1:length(fields)
        if isnumeric(trainingHistory.(fields{i})) && ~isscalar(trainingHistory.(fields{i}))
            numericFields{end+1} = fields{i};
        end
    end
    
    if isempty(numericFields)
        text(0.5, 0.5, 'No numeric time series data found in training history', ...
             'HorizontalAlignment', 'center', 'FontSize', 14);
        axis off;
        return;
    end
    
    % Plot available numeric fields
    numPlots = min(4, length(numericFields));
    
    for i = 1:numPlots
        subplot(2, 2, i);
        data = trainingHistory.(numericFields{i});
        plot(1:length(data), data, 'LineWidth', 2);
        title(numericFields{i}, 'Interpreter', 'none');
        xlabel('Index');
        ylabel('Value');
        grid on;
    end
    
    sgtitle('Available Training Data', 'FontSize', 16, 'FontWeight', 'bold');
end

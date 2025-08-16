function options = create_training_options(useGPU)
%CREATE_TRAINING_OPTIONS Define training options for CNN digit recognition
%
% Syntax:
%   options = create_training_options(useGPU)
%
% Inputs:
%   useGPU - Boolean indicating whether to use GPU acceleration
%
% Outputs:
%   options - TrainingOptionsADAM object with optimized settings
%
% Description:
%   Creates training options optimized for handwritten digit recognition.
%   Includes settings for optimizer, learning rate, batch size, epochs,
%   validation, and other training parameters.
%
% Training Configuration:
%   - Optimizer: ADAM (adaptive learning rate)
%   - Initial Learning Rate: 0.001
%   - Learning Rate Schedule: Piecewise (drops at specific epochs)
%   - Max Epochs: 20
%   - Mini Batch Size: 128 (or 64 for CPU)
%   - Validation: 15% of training data
%   - Shuffle: Every epoch
%   - Verbose Training: Enabled with plots
%   - L2 Regularization: 0.0001

    if nargin < 1
        useGPU = false;
    end
    
    fprintf('Configuring training options...\n');
    
    % Determine optimal batch size based on hardware
    if useGPU
        miniBatchSize = 128;  % Larger batch for GPU
        executionEnvironment = 'gpu';
        fprintf('  Hardware: GPU acceleration enabled\n');
    else
        miniBatchSize = 64;   % Smaller batch for CPU
        executionEnvironment = 'cpu';
        fprintf('  Hardware: CPU training (consider using GPU for faster training)\n');
    end
    
    % Define training parameters
    initialLearnRate = 0.001;
    maxEpochs = 20;
    validationFrequency = 50;  % Validate every 50 iterations
    
    % Create learning rate schedule
    % Reduce learning rate at epochs 10 and 15
    learnRateDropPeriod = 5;
    learnRateDropFactor = 0.5;
    
    fprintf('  Training Parameters:\n');
    fprintf('    Initial Learning Rate: %.4f\n', initialLearnRate);
    fprintf('    Max Epochs: %d\n', maxEpochs);
    fprintf('    Mini Batch Size: %d\n', miniBatchSize);
    fprintf('    Execution Environment: %s\n', executionEnvironment);
    fprintf('    Learning Rate Schedule: Drop by %.1f every %d epochs\n', ...
        learnRateDropFactor, learnRateDropPeriod);
    
    % Create training options
    options = trainingOptions('adam', ...
        'InitialLearnRate', initialLearnRate, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', learnRateDropPeriod, ...
        'LearnRateDropFactor', learnRateDropFactor, ...
        'MaxEpochs', maxEpochs, ...
        'MiniBatchSize', miniBatchSize, ...
        'ValidationFrequency', validationFrequency, ...
        'ValidationPatience', 5, ...  % Early stopping patience
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'VerboseFrequency', 10, ...  % Print progress every 10 iterations
        'Plots', 'training-progress', ...
        'ExecutionEnvironment', executionEnvironment, ...
        'L2Regularization', 0.0001, ...  % Weight decay
        'GradientThreshold', 1, ...  % Gradient clipping
        'CheckpointPath', '', ...  % No automatic checkpointing
        'OutputFcn', @(info) training_output_function(info));
    
    fprintf('  Regularization:\n');
    fprintf('    L2 Regularization: %.6f\n', options.L2Regularization);
    fprintf('    Gradient Threshold: %.1f\n', options.GradientThreshold);
    fprintf('    Validation Patience: %d epochs\n', options.ValidationPatience);
    
    fprintf('Training options configured successfully.\n');
end

function stop = training_output_function(info)
%TRAINING_OUTPUT_FUNCTION Custom output function for training monitoring
%
% This function is called during training to provide additional monitoring
% and can implement custom stopping criteria

    stop = false;  % Don't stop training by default
    
    % Display additional information at the end of each epoch
    if strcmp(info.State, 'iteration') && mod(info.Iteration, 50) == 0
        % Display progress every 50 iterations (handle different field names)
        if isfield(info, 'TrainingAccuracy')
            accuracy = info.TrainingAccuracy;
        elseif isfield(info, 'MiniBatchAccuracy')
            accuracy = info.MiniBatchAccuracy;
        else
            accuracy = 0;
        end
        
        if isfield(info, 'TrainingElapsedTime')
            elapsedTime = info.TrainingElapsedTime;
        elseif isfield(info, 'ElapsedTime')
            elapsedTime = info.ElapsedTime;
        else
            elapsedTime = 0;
        end
        
        fprintf('    Iteration %d: Training Accuracy = %.2f%%, Elapsed Time = %.1fs\n', ...
            info.Iteration, accuracy, elapsedTime);
    end
    
    % Check for potential issues during training
    if strcmp(info.State, 'iteration')
        % Check for NaN or Inf in training loss
        if isfield(info, 'TrainingLoss')
            loss = info.TrainingLoss;
        elseif isfield(info, 'MiniBatchLoss')
            loss = info.MiniBatchLoss;
        else
            loss = 0;
        end
        
        if isnan(loss) || isinf(loss)
            warning('Training loss is NaN or Inf at iteration %d. Consider reducing learning rate.', ...
                info.Iteration);
            stop = true;
            return;
        end
        
        % Check for extremely high loss (potential exploding gradients)
        if loss > 100
            warning('Training loss is extremely high (%.2f) at iteration %d. Stopping training.', ...
                loss, info.Iteration);
            stop = true;
            return;
        end
        
        % Check for validation accuracy plateau (if validation is being performed)
        if ~isempty(info.ValidationLoss) && info.Iteration > 200
            % If validation loss hasn't improved for many iterations, consider early stopping
            % This is handled by ValidationPatience parameter, but we can add custom logic here
        end
    end
    
    % Provide epoch-end summary
    if strcmp(info.State, 'epoch')
        fprintf('\n  === Epoch %d Summary ===\n', info.Epoch);
        
        % Handle different field names for accuracy
        if isfield(info, 'TrainingAccuracy')
            fprintf('    Training Accuracy: %.2f%%\n', info.TrainingAccuracy);
        end
        
        % Handle different field names for loss
        if isfield(info, 'TrainingLoss')
            fprintf('    Training Loss: %.4f\n', info.TrainingLoss);
        end
        
        % Validation metrics (if available)
        if isfield(info, 'ValidationAccuracy') && ~isempty(info.ValidationAccuracy)
            fprintf('    Validation Accuracy: %.2f%%\n', info.ValidationAccuracy);
        end
        if isfield(info, 'ValidationLoss') && ~isempty(info.ValidationLoss)
            fprintf('    Validation Loss: %.4f\n', info.ValidationLoss);
        end
        
        % Learning rate
        if isfield(info, 'LearnRate')
            fprintf('    Learning Rate: %.6f\n', info.LearnRate);
        elseif isfield(info, 'BaseLearnRate')
            fprintf('    Learning Rate: %.6f\n', info.BaseLearnRate);
        end
        
        % Elapsed time
        if isfield(info, 'TrainingElapsedTime')
            elapsedTime = info.TrainingElapsedTime;
        elseif isfield(info, 'ElapsedTime')
            elapsedTime = info.ElapsedTime;
        else
            elapsedTime = 0;
        end
        
        if elapsedTime > 0
            fprintf('    Elapsed Time: %.1f seconds\n', elapsedTime);
            
            % Estimate remaining time
            if info.Epoch > 1 && isfield(info, 'MaxEpochs')
                avgTimePerEpoch = elapsedTime / info.Epoch;
                remainingEpochs = info.MaxEpochs - info.Epoch;
                estimatedRemainingTime = avgTimePerEpoch * remainingEpochs;
                fprintf('    Estimated Remaining Time: %.1f minutes\n', estimatedRemainingTime / 60);
            end
        end
        
        fprintf('  =========================\n\n');
    end
    
    % Final training summary
    if strcmp(info.State, 'done')
        fprintf('\n=== TRAINING COMPLETED ===\n');
        
        % Handle elapsed time
        if isfield(info, 'TrainingElapsedTime')
            fprintf('Total Training Time: %.2f minutes\n', info.TrainingElapsedTime / 60);
        elseif isfield(info, 'ElapsedTime')
            fprintf('Total Training Time: %.2f minutes\n', info.ElapsedTime / 60);
        end
        
        % Handle training accuracy
        if isfield(info, 'TrainingAccuracy')
            fprintf('Final Training Accuracy: %.2f%%\n', info.TrainingAccuracy);
        end
        
        % Handle validation accuracy
        if isfield(info, 'ValidationAccuracy') && ~isempty(info.ValidationAccuracy)
            fprintf('Final Validation Accuracy: %.2f%%\n', info.ValidationAccuracy);
        end
        
        fprintf('Total Epochs: %d\n', info.Epoch);
        fprintf('Total Iterations: %d\n', info.Iteration);
        fprintf('=============================\n\n');
    end
end

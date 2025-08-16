function totalParams = calculate_model_parameters(net)
%CALCULATE_MODEL_PARAMETERS Calculate total number of trainable parameters in CNN
%
% Syntax:
%   totalParams = calculate_model_parameters(net)
%
% Inputs:
%   net - Trained CNN network (SeriesNetwork or DAGNetwork)
%
% Outputs:
%   totalParams - Total number of trainable parameters
%
% Description:
%   Calculates the total number of trainable parameters in a CNN model
%   by analyzing weights and biases in all layers.

    fprintf('Calculating model parameters...\n');
    
    totalParams = 0;
    
    try
        % Get network layers
        if isa(net, 'SeriesNetwork')
            layers = net.Layers;
        elseif isa(net, 'DAGNetwork')
            layers = net.Layers;
        else
            % Try to extract layers from the network structure
            if isprop(net, 'Layers')
                layers = net.Layers;
            else
                warning('Cannot extract layers from network. Using approximation.');
                totalParams = estimate_parameters_from_architecture();
                return;
            end
        end
        
        % Count parameters in each layer
        for i = 1:length(layers)
            layer = layers(i);
            layerParams = 0;
            
            % Check layer type and count parameters
            if isa(layer, 'nnet.cnn.layer.Convolution2DLayer')
                % Convolutional layer: weights + biases
                filterSize = layer.FilterSize;
                numFilters = layer.NumFilters;
                numChannels = layer.NumChannels;
                
                % Weights: FilterHeight * FilterWidth * NumChannels * NumFilters
                weightParams = filterSize(1) * filterSize(2) * numChannels * numFilters;
                
                % Biases: NumFilters (if bias is enabled)
                biasParams = numFilters;
                
                layerParams = weightParams + biasParams;
                
                fprintf('  Conv Layer %d: %d weights + %d biases = %d parameters\n', ...
                    i, weightParams, biasParams, layerParams);
                
            elseif isa(layer, 'nnet.cnn.layer.FullyConnectedLayer')
                % Fully connected layer: weights + biases
                inputSize = layer.InputSize;
                outputSize = layer.OutputSize;
                
                % Weights: InputSize * OutputSize
                weightParams = inputSize * outputSize;
                
                % Biases: OutputSize
                biasParams = outputSize;
                
                layerParams = weightParams + biasParams;
                
                fprintf('  FC Layer %d: %d weights + %d biases = %d parameters\n', ...
                    i, weightParams, biasParams, layerParams);
                
            elseif isa(layer, 'nnet.cnn.layer.BatchNormalizationLayer')
                % Batch normalization: scale + offset parameters
                % Number of parameters equals number of channels
                numChannels = layer.NumChannels;
                layerParams = numChannels * 2;  % scale + offset
                
                fprintf('  BN Layer %d: %d parameters (scale + offset)\n', ...
                    i, layerParams);
                
            else
                % Other layers (ReLU, Pooling, etc.) have no trainable parameters
                layerParams = 0;
            end
            
            totalParams = totalParams + layerParams;
        end
        
    catch ME
        warning('Error calculating exact parameters: %s', ME.message);
        fprintf('Using parameter estimation instead...\n');
        totalParams = estimate_parameters_from_architecture();
    end
    
    fprintf('\nTotal trainable parameters: %d\n', totalParams);
    
    % Display parameter breakdown
    display_parameter_breakdown(totalParams);
end

function estimatedParams = estimate_parameters_from_architecture()
%ESTIMATE_PARAMETERS_FROM_ARCHITECTURE Estimate parameters for standard architecture
%
% Provides parameter estimation when exact calculation fails

    fprintf('Estimating parameters for standard CNN architecture...\n');
    
    % Standard architecture parameter estimation:
    % Conv1: 3x3x1x8 + 8 bias = 72 + 8 = 80
    conv1_params = (3 * 3 * 1 * 8) + 8;
    
    % Conv2: 3x3x8x16 + 16 bias = 1152 + 16 = 1168  
    conv2_params = (3 * 3 * 8 * 16) + 16;
    
    % Conv3: 3x3x16x32 + 32 bias = 4608 + 32 = 4640
    conv3_params = (3 * 3 * 16 * 32) + 32;
    
    % FC1: 32x128 + 128 bias = 4096 + 128 = 4224
    fc1_params = (32 * 128) + 128;
    
    % FC2: 128x10 + 10 bias = 1280 + 10 = 1290
    fc2_params = (128 * 10) + 10;
    
    % Batch normalization layers (approximate)
    bn_params = 8 * 2 + 16 * 2 + 32 * 2;  % scale + offset for each conv layer
    
    estimatedParams = conv1_params + conv2_params + conv3_params + ...
                     fc1_params + fc2_params + bn_params;
    
    fprintf('Parameter estimation:\n');
    fprintf('  Conv layers: %d + %d + %d = %d\n', ...
        conv1_params, conv2_params, conv3_params, ...
        conv1_params + conv2_params + conv3_params);
    fprintf('  FC layers: %d + %d = %d\n', ...
        fc1_params, fc2_params, fc1_params + fc2_params);
    fprintf('  BN layers: %d\n', bn_params);
    fprintf('  Total estimated: %d\n', estimatedParams);
end

function display_parameter_breakdown(totalParams)
%DISPLAY_PARAMETER_BREAKDOWN Display parameter count in different units
%
% Shows parameter count in various formats for better understanding

    fprintf('\nParameter Count Breakdown:\n');
    fprintf('  Total parameters: %d\n', totalParams);
    fprintf('  In thousands (K): %.1fK\n', totalParams / 1000);
    fprintf('  In millions (M): %.3fM\n', totalParams / 1000000);
    
    % Memory estimation (assuming 32-bit floats)
    memoryBytes = totalParams * 4;  % 4 bytes per float32
    memoryKB = memoryBytes / 1024;
    memoryMB = memoryKB / 1024;
    
    fprintf('\nMemory Requirements (32-bit floats):\n');
    fprintf('  Parameters only: %.1f KB (%.2f MB)\n', memoryKB, memoryMB);
    
    % Model size category
    if totalParams < 10000
        category = 'Very Small';
    elseif totalParams < 100000
        category = 'Small';
    elseif totalParams < 1000000
        category = 'Medium';
    elseif totalParams < 10000000
        category = 'Large';
    else
        category = 'Very Large';
    end
    
    fprintf('\nModel Size Category: %s\n', category);
    
    % Comparison with common models
    fprintf('\nComparison with Common Models:\n');
    fprintf('  LeNet-5: ~60K parameters\n');
    fprintf('  AlexNet: ~60M parameters\n');
    fprintf('  VGG-16: ~138M parameters\n');
    fprintf('  This model: ~%.1fK parameters\n', totalParams / 1000);
end

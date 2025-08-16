function layers = create_cnn_architecture()
%CREATE_CNN_ARCHITECTURE Define CNN architecture for handwritten digit recognition
%
% Syntax:
%   layers = create_cnn_architecture()
%
% Outputs:
%   layers - Array of CNN layers optimized for digit classification
%
% Description:
%   Creates a Convolutional Neural Network architecture specifically
%   designed for MNIST digit classification. The architecture includes:
%   - Input layer for 28x28x1 grayscale images
%   - Multiple convolutional layers with ReLU activation
%   - Max pooling layers for dimensionality reduction
%   - Fully connected layers for classification
%   - Dropout for regularization
%   - Softmax and classification layers for 10-class output
%
% Architecture Details:
%   Input: 28x28x1 grayscale images
%   Conv1: 8 filters of size 3x3, stride 1, padding 'same'
%   ReLU1: ReLU activation
%   Pool1: 2x2 max pooling, stride 2
%   Conv2: 16 filters of size 3x3, stride 1, padding 'same'
%   ReLU2: ReLU activation
%   Pool2: 2x2 max pooling, stride 2
%   Conv3: 32 filters of size 3x3, stride 1, padding 'same'
%   ReLU3: ReLU activation
%   FC1: Fully connected layer with 128 units
%   ReLU4: ReLU activation
%   Dropout: 50% dropout for regularization
%   FC2: Fully connected layer with 10 units (output)
%   Softmax: Softmax activation for probability distribution
%   Classification: Classification layer for 10 classes

    fprintf('Creating CNN architecture for digit recognition...\n');
    
    % Define the network layers
    layers = [
        % Input layer - accepts 28x28x1 grayscale images
        imageInputLayer([28 28 1], ...
            'Name', 'input', ...
            'Normalization', 'none')  % We handle normalization in preprocessing
        
        % First Convolutional Block
        convolution2dLayer(3, 8, ...
            'Padding', 'same', ...
            'Stride', 1, ...
            'Name', 'conv1', ...
            'WeightsInitializer', 'he')
        
        batchNormalizationLayer('Name', 'bn1')
        
        reluLayer('Name', 'relu1')
        
        maxPooling2dLayer(2, ...
            'Stride', 2, ...
            'Name', 'pool1')
        
        % Second Convolutional Block
        convolution2dLayer(3, 16, ...
            'Padding', 'same', ...
            'Stride', 1, ...
            'Name', 'conv2', ...
            'WeightsInitializer', 'he')
        
        batchNormalizationLayer('Name', 'bn2')
        
        reluLayer('Name', 'relu2')
        
        maxPooling2dLayer(2, ...
            'Stride', 2, ...
            'Name', 'pool2')
        
        % Third Convolutional Block
        convolution2dLayer(3, 32, ...
            'Padding', 'same', ...
            'Stride', 1, ...
            'Name', 'conv3', ...
            'WeightsInitializer', 'he')
        
        batchNormalizationLayer('Name', 'bn3')
        
        reluLayer('Name', 'relu3')
        
        % Average Pooling to reduce spatial dimensions
        averagePooling2dLayer(7, 'Name', 'avgpool')
        
        % Fully Connected Layers
        fullyConnectedLayer(128, ...
            'Name', 'fc1', ...
            'WeightsInitializer', 'he')
        
        reluLayer('Name', 'relu4')
        
        % Dropout for regularization
        dropoutLayer(0.5, 'Name', 'dropout1')
        
        % Output layer - 10 classes for digits 0-9
        fullyConnectedLayer(10, ...
            'Name', 'fc2', ...
            'WeightsInitializer', 'he')
        
        % Softmax and classification layers
        softmaxLayer('Name', 'softmax')
        
        classificationLayer('Name', 'output')
    ];
    
    % Display architecture summary
    fprintf('CNN Architecture Created:\n');
    fprintf('  Total layers: %d\n', length(layers));
    fprintf('  Input size: [28, 28, 1]\n');
    fprintf('  Output classes: 10 (digits 0-9)\n');
    
    % Display layer details
    fprintf('\nLayer Details:\n');
    for i = 1:length(layers)
        layer = layers(i);
        layerType = class(layer);
        layerName = '';
        
        % Extract layer name if available
        if isprop(layer, 'Name') && ~isempty(layer.Name)
            layerName = layer.Name;
        end
        
        % Display layer information based on type
        switch layerType
            case 'nnet.cnn.layer.ImageInputLayer'
                fprintf('  %2d. %-25s - Input: [%d, %d, %d]\n', ...
                    i, sprintf('%s (%s)', layerName, 'ImageInput'), ...
                    layer.InputSize(1), layer.InputSize(2), layer.InputSize(3));
                
            case 'nnet.cnn.layer.Convolution2DLayer'
                fprintf('  %2d. %-25s - Filters: %d, Size: [%d, %d], Stride: [%d, %d]\n', ...
                    i, sprintf('%s (%s)', layerName, 'Conv2D'), ...
                    layer.NumFilters, layer.FilterSize(1), layer.FilterSize(2), ...
                    layer.Stride(1), layer.Stride(2));
                
            case 'nnet.cnn.layer.MaxPooling2DLayer'
                fprintf('  %2d. %-25s - Pool Size: [%d, %d], Stride: [%d, %d]\n', ...
                    i, sprintf('%s (%s)', layerName, 'MaxPool2D'), ...
                    layer.PoolSize(1), layer.PoolSize(2), ...
                    layer.Stride(1), layer.Stride(2));
                
            case 'nnet.cnn.layer.FullyConnectedLayer'
                fprintf('  %2d. %-25s - Output Size: %d\n', ...
                    i, sprintf('%s (%s)', layerName, 'FullyConnected'), ...
                    layer.OutputSize);
                
            case 'nnet.cnn.layer.DropoutLayer'
                fprintf('  %2d. %-25s - Probability: %.1f\n', ...
                    i, sprintf('%s (%s)', layerName, 'Dropout'), ...
                    layer.Probability);
                
            otherwise
                fprintf('  %2d. %-25s - %s\n', ...
                    i, sprintf('%s (%s)', layerName, 'Other'), layerType);
        end
    end
    
    fprintf('\nArchitecture designed for optimal digit recognition performance.\n');
end

# Usage Guide: Handwritten Digit Recognition

This guide provides detailed instructions for using the handwritten digit recognition system.

## 🚀 Getting Started

### 1. Basic Execution

The simplest way to use the system:

```matlab
% Navigate to the project directory
cd('handwritten_digit_recognition')

% Run the main script
main_digit_recognition
```

This will:
- Automatically load data
- Train the CNN model
- Evaluate performance
- Display results
- Save the trained model

### 2. Step-by-Step Execution

For more control, run individual components:

```matlab
%% Step 1: Load and preprocess data
[trainImages, trainLabels] = load_mnist_data('train');
[testImages, testLabels] = load_mnist_data('test');

[trainImages, trainLabels] = preprocess_digit_data(trainImages, trainLabels);
[testImages, testLabels] = preprocess_digit_data(testImages, testLabels);

%% Step 2: Create and train model
layers = create_cnn_architecture();
options = create_training_options(true);  % true for GPU
net = trainNetwork(trainImages, trainLabels, layers, options);

%% Step 3: Evaluate model
predictedLabels = classify(net, testImages);
accuracy = calculate_accuracy(predictedLabels, testLabels);
plot_confusion_matrix(testLabels, predictedLabels);
```

## 📊 Understanding the Results

### Training Output Interpretation

```
Epoch 1/20: Training Accuracy = 85.23%, Validation Accuracy = 87.45%
Epoch 5/20: Training Accuracy = 95.67%, Validation Accuracy = 96.12%
Epoch 10/20: Training Accuracy = 98.34%, Validation Accuracy = 98.01%
Final: Training Accuracy = 99.12%, Validation Accuracy = 98.45%
```

**What to look for:**
- **Convergence**: Accuracy should increase over epochs
- **Overfitting**: Training accuracy much higher than validation
- **Stability**: Consistent improvement without large fluctuations

### Performance Metrics

```
=== PERFORMANCE SUMMARY ===
Overall Test Accuracy: 98.45%
Best performing class: 1 (99.2% accuracy)
Worst performing class: 8 (97.1% accuracy)
```

**Interpretation:**
- **>98%**: Excellent performance
- **95-98%**: Good performance
- **90-95%**: Acceptable performance
- **<90%**: Needs improvement

## 🎯 Customization Options

### 1. Training Parameters

Modify `create_training_options.m`:

```matlab
% For faster training (lower accuracy)
initialLearnRate = 0.01;
maxEpochs = 10;
miniBatchSize = 256;

% For better accuracy (slower training)
initialLearnRate = 0.0001;
maxEpochs = 50;
miniBatchSize = 32;
```

### 2. Architecture Modifications

In `create_cnn_architecture.m`:

```matlab
% Larger model (better accuracy, slower training)
convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv3')
fullyConnectedLayer(256, 'Name', 'fc1')

% Smaller model (faster training, potentially lower accuracy)
convolution2dLayer(3, 4, 'Padding', 'same', 'Name', 'conv1')
convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'conv2')
fullyConnectedLayer(64, 'Name', 'fc1')
```

### 3. Data Augmentation

Enable in `preprocess_digit_data.m`:

```matlab
% Force data augmentation
[processedImages, processedLabels] = apply_data_augmentation(processedImages, processedLabels);
```

## 🧪 Testing Your Model

### 1. Interactive Testing

```matlab
% Load your trained model
load('models/digit_recognition_cnn_YYYYMMDD_HHMMSS.mat', 'net');

% Start interactive testing
interactive_digit_testing(net);
```

**Available test modes:**
1. **Synthetic digits**: Computer-generated test images
2. **Custom images**: Your own digit photos
3. **Noisy digits**: Test robustness with corrupted images
4. **Batch testing**: Multiple images at once
5. **Confidence analysis**: Detailed prediction analysis

### 2. Programmatic Testing

```matlab
% Test single image
testImg = imread('my_digit.png');
processedImg = preprocess_single_image(testImg);
[predictedLabel, scores] = classify(net, processedImg);

fprintf('Predicted digit: %s (%.1f%% confidence)\n', ...
    string(predictedLabel), max(scores) * 100);
```

### 3. Batch Testing

```matlab
% Test multiple images
imageFolder = 'test_images/';
imageFiles = dir(fullfile(imageFolder, '*.png'));

for i = 1:length(imageFiles)
    img = imread(fullfile(imageFolder, imageFiles(i).name));
    processed = preprocess_single_image(img);
    prediction = classify(net, processed);
    
    fprintf('File: %s, Prediction: %s\n', ...
        imageFiles(i).name, string(prediction));
end
```

## 📈 Performance Optimization

### 1. GPU Acceleration

```matlab
% Check GPU availability
if gpuDeviceCount > 0
    gpu = gpuDevice();
    fprintf('Using GPU: %s\n', gpu.Name);
    useGPU = true;
else
    fprintf('No GPU detected, using CPU\n');
    useGPU = false;
end
```

### 2. Memory Management

```matlab
% For large datasets, process in batches
batchSize = 1000;
for i = 1:batchSize:size(testImages, 4)
    endIdx = min(i + batchSize - 1, size(testImages, 4));
    batch = testImages(:, :, :, i:endIdx);
    predictions = classify(net, batch);
    % Process predictions...
end
```

### 3. Parallel Processing

```matlab
% Enable parallel pool for faster preprocessing
if isempty(gcp('nocreate'))
    parpool('local');
end

% Use parallel processing for batch operations
parfor i = 1:numImages
    processedImages(:, :, :, i) = preprocess_single_image(images(:, :, :, i));
end
```

## 🔧 Troubleshooting Common Issues

### 1. Memory Issues

**Problem**: "Out of memory" errors during training

**Solutions:**
```matlab
% Reduce batch size
options.MiniBatchSize = 32;  % Instead of 128

% Use CPU instead of GPU
options.ExecutionEnvironment = 'cpu';

% Clear unnecessary variables
clear largeVariable1 largeVariable2;
```

### 2. Poor Performance

**Problem**: Accuracy below 90%

**Debugging steps:**
```matlab
% Check data quality
figure; imshow(trainImages(:,:,1,1)); title('Sample training image');

% Verify labels
unique_labels = categories(trainLabels);
fprintf('Found classes: %s\n', strjoin(string(unique_labels), ', '));

% Check for class imbalance
for i = 1:length(unique_labels)
    count = sum(trainLabels == unique_labels{i});
    fprintf('Class %s: %d samples\n', unique_labels{i}, count);
end
```

### 3. Training Not Converging

**Problem**: Loss not decreasing or accuracy not improving

**Solutions:**
```matlab
% Reduce learning rate
options.InitialLearnRate = 0.0001;  % Instead of 0.001

% Increase training epochs
options.MaxEpochs = 50;  % Instead of 20

% Add learning rate scheduling
options.LearnRateSchedule = 'piecewise';
options.LearnRateDropFactor = 0.5;
options.LearnRateDropPeriod = 10;
```

### 4. Overfitting

**Problem**: Training accuracy much higher than validation accuracy

**Solutions:**
```matlab
% Add more dropout
dropoutLayer(0.7, 'Name', 'dropout1')  % Instead of 0.5

% Reduce model complexity
fullyConnectedLayer(64, 'Name', 'fc1')  % Instead of 128

% Add more data augmentation
augmentationFactor = 3;  % Instead of 2
```

## 📊 Evaluation and Analysis

### 1. Detailed Performance Analysis

```matlab
% Get comprehensive performance metrics
classPerformance = calculate_class_performance(testLabels, predictedLabels);
display_class_performance(classPerformance);

% Find problematic classes
[bestClass, worstClass] = find_best_worst_classes(classPerformance);
```

### 2. Confusion Matrix Analysis

```matlab
% Generate and analyze confusion matrix
confMat = plot_confusion_matrix(testLabels, predictedLabels, 'Model Performance');

% Find most common misclassifications
[maxError, errorIdx] = max(confMat - diag(diag(confMat)), [], 'all');
[trueClass, predClass] = ind2sub(size(confMat), errorIdx);
fprintf('Most common error: %d misclassified as %d (%d times)\n', ...
    trueClass-1, predClass-1, maxError);
```

### 3. Visualizing Results

```matlab
% Show sample predictions
visualize_predictions(testImages, testLabels, predictedLabels, 16);

% Display training progress
if exist('trainingInfo', 'var')
    plot_training_progress(trainingInfo);
end

% Show model architecture summary
analyzeNetwork(net);
```

## 💾 Model Management

### 1. Saving Models

```matlab
% Save with timestamp
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
filename = sprintf('digit_cnn_%s.mat', timestamp);
save(filename, 'net', 'accuracy', 'classPerformance', 'trainingTime');
```

### 2. Loading and Using Saved Models

```matlab
% Load model
modelFile = 'digit_cnn_20250101_120000.mat';
loadedData = load(modelFile);
net = loadedData.net;

% Use for predictions
newPredictions = classify(net, newImages);
```

### 3. Model Versioning

```matlab
% Create model info structure
modelInfo.version = '1.0';
modelInfo.date = datestr(now);
modelInfo.accuracy = accuracy;
modelInfo.architecture = 'CNN-3Conv-2FC';
modelInfo.parameters = calculate_model_parameters(net);

% Save with model
save(filename, 'net', 'modelInfo');
```

## 🎨 Visualization Options

### 1. Custom Plot Styling

```matlab
% Set default figure properties
set(0, 'DefaultFigureColor', 'white');
set(0, 'DefaultAxesFontSize', 12);
set(0, 'DefaultAxesFontWeight', 'bold');
```

### 2. Export Results

```matlab
% Save figures
figHandles = findall(0, 'Type', 'figure');
for i = 1:length(figHandles)
    figName = sprintf('result_figure_%d.png', i);
    saveas(figHandles(i), figName);
end

% Export to PDF
print('confusion_matrix', '-dpdf', '-r300');
```

### 3. Create Custom Visualizations

```matlab
% Plot accuracy comparison
figure;
classes = 0:9;
bar(classes, classPerformance * 100);
title('Per-Class Accuracy');
xlabel('Digit Class');
ylabel('Accuracy (%)');
ylim([0, 100]);
```

## 🚀 Advanced Usage

### 1. Ensemble Methods

```matlab
% Train multiple models
models = cell(1, 5);
for i = 1:5
    % Add randomness to training
    shuffleIdx = randperm(size(trainImages, 4));
    models{i} = trainNetwork(trainImages(:,:,:,shuffleIdx), ...
                            trainLabels(shuffleIdx), layers, options);
end

% Ensemble prediction
ensemblePredictions = [];
for i = 1:5
    predictions = classify(models{i}, testImages);
    ensemblePredictions = [ensemblePredictions, double(predictions)];
end

% Majority voting
finalPredictions = mode(ensemblePredictions, 2);
```

### 2. Transfer Learning

```matlab
% Load pre-trained network (if available)
pretrainedNet = load('pretrained_digit_model.mat');

% Modify for your data
layers = pretrainedNet.net.Layers;
layers(end-2) = fullyConnectedLayer(10);  % 10 classes
layers(end) = classificationLayer;

% Fine-tune with lower learning rate
options.InitialLearnRate = 0.0001;
fineTunedNet = trainNetwork(trainImages, trainLabels, layers, options);
```

### 3. Hyperparameter Optimization

```matlab
% Define hyperparameter search space
learningRates = [0.01, 0.001, 0.0001];
batchSizes = [32, 64, 128];
dropoutRates = [0.3, 0.5, 0.7];

bestAccuracy = 0;
bestParams = struct();

% Grid search
for lr = learningRates
    for bs = batchSizes
        for dr = dropoutRates
            % Create custom options
            tempOptions = create_training_options(true);
            tempOptions.InitialLearnRate = lr;
            tempOptions.MiniBatchSize = bs;
            
            % Create custom layers with dropout
            tempLayers = create_custom_layers(dr);
            
            % Train and evaluate
            tempNet = trainNetwork(trainImages, trainLabels, tempLayers, tempOptions);
            tempPredictions = classify(tempNet, testImages);
            tempAccuracy = calculate_accuracy(tempPredictions, testLabels);
            
            % Track best parameters
            if tempAccuracy > bestAccuracy
                bestAccuracy = tempAccuracy;
                bestParams.learningRate = lr;
                bestParams.batchSize = bs;
                bestParams.dropoutRate = dr;
            end
            
            fprintf('LR=%.4f, BS=%d, DR=%.1f: Accuracy=%.2f%%\n', ...
                lr, bs, dr, tempAccuracy * 100);
        end
    end
end

fprintf('Best parameters: LR=%.4f, BS=%d, DR=%.1f, Accuracy=%.2f%%\n', ...
    bestParams.learningRate, bestParams.batchSize, bestParams.dropoutRate, bestAccuracy * 100);
```

This usage guide provides comprehensive instructions for effectively using the handwritten digit recognition system, from basic execution to advanced customization and optimization techniques.

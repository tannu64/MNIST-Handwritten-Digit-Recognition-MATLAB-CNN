%% Handwritten Digit Recognition using Deep Learning in MATLAB
% Author: Created for comprehensive digit classification using CNN
% Date: Implementation based on requirements
% Objective: Design and implement a deep learning-based handwritten digit 
%           recognition system using the MNIST dataset with CNN architecture

%% Clear workspace and initialize
clear; clc; close all;

% Add current directory to path for accessing helper functions
addpath(genpath(pwd));

% Set random seed for reproducibility
rng(42);

fprintf('=== Handwritten Digit Recognition using Deep Learning ===\n\n');

%% Step 1: Setup Environment and Check Toolbox
fprintf('Step 1: Setting up environment...\n');

% Check if Deep Learning Toolbox is available
if ~license('test', 'Neural_Network_Toolbox')
    error('Deep Learning Toolbox is required for this implementation.');
end

% Check GPU availability for faster training
if gpuDeviceCount > 0
    fprintf('GPU detected: %s\n', gpuDevice().Name);
    useGPU = true;
else
    fprintf('No GPU detected. Using CPU for training.\n');
    useGPU = false;
end

%% Step 2: Load and Prepare MNIST Dataset
fprintf('\nStep 2: Loading MNIST dataset...\n');

try
    % Load MNIST training data
    [trainImages, trainLabels] = load_mnist_data('train');
    fprintf('Training data loaded: %d images\n', size(trainImages, 4));
    
    % Load MNIST test data  
    [testImages, testLabels] = load_mnist_data('test');
    fprintf('Test data loaded: %d images\n', size(testImages, 4));
    
catch ME
    % If MNIST data not available, use MATLAB's built-in digit data
    fprintf('Using MATLAB built-in digit dataset...\n');
    [trainImages, trainLabels, testImages, testLabels] = load_builtin_digit_data();
end

%% Step 3: Data Preprocessing
fprintf('\nStep 3: Preprocessing data...\n');

% Preprocess training data
[trainImages, trainLabels] = preprocess_digit_data(trainImages, trainLabels);

% Preprocess test data
[testImages, testLabels] = preprocess_digit_data(testImages, testLabels);

% Display data information
fprintf('Training set: %d images of size %dx%dx%d\n', ...
    size(trainImages, 4), size(trainImages, 1), size(trainImages, 2), size(trainImages, 3));
fprintf('Test set: %d images of size %dx%dx%d\n', ...
    size(testImages, 4), size(testImages, 1), size(testImages, 2), size(testImages, 3));

% Visualize sample images
visualize_sample_digits(trainImages, trainLabels, 'Training Data Samples');

%% Step 4: Define CNN Architecture
fprintf('\nStep 4: Defining CNN architecture...\n');

% Create CNN layers
layers = create_cnn_architecture();

% Display network architecture
fprintf('CNN Architecture created with %d layers\n', length(layers));
analyzeNetwork(layers);

%% Step 5: Set Training Options
fprintf('\nStep 5: Setting training options...\n');

% Define training options
options = create_training_options(useGPU);

% Display training configuration
fprintf('Training configuration:\n');
fprintf('  - Optimizer: ADAM\n');
fprintf('  - Initial Learning Rate: %.4f\n', options.InitialLearnRate);
fprintf('  - Max Epochs: %d\n', options.MaxEpochs);
fprintf('  - Mini Batch Size: %d\n', options.MiniBatchSize);

%% Step 6: Train CNN Model
fprintf('\nStep 6: Training CNN model...\n');
fprintf('This may take several minutes depending on your hardware...\n');

% Start training timer
tic;

% Train the network
net = trainNetwork(trainImages, trainLabels, layers, options);

% Calculate training time
trainingTime = toc;
fprintf('Training completed in %.2f seconds (%.2f minutes)\n', ...
    trainingTime, trainingTime/60);

%% Step 7: Test and Evaluate Model
fprintf('\nStep 7: Evaluating model performance...\n');

% Make predictions on test set
predictedLabels = classify(net, testImages);

% Calculate accuracy
accuracy = calculate_accuracy(predictedLabels, testLabels);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

% Generate and display confusion matrix
plot_confusion_matrix(testLabels, predictedLabels, 'Test Set Confusion Matrix');

% Calculate per-class performance
class_performance = calculate_class_performance(testLabels, predictedLabels);
display_class_performance(class_performance);

%% Step 8: Visualize Results
fprintf('\nStep 8: Visualizing results...\n');

% Debug the variables before calling visualization
fprintf('Before visualization - Debug info:\n');
fprintf('testImages size: %s\n', mat2str(size(testImages)));
fprintf('testLabels type: %s, length: %d\n', class(testLabels), length(testLabels));
fprintf('predictedLabels type: %s, length: %d\n', class(predictedLabels), length(predictedLabels));

% Show sample predictions
visualize_predictions(testImages, testLabels, predictedLabels, 16);

% Plot training progress if available
if isfield(net, 'TrainingHistory')
    plot_training_progress(net.TrainingHistory);
end

%% Step 9: Save Trained Model
fprintf('\nStep 9: Saving trained model...\n');

% Create models directory if it doesn't exist
modelsDir = 'models';
if ~exist(modelsDir, 'dir')
    fprintf('Creating models directory...\n');
    try
        mkdir(modelsDir);
        fprintf('Models directory created successfully.\n');
    catch ME
        fprintf('Warning: Could not create models directory: %s\n', ME.message);
        fprintf('Saving to current directory instead.\n');
        modelsDir = '.';
    end
else
    fprintf('Models directory already exists.\n');
end

% Save the trained network
modelFileName = sprintf('%s/digit_recognition_cnn_%s.mat', ...
    modelsDir, datestr(now, 'yyyymmdd_HHMMSS'));

fprintf('Saving model to: %s\n', modelFileName);
try
    save(modelFileName, 'net', 'accuracy', 'class_performance', 'trainingTime');
    fprintf('Model saved successfully as: %s\n', modelFileName);
catch ME
    fprintf('Error saving model: %s\n', ME.message);
    % Fallback: save to current directory with simple filename
    fallbackFileName = sprintf('digit_recognition_model_%s.mat', datestr(now, 'yyyymmdd_HHMMSS'));
    fprintf('Attempting fallback save to: %s\n', fallbackFileName);
    save(fallbackFileName, 'net', 'accuracy', 'class_performance', 'trainingTime');
    fprintf('Model saved as fallback: %s\n', fallbackFileName);
end

%% Step 10: Performance Summary
fprintf('\n=== PERFORMANCE SUMMARY ===\n');
fprintf('Overall Test Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Training Time: %.2f minutes\n', trainingTime/60);
fprintf('Model Architecture: %d-layer CNN\n', length(layers));
fprintf('Total Parameters: %d\n', calculate_model_parameters(net));

% Show best and worst performing classes
[bestClass, worstClass] = find_best_worst_classes(class_performance);
fprintf('Best performing class: %d (%.2f%% accuracy)\n', ...
    bestClass, class_performance(bestClass+1) * 100);
fprintf('Worst performing class: %d (%.2f%% accuracy)\n', ...
    worstClass, class_performance(worstClass+1) * 100);

fprintf('\n=== IMPLEMENTATION COMPLETED SUCCESSFULLY ===\n');

%% Interactive Testing (Optional)
% Uncomment the following line to enable interactive digit testing
% interactive_digit_testing(net);

function [processedImages, processedLabels] = preprocess_digit_data(images, labels)
%PREPROCESS_DIGIT_DATA Preprocess digit images and labels for CNN training
%
% Syntax:
%   [processedImages, processedLabels] = preprocess_digit_data(images, labels)
%
% Inputs:
%   images - 4D array of images [height, width, channels, num_images]
%   labels - Categorical array or numeric array of labels
%
% Outputs:
%   processedImages - Preprocessed 4D array ready for CNN training
%   processedLabels - Categorical labels ready for classification
%
% Description:
%   This function performs comprehensive preprocessing including:
%   - Image normalization to [0, 1] range
%   - Data type conversion for optimal training
%   - Label format standardization
%   - Optional data augmentation
%   - Image quality validation

    fprintf('Preprocessing digit data...\n');
    
    % Validate inputs
    if ndims(images) ~= 4
        error('Images must be a 4D array [height, width, channels, num_images]');
    end
    
    % Get data dimensions
    [height, width, channels, numImages] = size(images);
    fprintf('Input dimensions: [%d, %d, %d, %d]\n', height, width, channels, numImages);
    
    %% Step 1: Image Preprocessing
    fprintf('  Step 1: Normalizing pixel values...\n');
    
    % Convert to double for processing
    processedImages = double(images);
    
    % Normalize pixel values to [0, 1] range
    if max(processedImages(:)) > 1
        processedImages = processedImages / 255.0;
    end
    
    % Ensure values are in valid range
    processedImages = max(0, min(1, processedImages));
    
    % Convert back to single precision for memory efficiency and GPU compatibility
    processedImages = single(processedImages);
    
    %% Step 2: Image Quality Enhancement
    fprintf('  Step 2: Enhancing image quality...\n');
    
    % Apply contrast enhancement to improve feature visibility
    processedImages = enhance_contrast(processedImages);
    
    % Remove any potential NaN or Inf values
    processedImages(isnan(processedImages)) = 0;
    processedImages(isinf(processedImages)) = 0;
    
    %% Step 3: Data Validation
    fprintf('  Step 3: Validating data quality...\n');
    
    % Check for and remove corrupted images
    [processedImages, processedLabels, removedIndices] = remove_corrupted_images(processedImages, labels);
    
    if ~isempty(removedIndices)
        fprintf('    Removed %d corrupted images\n', length(removedIndices));
    end
    
    %% Step 4: Label Preprocessing
    fprintf('  Step 4: Processing labels...\n');
    
    % Ensure labels are categorical
    if ~iscategorical(processedLabels)
        processedLabels = categorical(processedLabels);
    end
    
    % Verify label consistency
    uniqueLabels = categories(processedLabels);
    fprintf('    Found %d unique classes: %s\n', length(uniqueLabels), ...
        strjoin(string(uniqueLabels), ', '));
    
    % Ensure we have all digits 0-9
    expectedLabels = categorical(0:9);
    missingLabels = setdiff(expectedLabels, processedLabels);
    if ~isempty(missingLabels)
        warning('Missing labels in dataset: %s', strjoin(string(missingLabels), ', '));
    end
    
    %% Step 5: Data Augmentation (Optional)
    if should_apply_augmentation(numImages)
        fprintf('  Step 5: Applying data augmentation...\n');
        [processedImages, processedLabels] = apply_data_augmentation(processedImages, processedLabels);
    else
        fprintf('  Step 5: Skipping data augmentation (sufficient data available)\n');
    end
    
    %% Step 6: Final Validation and Summary
    fprintf('  Step 6: Final validation...\n');
    
    % Ensure correct dimensions for CNN input
    if size(processedImages, 3) ~= 1
        error('Images must have exactly 1 channel (grayscale)');
    end
    
    if size(processedImages, 1) ~= 28 || size(processedImages, 2) ~= 28
        error('Images must be 28x28 pixels');
    end
    
    % Display preprocessing summary
    finalNumImages = size(processedImages, 4);
    fprintf('\nPreprocessing Summary:\n');
    fprintf('  Original images: %d\n', numImages);
    fprintf('  Final images: %d\n', finalNumImages);
    fprintf('  Image size: [%d, %d, %d]\n', size(processedImages, 1), ...
        size(processedImages, 2), size(processedImages, 3));
    fprintf('  Data type: %s\n', class(processedImages));
    fprintf('  Value range: [%.3f, %.3f]\n', min(processedImages(:)), max(processedImages(:)));
    fprintf('  Label type: %s\n', class(processedLabels));
    fprintf('  Number of classes: %d\n', length(categories(processedLabels)));
    
    % Display class distribution
    display_class_distribution(processedLabels);
end

function enhancedImages = enhance_contrast(images)
%ENHANCE_CONTRAST Enhance contrast of digit images
%
% Applies adaptive histogram equalization to improve feature visibility

    enhancedImages = images;
    numImages = size(images, 4);
    
    % Process images in batches for memory efficiency
    batchSize = 1000;
    numBatches = ceil(numImages / batchSize);
    
    for batch = 1:numBatches
        startIdx = (batch - 1) * batchSize + 1;
        endIdx = min(batch * batchSize, numImages);
        currentBatchSize = endIdx - startIdx + 1;
        
        for i = 1:currentBatchSize
            imgIdx = startIdx + i - 1;
            img = images(:, :, 1, imgIdx);
            
            % Apply adaptive histogram equalization
            enhanced = adapthisteq(img);
            
            % Blend with original to avoid over-enhancement
            blendRatio = 0.3;
            enhancedImages(:, :, 1, imgIdx) = ...
                blendRatio * enhanced + (1 - blendRatio) * img;
        end
        
        if batch == 1 || mod(batch, 10) == 0
            fprintf('    Enhanced %d/%d images\n', endIdx, numImages);
        end
    end
end

function [cleanImages, cleanLabels, removedIndices] = remove_corrupted_images(images, labels)
%REMOVE_CORRUPTED_IMAGES Remove images that are corrupted or of poor quality
%
% Identifies and removes images that are completely black, white, or have
% other quality issues that would negatively impact training

    numImages = size(images, 4);
    keepIndices = true(numImages, 1);
    
    fprintf('    Checking image quality...\n');
    
    for i = 1:numImages
        img = images(:, :, 1, i);
        
        % Check for completely black images
        if max(img(:)) < 0.01
            keepIndices(i) = false;
            continue;
        end
        
        % Check for completely white images
        if min(img(:)) > 0.99
            keepIndices(i) = false;
            continue;
        end
        
        % Check for extremely low contrast
        if (max(img(:)) - min(img(:))) < 0.05
            keepIndices(i) = false;
            continue;
        end
        
        % Check for NaN or Inf values
        if any(isnan(img(:))) || any(isinf(img(:)))
            keepIndices(i) = false;
            continue;
        end
    end
    
    % Remove corrupted images and corresponding labels
    cleanImages = images(:, :, :, keepIndices);
    cleanLabels = labels(keepIndices);
    removedIndices = find(~keepIndices);
end

function shouldAugment = should_apply_augmentation(numImages)
%SHOULD_APPLY_AUGMENTATION Determine if data augmentation should be applied
%
% Data augmentation is applied if the dataset is small to improve
% generalization and prevent overfitting

    % Apply augmentation if we have fewer than 5000 images per class
    minImagesPerClass = 5000;
    shouldAugment = numImages < (minImagesPerClass * 10);  % 10 classes (0-9)
end

function [augmentedImages, augmentedLabels] = apply_data_augmentation(images, labels)
%APPLY_DATA_AUGMENTATION Apply data augmentation to increase dataset size
%
% Applies various transformations to create additional training examples

    fprintf('    Applying data augmentation...\n');
    
    originalSize = size(images, 4);
    
    % Define augmentation parameters
    augmentationFactor = 2;  % Create 2x more data
    
    % Calculate target size
    targetSize = originalSize * augmentationFactor;
    
    % Initialize augmented arrays
    augmentedImages = zeros(28, 28, 1, targetSize, 'single');
    augmentedLabels = categorical.empty(targetSize, 0);
    
    % Copy original data
    augmentedImages(:, :, :, 1:originalSize) = images;
    augmentedLabels(1:originalSize) = labels;
    
    % Generate augmented examples
    for i = 1:(targetSize - originalSize)
        % Select random original image
        originalIdx = randi(originalSize);
        originalImg = images(:, :, 1, originalIdx);
        originalLabel = labels(originalIdx);
        
        % Apply random augmentation
        augmentedImg = apply_random_augmentation(originalImg);
        
        % Store augmented image and label
        augmentedImages(:, :, 1, originalSize + i) = augmentedImg;
        augmentedLabels(originalSize + i) = originalLabel;
        
        % Progress update
        if mod(i, 1000) == 0
            fprintf('      Generated %d/%d augmented images\n', i, targetSize - originalSize);
        end
    end
    
    fprintf('    Data augmentation completed: %d -> %d images\n', originalSize, targetSize);
end

function augmentedImg = apply_random_augmentation(img)
%APPLY_RANDOM_AUGMENTATION Apply random transformation to a single image
%
% Applies one or more random transformations while preserving digit characteristics

    augmentedImg = img;
    
    % List of possible augmentations
    augmentations = {'rotation', 'translation', 'scaling', 'shearing', 'noise'};
    
    % Randomly select 1-2 augmentations to apply
    numAugmentations = randi([1, 2]);
    selectedAugmentations = augmentations(randperm(length(augmentations), numAugmentations));
    
    for i = 1:length(selectedAugmentations)
        switch selectedAugmentations{i}
            case 'rotation'
                % Small rotation (-15 to +15 degrees)
                angle = (rand() - 0.5) * 30;
                augmentedImg = imrotate(augmentedImg, angle, 'bilinear', 'crop');
                
            case 'translation'
                % Small translation (-3 to +3 pixels)
                shiftX = round((rand() - 0.5) * 6);
                shiftY = round((rand() - 0.5) * 6);
                augmentedImg = imtranslate(augmentedImg, [shiftX, shiftY]);
                
            case 'scaling'
                % Small scaling (0.9 to 1.1)
                scaleFactor = 0.9 + rand() * 0.2;
                augmentedImg = imresize(augmentedImg, scaleFactor);
                
                % Crop or pad to maintain 28x28 size
                [h, w] = size(augmentedImg);
                if h ~= 28 || w ~= 28
                    if h > 28 || w > 28
                        % Crop from center
                        startR = round((h - 28) / 2) + 1;
                        startC = round((w - 28) / 2) + 1;
                        augmentedImg = augmentedImg(startR:startR+27, startC:startC+27);
                    else
                        % Pad with zeros
                        padR = (28 - h) / 2;
                        padC = (28 - w) / 2;
                        augmentedImg = padarray(augmentedImg, [floor(padR), floor(padC)], 0, 'both');
                        if size(augmentedImg, 1) < 28
                            augmentedImg = padarray(augmentedImg, [1, 0], 0, 'post');
                        end
                        if size(augmentedImg, 2) < 28
                            augmentedImg = padarray(augmentedImg, [0, 1], 0, 'post');
                        end
                        augmentedImg = augmentedImg(1:28, 1:28);
                    end
                end
                
            case 'shearing'
                % Small shearing
                shearX = (rand() - 0.5) * 0.2;
                shearY = (rand() - 0.5) * 0.2;
                tform = affine2d([1 shearX 0; shearY 1 0; 0 0 1]);
                augmentedImg = imwarp(augmentedImg, tform, 'OutputView', imref2d(size(augmentedImg)));
                
            case 'noise'
                % Add small amount of Gaussian noise
                noiseLevel = 0.02;
                noise = randn(size(augmentedImg)) * noiseLevel;
                augmentedImg = augmentedImg + noise;
                
        end
    end
    
    % Ensure values remain in [0, 1] range
    augmentedImg = max(0, min(1, augmentedImg));
end

function display_class_distribution(labels)
%DISPLAY_CLASS_DISTRIBUTION Display the distribution of classes in the dataset
%
% Shows how many examples exist for each digit class

    fprintf('\nClass Distribution:\n');
    
    uniqueLabels = categories(labels);
    
    for i = 1:length(uniqueLabels)
        label = uniqueLabels{i};
        count = sum(labels == label);
        percentage = (count / length(labels)) * 100;
        fprintf('  Class %s: %d samples (%.1f%%)\n', label, count, percentage);
    end
    
    % Check for class imbalance
    counts = zeros(length(uniqueLabels), 1);
    for i = 1:length(uniqueLabels)
        counts(i) = sum(labels == uniqueLabels{i});
    end
    
    imbalanceRatio = max(counts) / min(counts);
    if imbalanceRatio > 2
        warning('Class imbalance detected (ratio: %.1f). Consider balancing the dataset.', imbalanceRatio);
    else
        fprintf('  Dataset is reasonably balanced (ratio: %.1f)\n', imbalanceRatio);
    end
end

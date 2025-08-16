function visualize_sample_digits(images, labels, titleStr, numSamples)
%VISUALIZE_SAMPLE_DIGITS Display sample digit images with their labels
%
% Syntax:
%   visualize_sample_digits(images, labels, titleStr, numSamples)
%
% Inputs:
%   images     - 4D array [H, W, C, N] of digit images
%   labels     - Categorical array of digit labels
%   titleStr   - String for figure title (optional)
%   numSamples - Number of samples to display (optional, default: 25)
%
% Description:
%   Creates a grid visualization showing sample digit images with their
%   corresponding labels. Useful for data exploration and verification.

    if nargin < 3
        titleStr = 'Sample Digit Images';
    end
    if nargin < 4
        numSamples = 25;  % Default to 5x5 grid
    end
    
    % Validate inputs
    if size(images, 4) < numSamples
        numSamples = size(images, 4);
        fprintf('Reducing sample count to %d (total available images)\n', numSamples);
    end
    
    if length(labels) < numSamples
        numSamples = length(labels);
        fprintf('Reducing sample count to %d (total available labels)\n', numSamples);
    end
    
    % Determine grid size
    gridSize = ceil(sqrt(numSamples));
    
    % Randomly select samples to display
    totalImages = size(images, 4);
    sampleIndices = randperm(totalImages, numSamples);
    
    % Create figure
    figure('Name', titleStr, 'Position', [150, 200, 800, 800]);
    
    fprintf('Displaying %d sample images in %dx%d grid...\n', numSamples, gridSize, gridSize);
    
    for i = 1:numSamples
        idx = sampleIndices(i);
        
        % Extract image and label
        img = images(:, :, 1, idx);
        label = labels(idx);
        
        % Create subplot
        subplot(gridSize, gridSize, i);
        
        % Display image
        imshow(img, []);
        
        % Add title with label
        title(sprintf('Label: %s', string(label)), 'FontSize', 10);
        
        % Remove axis ticks for cleaner appearance
        set(gca, 'XTick', [], 'YTick', []);
    end
    
    % Add main title
    sgtitle(titleStr, 'FontSize', 16, 'FontWeight', 'bold');
    
    % Adjust spacing between subplots
    set(gcf, 'Units', 'normalized');
    
    fprintf('Sample visualization complete.\n');
end

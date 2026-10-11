
function test_model()

    clc;
    clear;

    %% 1. Activity names mapping
    LABEL_NAMES = {
        'Walking'
        'Running'
        'Shuffling'
        'Stairs (ascending)'
        'Stairs (descending)'
        'Standing'
        'Sitting'
        'Lying'
        'Cycling (sit)'
        'Cycling (stand)'
        'Cycling (sit, inactive)'
        'Cycling (stand, inactive)'
    };

    % Raw dataset labels mapped to zero-based model class indices
    RAW_LABELS = [1 2 3 4 5 6 7 8 13 14 130 140];
    CLASS_INDICES = 0:11;

    %% 2. Normalization parameters from training
    MEAN_VAL = [-0.885417, 0.001669, -0.147612, ...
                -0.507975, 0.059537, 0.532519];

    STD_VAL = [0.295474, 0.225258, 0.338422, ...
               0.499131, 0.308348, 0.587967];

    %% 3. Load trained model
    % This file must contain the trained MATLAB network.
    % Expected variable name: net
    modelFile = 'best_model.mat';

    if ~isfile(modelFile)
        error(['Cannot find best_model.mat. Convert/import the trained ' ...
               'PyTorch weights before running inference.']);
    end

    loadedData = load(modelFile);

    if isfield(loadedData, 'net')
        net = loadedData.net;
    else
        error('The MAT file must contain a trained network named net.');
    end

    %% 4. Load test dataset
    testFile = fullfile('dataset', 'harth', 'S029.csv');

    if ~isfile(testFile)
        error('Cannot find test file: %s', testFile);
    end

    fprintf('Reading sample from %s...\n', testFile);

    df = readtable(testFile);

    %% 5. Extract a window of 128 continuous samples
    startIdx = 1001;   % MATLAB indexing: Python row index 1000
    windowLength = 128;

    if height(df) < startIdx + windowLength - 1
        error('The CSV does not contain enough rows for this window.');
    end

    windowData = df(startIdx:startIdx + windowLength - 1, :);

    featureCols = {
        'back_x'
        'back_y'
        'back_z'
        'thigh_x'
        'thigh_y'
        'thigh_z'
    };

    rawFeatures = windowData{:, featureCols};

    %% 6. Extract the ground-truth label at the window center
    rawLabel = windowData.label(65);

    mapIdx = find(RAW_LABELS == rawLabel, 1);

    if isempty(mapIdx)
        trueClassIdx = -1;
        trueClassName = 'Unknown';
    else
        trueClassIdx = CLASS_INDICES(mapIdx);
        trueClassName = LABEL_NAMES{mapIdx};
    end

    %% 7. Normalize sensor features
    normFeatures = (rawFeatures - MEAN_VAL) ./ STD_VAL;

    % Convert to single precision.
    % MATLAB sequence input format: channels x time steps
    inputData = single(normFeatures');

    %% 8. Run inference
    % Output should contain 12 class scores.
    scores = predict(net, inputData);

    scores = gather(extractdataIfNeeded(scores));
    scores = double(scores(:));

    % Convert logits to probabilities using numerically stable softmax
    scores = scores - max(scores);
    probabilities = exp(scores) ./ sum(exp(scores));

    [confidence, predictedPosition] = max(probabilities);

    % Convert MATLAB's 1-based index to zero-based class index
    predictedIdx = predictedPosition - 1;
    confidence = confidence * 100;

    predictedClassName = LABEL_NAMES{predictedPosition};

    %% 9. Display results
    fprintf('\n================ Inference Result ================\n');
    fprintf('True Activity      : %s (Raw Label: %g)\n', ...
        trueClassName, rawLabel);
    fprintf('Predicted Activity : %s (%.2f%% confidence)\n', ...
        predictedClassName, confidence);
    fprintf('==================================================\n');

    if predictedIdx == trueClassIdx
        fprintf('Success! Prediction matches the true label.\n');
    else
        fprintf('Prediction does not match.\n');
    end
end

function x = extractdataIfNeeded(x)
    % Support both dlnetwork output and ordinary numeric arrays.
    if isa(x, 'dlarray')
        x = extractdata(x);
    end
end

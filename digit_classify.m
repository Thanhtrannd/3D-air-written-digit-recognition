function C = digit_classify(testdata)
%Function C = digit_classify(testdata) takes input as a matrix N*3 data sample
% of an air-written digit collected by LeapMotion sensor and does the 
% recognition of the written digit.
%Input:
%   testdata: a matrix N*3 data sample (N number of 3-D location datapoint trajectories)
%Output:
%   C: The label of the written digit predicted by the trained neural
%   network model
    
    %% Turn sample data from a matrix to a cell array with one cell because
    % of input format of called functions
    test_datacell ={};
    test_datacell{1,1} = testdata;
    % number of timesteps (data points) of the given to-be-classified sample
    nTimesteps = size(test_datacell{1},1); 
    % Load data set which was used to build network model
    load data.mat data class
    %% Step 1: Normalize data
    test_datacell = data_normalization(test_datacell);
    
    %% Step 2: Extract features to ensure consistent size of data sample and
    model_data_size = size(data,1)/3;
    % If the input testdata sample has more point trajectories than the
    % data samples used to train the model, the testdata sample needs to
    % proceed feature extraction
    if nTimesteps > model_data_size
        test_datacell = feature_extraction(test_datacell, model_data_size);
    % SPECIAL CASE: MODEL RETRAINING NEEDED
    elseif nTimesteps < model_data_size
        % There is no need to extract feature for the given test data
        % sample because its number of datapoints is the new standard
%         test_datacell = feature_extraction(test_datacell, nTimesteps);
        % Repreprocess raw data
        load raw_data.mat raw_data class
        normalized_traindata = data_normalization(raw_data);
        extracted_traindata = feature_extraction(normalized_traindata,nTimesteps);
        train_data = data_reallocation(extracted_traindata);
        % Retrain network model and identify parameters
        [traindata_, trainclass_, testdata_, testclass_] = data_Holdout_splitting(train_data, class, 0.1);
        nHidden = [16,16];
        [wHidden, wOutput, nHidden] = mlp_train_val(traindata_, trainclass_, nHidden);
        net_testclass = feed_forward(testdata_, wHidden_, wOutput_);
        % check accuracy on traindata test set and print prompt
        fprintf("The accuracy of reimplemented network is %d.\n", sum(net_testclass==testclass_)/length(testclass_)); 
    end
    %% Step 3: Reallocate test data sample (turn a cell array with one cell into a column vector)
    test_datavec = data_reallocation(test_datacell);
    
    %% Step 4: Predict label for input testdata (feed-forward operation)
    if nTimesteps < model_data_size
        % SPECIAL CASE: USE RETRAINED PARAMETERS
        C = feed_forward(test_datavec, wHidden_, wOutput_);
    else
        % NORMAL CASE: USE TRAINED PARAMETERS
        load parameters.mat
        C = feed_forward(test_datavec, wHidden, wOutput);
    end
    
       
    
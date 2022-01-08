% Load pre-processed data for training neural network model and store parameters
% for using in classifier function C = digit_classify(testdata)

close all, clear all, clc
% Load preprocessed data (variables: data, class)
load data.mat

%% Step 1: Split data into train set and test set (90% train and 10% test)
[traindata, trainclass, testdata, testclass] = data_Holdout_splitting(data, class, 0.1);

%% Step 2: Train and validate model (using 90% of data set)
% Architect of hidden layers (row vector). The number of elements is the number of 
% hidden layers and the value of each element is the number of neurons in 
% that hidden layer.
nHidden = [16,16];
[wHidden, wOutput, nHidden] = mlp_train_val(traindata, trainclass, nHidden);

%% Step 3: Test model with test set (using 10% of data set)
net_testclass = feed_forward(testdata, wHidden, wOutput);
fprintf("The accuracy of the network is %d percent.\n", round(sum(net_testclass==testclass)/length(testclass),2)*100); % check accuracy on traindata test set

%% Step 4: Store parameters
% (This step is commented to avoid configuration in trained model parameters)
% save parameters.mat wHidden wOutput % uncomment this line if new model needs to be trained

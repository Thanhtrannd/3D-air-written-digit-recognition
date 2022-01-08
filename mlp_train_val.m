function [wHidden, wOutput, nHidden] = ...
  mlp_train_val(traindata_, trainclass_, nHidden, maxEpochs)
% mlp_train_val.m implements a shallow multilayer perceptron network using
% 90% of input data for training and the rest 10% for validation purpose.
% Input:
%   traindata_: input data set which is a data matrix with the orientation as that number of rows
%   is number of features and number of columns is the number of data
%   samples. 
%   trainclass_: input data class which is a row vector with the number of
%   elements is the number of data samples.
%   nHidden: a row vector of which number of elements represents the number
%   of hidden layers and the value of each elements is the number of
%   neurons in per layer.
% Output:
%   t: scalar representing the number of epoches
%   wHidden: a cell array of matrices of weights of each hidden layer. The
%   number of cell is equal to the number of hidden layers. Each weight
%   matrix has the number of columns representing the number of neurons on
%   that hidden layer and the number of rows representing the number of
%   neurons on the preview layer.
%   wOutput: a matrix of weights of output layer. The number of columns is
%   equal to neurons on the output layer and is all equal to the number of
%   unique class in the input data set. The number of rows is equal to the
%   number of neurons (+1) on the last hidden layer.

%% Step 1: Split data into 2 sets, one for training model (90% of input data)
% and one for validate the model (10% of input data) and identify when 
% overfitting might occurs
[traindata, trainclass, valdata, valclass] = data_Holdout_splitting(traindata_, trainclass_, 0.1);

%% Step 2: Acknowledge some attributes to recognize the suitable model's
% architecture and parameters
N = size(traindata, 2); % number of samples
d = size(traindata, 1); % number of features
nclass = length(unique(trainclass)); % number of classes
nHiddenLayers = size(nHidden, 2); % number of hidden layers
% Initialize maximum epochs if not provided
if ~exist('maxEpochs', 'var')
  maxEpochs = 100000;
end
%% Step 3: Print information about Network Architecture
fprintf("Network Architecture:\n");
fprintf("The input layer has (%d+1) neurons representing extended feature vectors of input data samples.\n", d);
fprintf("There are %d hidden layers in the network with extendedly each layer having ", nHiddenLayers);
for hidlayer = 1:nHiddenLayers
    fprintf("(%d + 1)", nHidden(hidlayer));
    if hidlayer~=nHiddenLayers
        fprintf(", ");
    end
end
fprintf(" neurons respectively.\n");
fprintf("The output layer has %d neurons representing the network data classes.\n", nclass);
fprintf("This function utilizes validation data (10 percent) from the input data to validate parameters \nadjusted by training the model with train data (90 percent) of the input data set.\n");
fprintf("The total number of epochs is %d.\n", maxEpochs);
disp("Parameters are returned after the training finishes.");

%% Step 4: Initialize model parameters and attributes of model
% Extend input train data
extendedInput = [traindata; ones(1, N)];

% Tranform trainclass vector into a matrix form
trainOutput = zeros(nclass, N);
for i = 1:N
  trainOutput(trainclass(i)+1, i) = 1;
end

% Initialize weights for each hidden layer
wHidden = {};
for hidlayer = 1:nHiddenLayers
    if hidlayer == 1
        wHidden{hidlayer} = (rand(d+1, nHidden(hidlayer))-0.5) / 10;
    else
        wHidden{hidlayer} = (rand(nHidden(hidlayer-1)+1, nHidden(hidlayer))-0.5) / 10;
    end
end
% Initialize weights for output layers
wOutput = (rand(nHidden(nHiddenLayers)+1, nclass)-0.5) / 10;

% Initialize storage place of wHidden and wOutput per 100 epoch. This track
% is only used to track back the epoch when overfitting occurs if any.
wHidden_track = {};
wOutput_track = {};

% loss function value vector initialisation
Jtrain = zeros(1, maxEpochs); % of train data
Jval = zeros(1, maxEpochs); % of validation data
% Define learning rate
rho = 0.0001; 

% fh1 = figure; % uncomment (line 92 and line 131-139) for visualization
% Initialize epoch count
t = 0; % Epoch count

%% Step 5: Train model
while 1 
    t = t+1;
    % Initialize cel arrays to store output per hidden layer
    vHidden = {};
    yHidden = {};
    % Feed-forward operation
    for hidlayer = 1:nHiddenLayers
        % hidden layer net activation
        if hidlayer == 1
            vHidden{hidlayer} = wHidden{hidlayer}'*extendedInput; 
        else
            vHidden{hidlayer} = wHidden{hidlayer}'*yHidden{hidlayer-1};
        end
        % hidden layer activation function
        yHidden{hidlayer} = tanh(vHidden{hidlayer}); 
        yHidden{hidlayer} = [yHidden{hidlayer}; ones(1,N)]; % hidden layer extended output
    end
    
    % output layer net activation
    vOutput = wOutput'*yHidden{nHiddenLayers}; 
    % output layer output without activation function
    yOutput = vOutput; 
    
    % loss function evaluation
    Jtrain(t) = 1/2*sum(sum((yOutput-trainOutput).^2)); % of traindata
    Jval(t) = cal_cost(wHidden, wOutput, valdata, valclass); % of validation data
    
    % save wHidden and wOutput for every 100 epoches to have a track of
    % parameters when overfitting exists
    if (mod(t, 100) == 0) 
        wHidden_track{t/100} = wHidden;
        wOutput_track{t/100} = wOutput;
    end
    
    % Plot training error at every 100 epoch % uncomment (line 92 and line 131-139) for visualization
    if (mod(t, 1000) == 0) 
        semilogy(1:t, Jtrain(1:t), "b-"), hold on;
        semilogy(1:t, Jval(1:t), "r-");
        title(sprintf('Training (epoch %d)', t));
        ylabel('Error');
        legend("training","validation");
        drawnow;
    end
    
    % Check if the learning is good enough, if yes, stop the training
    if (Jtrain(t) <1e-10) 
    break;
    end
    
    % Check if maximum epochs has been reached, if yes, stop the training
    if t >= maxEpochs 
    break;
    end
    % Check if the improvement is too small, if yes, stop the training
    if t > 1 % this is not the first epoch
        if abs(Jtrain(t) - Jtrain(t-1)) < 1e-10 
          break;
        end
    end
  
    % Update the sensitivities and the weights
    % For output layer
    deltaOutput = (yOutput - trainOutput);
    deltawOutput = -rho * yHidden{nHiddenLayers} * deltaOutput'; % (E7)
    wOutput = wOutput + deltawOutput; % update wOutput
    
    % For hidden layers
    deltaHidden = {};
    for hidlayer = nHiddenLayers:-1:1
        if hidlayer == nHiddenLayers % if this is the last hidden layer, consider output layer parameters
            deltaHidden{hidlayer} = (wOutput(1:end-1,:)*deltaOutput).*(1-yHidden{hidlayer}(1:end-1,:).^2);
            deltawHidden{hidlayer} = -rho * yHidden{hidlayer-1} * deltaHidden{hidlayer}';
        else % otherwise, consider the next hidden layer
            deltaHidden{hidlayer} = (wHidden{hidlayer+1}(1:end-1,:)*deltaHidden{hidlayer+1}).*(1-yHidden{hidlayer}(1:end-1,:).^2);
            if hidlayer == 1 % if this is the first hidden layer, consider input layer parameters
                deltawHidden{hidlayer} = -rho * extendedInput * deltaHidden{hidlayer}';
            else % otherwise, consider the previous hidden layer
                deltawHidden{hidlayer} = -rho * yHidden{hidlayer-1} * deltaHidden{hidlayer}';
            end
        end
        wHidden{hidlayer} = wHidden{hidlayer} + deltawHidden{hidlayer}; % update wHidden
    end
end

%% Step 6: Check if overfitting exists
% Overfitting is likely identified when the validation data has loss
% function value starting to increase
[min_Jval, min_Jval_ind] = min(Jval(100:100:end));

% Acknowledge overfitting if the last loss function value is not the
% minimum value
if min_Jval ~= Jval(end)
    fprintf("Overfitting exists from the %d epoch.\n", min_Jval_ind*100);
    wHidden = wHidden_track{min_Jval_ind};
    wOutput = wOutput_track{min_Jval_ind};
    disp("The return parameters are the ones updated just right before overfitting occurs.");
end
end

function Jval = cal_cost(wHidden, wOutput, valdata, valclass)
% This function operates feed-forward for the input data using given
% parameters and calculates loss function value

% Acknowledge attributes
nHiddenLayers = length(wHidden);
nVal = size(valdata, 2); % number of validation data samples
nclass = length(unique(valclass)); % number of classes
extendedVal = [valdata; ones(1, nVal)]; % extended input test data

% Tranform input class into a matrix to ease loss calculation
valOutput = zeros(nclass, nVal);
for i = 1:nVal
  valOutput(valclass(i)+1, i) = 1;
end

% Initialize cel arrays to store output per hidden layer
vHiddenVal = {}; % Input into per hidden layers
yHiddenVal = {}; % Output per hidden layers (apply tanh activation function onto input)

% Feed-forward operation
for hidlayer = 1:nHiddenLayers
    % hidden layer net activation
    if hidlayer == 1
        vHiddenVal{hidlayer} = wHidden{hidlayer}'*extendedVal; 
    else
        vHiddenVal{hidlayer} = wHidden{hidlayer}'*yHiddenVal{hidlayer-1};
    end
    % hidden layer activation function
    yHiddenVal{hidlayer} = tanh(vHiddenVal{hidlayer}); % hidden layer activation function
    yHiddenVal{hidlayer} = [yHiddenVal{hidlayer}; ones(1,nVal)]; % hidden layer extended output
end

vOutputVal = wOutput'*yHiddenVal{nHiddenLayers}; % output layer net activation
yOutputVal = vOutputVal; % output layer output without activation f

% loss function evaluation
Jval = 1/2*sum(sum((yOutputVal-valOutput).^2));
end
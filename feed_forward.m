function net_class = feed_forward(data, wHidden, wOutput)
%Function net_class = feed_forward(data, wHidden, wOutput)
%feed input data sample into trained model to predict its label
%Input:
%   data: Input data sample to be feed into model (a matrix with columns
%   corresponding samples and rows corresponding features)
%   wHidden: Parameters of hidden layers (a cell array with each cell
%   representing weights of each hidden layer) of the trained model
%   wOutput: Parameters of output layer (a matrix) of the trained model
%Output:
%   net_class: The label of the input data sample predicted by the
%   model

nHiddenLayers = length(wHidden); % number of hidden layers
ntest = size(data, 2); % number of data samples
extendeddata = [data; ones(1, ntest)]; % extended input data

% Initialize cel arrays to store output per hidden layer
vHiddentest = {}; % Input into per hidden layers
yHiddentest = {}; % Output per hidden layers (apply tanh activation function onto input)
% Feed-forward operation
for hidlayer = 1:nHiddenLayers
    % hidden layer net activation
    if hidlayer == 1
        vHiddentest{hidlayer} = wHidden{hidlayer}'*extendeddata;
    else
        vHiddentest{hidlayer} = wHidden{hidlayer}'*yHiddentest{hidlayer-1};
    end
    % hidden layer activation function
    yHiddentest{hidlayer} = tanh(vHiddentest{hidlayer}); 
    yHiddentest{hidlayer} = [yHiddentest{hidlayer}; ones(1,ntest)]; % hidden layer extended output
end

% output layer net activation
vOutputtest = wOutput'*yHiddentest{nHiddenLayers};
% output layer output without activation function
yOutputtest = vOutputtest;

% find most possibly correct class label for input data
[~, net_class] = max(yOutputtest, [], 1);
% because the max function outputs as indices from 1-10 and our class is from 0-9
net_class = net_class - 1; 
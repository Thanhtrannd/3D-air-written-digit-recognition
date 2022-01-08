function [reallocated_data] = data_reallocation(input_data)
%Function [reallocated_data] = data_reallocation(input_data) turns a matrix
%of timeseries data into featured column vectors. It creates a matrix 
%including samples in terms of column vectors with rows representing sample features.  
%Input:
%   input_data:  A cell array representing, per cell, a matrix of
%   datapoints per sample.
%Output:
%   reallocated_data: A matrix presenting samples on columns and features
%   on rows.

reallocated_data = [];
nSamples = length(input_data);

% Loop through all sample in the input cell array
for i=1:nSamples
    sample_temp = [];
    % Loop through all sample timesteps
    for ts = 1 : size(input_data{1,i},1)
        % Turn all location data points into a feature data orderly
        % [x1; y1; z1; x2; y2; z2; ... ; x19; y19; z19]
        sample_temp = [sample_temp; input_data{1,i}(ts,:)'];
    
    end
    % Store sample vector. Therefore, The number of collumns is equal to 
    % the number of samples inside the input data and the number of rows is
    % the number of data features 
    reallocated_data = [reallocated_data sample_temp];
end
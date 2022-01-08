function [normalized_data] = data_normalization(input_data)
%Function [normalized_data] = data_normalization(input_data)normalizes the 
%input data and returns a cell array with one sample per cell
%(using min-max scaling method).
%Input:
%   input_data:  A cell array representing, per cell, a matrix of
%   datapoints per sample
%Output:
%   normalized_data: A cell array presenting normalized data stored in a cell
%   array

normalized_data = {};
nSamples = length(input_data);
for i=1:nSamples
    sample_min = repmat(min(input_data{1,i}), size(input_data{1,i},1), 1);
    sample_max = repmat(max(input_data{1,i}), size(input_data{1,i},1), 1);
    normalized_data{1,i} = (input_data{1,i}-sample_min)./(sample_max-sample_min);
end
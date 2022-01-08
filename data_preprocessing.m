% Pre-processing the raw_data which is a cell array of inconsistent number
% of features and turns it into a matrix of data samples which is ready for
% training phase.

load raw_data.mat
nSamples = length(raw_data);


%% Step 1: data normalization (min-max scaling)
normalized_data = data_normalization(raw_data);

%% Step 2: Extract useful sample features with the minimum number of
% 3D features (x,y,z) as exists
% find min number of timesteps
min_Ntimesteps = 100;
for i=1:nSamples
    if size(raw_data{1,i}, 1) < min_Ntimesteps
        min_Ntimesteps = size(raw_data{1,i}, 1);
    end
end
extracted_data = feature_extraction(normalized_data,min_Ntimesteps);

%% Step 3: Transform data from a cell array of 3D time-series data to a matrix
%  with columns as the number of samples and rows representing features.
data = data_reallocation(extracted_data);

%% Step 4: Store preprocessed data for later constructing neural network model
% (This step is commented to avoid configuration in pre_processed training data set)
% save data.mat data class
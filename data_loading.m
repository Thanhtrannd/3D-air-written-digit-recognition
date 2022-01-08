%% Load all data file available in the directory and store them into a cell array
% Note: Provided separate raw data files for the method training phase are
% not included in the submission. Therefore, in order to run this function,
% those data files must be available in the same directory as this file.

%% Step 1: Load and store all raw stroke data file into a cell array
% All files with .mat suffix
mat = dir('*.mat');
% Initialize a cell array to store 3D time-series data samples
raw_data = {};
file_struct = struct();
for q = 1:length(mat) 
    file_struct = load(mat(q).name);
    raw_data{q}=file_struct.pos;
end

%% Step 2: Create a vector containing sample class
% (data files are loaded in the order that 100 samples for each class from 0 to 9)
class = [];
for num = 0:9
    class = [class num*ones(1, 100)];
end

%% Step 3:Save loaded raw data (this step is commented to avoid configuration in raw_data set)
% save raw_data.mat raw_data class

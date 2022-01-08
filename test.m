% Test function digit_classify.m and the trained model with all raw data 
% available in training phase 

% Load preprocessed data (variables: data, class)
load raw_data.mat raw_data class
load parameters.mat

% Initialize vector to keep track of predict performance
isCorrect = zeros(1,length(raw_data));

% Loop through the whole available data set
for sample = 1:length(raw_data)
    % testdata (one sample at a time)
    testdata_ = raw_data{1,sample};
    testclass_ = class(sample);
    % Predict class for testdata with function digit_classify
    net_testlabel = digit_classify(testdata_);
    
    % Printing result
%     fprintf("The test data class is %d and the network label that test data as class %d.\n", testclass_, net_testlabel);
    if testclass_ == net_testlabel
        isCorrect(sample) = 1;
%         disp("The model has correctly labelled the given test data");
    else
%         disp("The model has incorrectly labelled the given test data");
    end
end

accuracy = sum(isCorrect)/length(isCorrect)
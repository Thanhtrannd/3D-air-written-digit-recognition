function [traindata, trainclass, testdata, testclass] = data_Holdout_splitting(data, class, testdataproportion)
%data_Holdout_splitting.m splits data into 2 parts (train sets and test
%sets) ensuring same proportion of samples of each class per set as the
%proportion of each set in the input data set.
%Input:
%   data: dataset(rows representing features and columns representing
%   samples)
%   class: sample label of the given data (1 row and as much column as the
%   number of samples)
%   testdataproportion: a number (less than 1) representing the proportion
%   of test set, the train set is the rest of the data.
%Output:
%   traindata: splitted train data set (rows representing features and 
%   columns representing samples)
%   trainclass: sample class of the train data set (1 row and as much 
%   column as the number of samples)
%   testdata: splitted test data set (rows representing features and 
%   columns representing samples)
%   testclass: sample class of the test data set (1 row and as much 
%   column as the number of samples)

if testdataproportion >= 1
    disp("Invalid input: test data proportion must be less than 1");
    traindata = [];
    trainclass = [];

    testdata = [];
    testclass = [];
else
    trainInd = [];
    testInd = [];
    
    % Loop through each class and split the indices to the given proportion
    for label = 0:length(class)
        % find indices of class in question
        classInds = find(class==label);
        % number of samples whose class is in question
        nIndclass = length(classInds);
        % random permutation of indices of elements of classInds
        randomInds = randperm(length(classInds));

        % number of samples with such class in test data
        nTestInd = round(testdataproportion*nIndclass);
        % number of samples with such class in train data
        nTrainInd = nIndclass - nTestInd; 

        % take test indices from vector of indices of class in question
        testlabelInd = classInds(randomInds(1:nTestInd));
        % take train indices from vector of indices of class in question
        trainlabelInd = classInds(randomInds(nTestInd+1:end)); 

        % store train indices of this class
        trainInd = [trainInd, trainlabelInd]; 
        % store test indices of this class
        testInd = [testInd, testlabelInd]; 
    end
    
    % Split data and class based on picked indices of each set
    traindata = data(:,trainInd);
    trainclass = class(trainInd);

    testdata = data(:,testInd);
    testclass = class(testInd);
end
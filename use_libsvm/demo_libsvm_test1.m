
% Classification type: Binary classification
% Parameter selection: Manually pick by the user
% Classification: on the separated test set
% Kernel: default (RBF)
% Data set: heart_scale
%
% Then plot the results vs their true class. In order to visualize the high
% dimensional data, we apply MDS to the 13D data and reduce the dimension
% to 2D

% Load training data
trainData = training;
trainLabel = training_label;

% Load test data set
testData = test;
testLabel = test_label; 

label_vector = cat(1,trainLabel,testLabel);
trainingIndex = zeros(size(label_vector,1),1); trainingIndex(1:size(trainLabel,1)) = 1;
testIndex = zeros(size(label_vector,1),1); testIndex(size(trainLabel,1)+1:size(label_vector,1)) = 1; 

% Train the SVM
model = svmtrain(trainLabel, trainData, '-b 1');
% Use the SVM model to classify the data
[predict_label, accuracy, prob_values] = svmpredict(testLabel, testData, model, '-b 1'); % test the training data


% ================================
% ===== Showing the results ======
% ================================

% Assign color for each class
colorList = generateColorList(2); % This is my own way to assign the color...don't worry about it
colorList = prism(100);

% true (ground truth) class
trueClassIndex = zeros(size(label_vector,1),1);
trueClassIndex(label_vector==1) = 2;
trueClassIndex(label_vector==0) = 1;
colorTrueClass = colorList(trueClassIndex,:);
% result Class
resultClassIndex = zeros(length(predict_label),1);
resultClassIndex(predict_label==1) = 2;
resultClassIndex(predict_label==0) = 1;
colorResultClass = colorList(resultClassIndex,:);

% Reduce the dimension to 2D
distanceMatrix = pdist(cat(1,training,test),'euclidean'); % computes the distance between objects in the data matrix cat(1,training,test)
newCoor = mdscale(distanceMatrix,2); % configuration of n points in distanceMatrix in 2 dimensions 

% Plot the whole data set
x = newCoor(:,1);
y = newCoor(:,2);
patchSize = 30; %max(prob_values,[],2);
colorTrueClassPlot = colorTrueClass;
figure; scatter(x,y,patchSize,colorTrueClassPlot,'filled');
title('whole data set');

% Plot the test data
x = newCoor(testIndex==1,1);
y = newCoor(testIndex==1,2);
patchSize = 80*max(prob_values,[],2);
colorTrueClassPlot = colorTrueClass(testIndex==1,:);
figure; hold on;
scatter(x,y,2*patchSize,colorTrueClassPlot,'o','filled');
scatter(x,y,patchSize,colorResultClass,'o','filled');

% Plot the training set
x = newCoor(trainingIndex==1,1);
y = newCoor(trainingIndex==1,2);
patchSize = 30;
colorTrueClassPlot = colorTrueClass(trainingIndex==1,:);
scatter(x,y,patchSize,colorTrueClassPlot,'o');
title('classification results');
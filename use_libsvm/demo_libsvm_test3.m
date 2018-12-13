%
% Classification type: Binary classification
% Parameter selection: More automatic way. Having multiple scale of
% parameter selection.
% Classification: on the separated test set
% Kernel: default (not specified)
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

% ###################################################################
% From here on, we do 3-fold cross validation on the train data set
% ###################################################################

% ###################################################################
% cross validation scale 1
% This is the big scale (rough)
% ###################################################################
stepSize = 1;
log2c_list = -20:stepSize:20;
log2g_list = -20:stepSize:20;

numLog2c = length(log2c_list);
numLog2g = length(log2g_list);
cvMatrix = zeros(numLog2c,numLog2g);
bestcv = 0;
for i = 1:numLog2c
    log2c = log2c_list(i);
    for j = 1:numLog2g
        log2g = log2g_list(j);
        % -v 3 --> 3-fold cross validation
        param = ['-q -v 3 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
        cv = svmtrain(trainLabel, trainData, param);
        cvMatrix(i,j) = cv;
        if (cv >= bestcv),
            bestcv = cv; bestLog2c = log2c; bestLog2g = log2g;
        end
        % fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
    end
end

disp(['CV scale1: best log2c:',num2str(bestLog2c),' best log2g:',num2str(bestLog2g),' accuracy:',num2str(bestcv),'%']);

% Plot the results
figure;
imagesc(cvMatrix); colormap('jet'); colorbar;
set(gca,'XTick',1:numLog2g)
set(gca,'XTickLabel',sprintf('%3.1f|',log2g_list))
xlabel('Log_2\gamma');
set(gca,'YTick',1:numLog2c)
set(gca,'YTickLabel',sprintf('%3.1f|',log2c_list))
ylabel('Log_2c');


% ###################################################################
% cross validation scale 2
% This is the medium scale
% ###################################################################
prevStepSize = stepSize;
stepSize = prevStepSize/2;
log2c_list = bestLog2c-prevStepSize:stepSize:bestLog2c+prevStepSize;
log2g_list = bestLog2g-prevStepSize:stepSize:bestLog2g+prevStepSize;

numLog2c = length(log2c_list);
numLog2g = length(log2g_list);
cvMatrix = zeros(numLog2c,numLog2g);
bestcv = 0;
for i = 1:numLog2c
    log2c = log2c_list(i);
    for j = 1:numLog2g
        log2g = log2g_list(j);
        % -v 3 --> 3-fold cross validation
        param = ['-q -v 3 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
        cv = svmtrain(trainLabel, trainData, param);
        cvMatrix(i,j) = cv;
        if (cv >= bestcv),
            bestcv = cv; bestLog2c = log2c; bestLog2g = log2g;
        end
        % fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
    end
end

disp(['CV scale2: best log2c:',num2str(bestLog2c),' best log2g:',num2str(bestLog2g),' accuracy:',num2str(bestcv),'%']);

% Plot the results
figure;
imagesc(cvMatrix); colormap('jet'); colorbar;
set(gca,'XTick',1:numLog2g)
set(gca,'XTickLabel',sprintf('%3.1f|',log2g_list))
xlabel('Log_2\gamma');
set(gca,'YTick',1:numLog2c)
set(gca,'YTickLabel',sprintf('%3.1f|',log2c_list))
ylabel('Log_2c');



% ###################################################################
% cross validation scale 3
% This is the small scale
% ###################################################################
prevStepSize = stepSize;
stepSize = prevStepSize/2;
log2c_list = bestLog2c-prevStepSize:stepSize:bestLog2c+prevStepSize;
log2g_list = bestLog2g-prevStepSize:stepSize:bestLog2g+prevStepSize;

numLog2c = length(log2c_list);
numLog2g = length(log2g_list);
cvMatrix = zeros(numLog2c,numLog2g);
bestcv = 0;
for i = 1:numLog2c
    log2c = log2c_list(i);
    for j = 1:numLog2g
        log2g = log2g_list(j);
        % -v 3 --> 3-fold cross validation
        param = ['-q -v 3 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
        cv = svmtrain(trainLabel, trainData, param);
        cvMatrix(i,j) = cv;
        if (cv >= bestcv),
            bestcv = cv; bestLog2c = log2c; bestLog2g = log2g;
        end
        % fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
    end
end

disp(['CV scale3: best log2c:',num2str(bestLog2c),' best log2g:',num2str(bestLog2g),' accuracy:',num2str(bestcv),'%']);

% Plot the results
figure;
imagesc(cvMatrix); colormap('jet'); colorbar;
set(gca,'XTick',1:numLog2g)
set(gca,'XTickLabel',sprintf('%3.1f|',log2g_list))
xlabel('Log_2\gamma');
set(gca,'YTick',1:numLog2c)
set(gca,'YTickLabel',sprintf('%3.1f|',log2c_list))
ylabel('Log_2c');

% ################################################################
% Test phase
% Use the parameters to classify the test set
% ################################################################
param = ['-q -c ', num2str(2^bestLog2c), ' -g ', num2str(2^bestLog2g), ' -b 1'];
bestModel = svmtrain(trainLabel, trainData, param);
[predict_label_cv, accuracy_cv, prob_values_cv] = svmpredict(testLabel, testData, bestModel, '-b 1'); % test the training data

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
resultClassIndex = zeros(length(predict_label_cv),1);
resultClassIndex(predict_label_cv==1) = 2;
resultClassIndex(predict_label_cv==0) = 1;
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
patchSize = 50*max(prob_values_cv,[],2);
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

figure(111); subplot(1,3,1); imagesc(testLabel); title('testLabel'); subplot(1,3,2); imagesc(predict_label); title('SVM classification result'); subplot(1,3,3); imagesc(predict_label_cv); title('3-fold cross validation');

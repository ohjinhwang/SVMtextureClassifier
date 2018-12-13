function [CA_test, CA_training decisValueWinner] = demo_libsvm_test12(numTrial, training, training_label, test, test_label)
% run svm on the data set with train and test set are separated
% <train,test>Data: N x D matrix, each row represent feature vector of an observation
% <train,test>Label: N x 1 matrix containing the label for each observation


for n = 1:1
% Load training data
trainData = training;%_filt1_n;
trainLabel = training_label;

% Load test data set
testData = test;%_filt1_n;
testLabel = test_label;

labelList = unique(trainLabel);
NClass = length(labelList);
[Ntrain D] = size(trainData);
if ~exist('run','var')
    run = [1:Ntrain]';
end

%%
% #######################
% Parameter selection
% #######################
% First we randomly pick some observations from the training set for parameter selection
tmp = randperm(Ntrain);
evalIndex = tmp(1:ceil(Ntrain/2));
evalData = trainData(evalIndex,:);
evalLabel = trainLabel(evalIndex,:);

% #######################
% Automatic Cross Validation
% Parameter selection using n-fold cross validation
% #######################
% ================================================================
% Note that the cross validation for parameter selection can use different
% number of fold. In tis example
% Ncv_param = 3 but
% Ncv_classif = 5;
% Also note that we don't have to specify the fold for cv for parameter
% selection as the algorithm will pick observations into each fold
% randomly.
% ================================================================
optionCV.stepSize = 5;
optionCV.c = 1;
optionCV.gamma = 1/D;
optionCV.stepSize = 5;
optionCV.bestLog2c = 0;
optionCV.bestLog2g = log2(1/D);
optionCV.epsilon = 0.005;
optionCV.Nlimit = 20;
optionCV.svmCmd = '-q -t 2 -s 0';
Ncv_param = 5;
% Ncv-fold cross validation cross validation

% Put the kernel Phi(data)
[bestc, bestg, bestcv] = automaticParameterSelection2(evalLabel, evalData, Ncv_param, optionCV);


%%

% #######################
% Train/Classification
% #######################
% Using the multi-class pairwise SVM
% If we have K classes, we will have K-choose-2 pairs
% For instance, if we have k=4, we will have the classification pairs:
% {(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)}
% And let's say that the labeling results for sample xi are
% {1 3 4 3 4 3}, we decide class 3 for the sample according to the majority vote scheme

% options:
% -s svm_type : set type of SVM (default 0)
% 	0 -- C-SVC
% 	1 -- nu-SVC
% 	2 -- one-class SVM
% 	3 -- epsilon-SVR
% 	4 -- nu-SVR
% -t kernel_type : set type of kernel function (default 2)
% 	0 -- linear: u'*v
% 	1 -- polynomial: (gamma*u'*v + coef0)^degree
% 	2 -- radial basis function: exp(-gamma*|u-v|^2)
% 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
% 	4 -- precomputed kernel (kernel values in training_set_file)
% -d degree : set degree in kernel function (default 3)
% -g gamma : set gamma in kernel function (default 1/num_features)
% -r coef0 : set coef0 in kernel function (default 0)
% -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
% -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
% -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
% -m cachesize : set cache memory size in MB (default 100)
% -e epsilon : set tolerance of termination criterion (default 0.001)
% -h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
% -b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
% -wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
% -v n: n-fold cross validation mode
% -q : quiet mode (no outputs)

% ================================
% Basic SVM
% ================================
cmd = [optionCV.svmCmd,' -b 1 -c ',num2str(bestc),' -g ',num2str(bestg)];
% Train the SVM
model = svmtrain(trainLabel, trainData, cmd);
% Use the SVM model to classify the data
[predictedLabel, accuracy, decisValueWinner] = svmpredict(testLabel, testData, model, '-b 1'); % run the SVM model on the test data

% % --- test and train combined in one function ----
acc = svmPredictAccuracy(testLabel, testData, trainLabel, trainData, cmd);

% % N-cross validation
[outputLabel, accuracy, outputValue] = svmNFoldCrossValidation(trainLabel, trainData, 5,cmd);
CA_training(n,1) = accuracy*100;
TP = 0; TN = 0; FP = 0; FN = 0;
for m = 1:size(outputLabel,1)
   
    if (outputLabel(m,1) == 1 && training_label(m,1) == 1)
        TN = TN + 1;
    else if (outputLabel(m,1) == 2 && training_label(m,1) == 2)
            TP = TP + 1;
        else if (outputLabel(m,1) == 1 && training_label(m,1) == 2)
                FN = FN + 1;
            else
                FP = FP + 1;
            end
        end
    end
    
end

CA_training(n,2) = TP/(TP + FN)*100;
CA_training(n,3) = TN/(TN + FP)*100;

testLabel_int(testLabel == 0,1) = 1;
testLabel_int(testLabel == 1,1) = 2;
predictedLabel_int(predictedLabel == 0,1) = 1;
predictedLabel_int(predictedLabel == 1,1) = 2;

%%
% #######################
% Make confusion matrix for the overall classification
% #######################
[confusionMatrixAll,orderAll] = confusionmat(testLabel,predictedLabel);
figure(1); imagesc(confusionMatrixAll');
xlabel('actual class label');
ylabel('predicted class label');
title(['confusion matrix for overall classification']);
% Calculate the overall accuracy from the overall predicted class label
accuracyAll = trace(confusionMatrixAll)/sum(confusionMatrixAll(:));
disp(['Total accuracy is ',num2str(accuracyAll*100),'%']);
CA_test(n,1) = accuracyAll*100;


% Compare the actual and predicted class
figure(2);
subplot(1,2,1); imagesc(testLabel); title('actual class');
subplot(1,2,2); imagesc(predictedLabel); title('predicted class');

%%
% ################################
% Plot the clustering results in 2D
% ################################
% Pick the 2D representation to plot
data = testData;
if D==2
    data2D = data(:,1:2);
elseif D>2
    % Dimensionality reduction to 2D
    
    %     % ******** Using MDS (Take longer time)
    distanceMatrix = pdist(data,'euclidean');
    data2D = mdscale(distanceMatrix,2);
    
    % ******** Using classical MDS (Pretty short time)
    %     distanceMatrix = pdist(data,'euclidean');
    %     data2D = cmdscale(distanceMatrix); data2D = data2D(:,1:2);
end
% plot the true label for the test set
tmp = min(exp(zscore(decisValueWinner)),10);
tmp = tmp-min(tmp(:))+1;
tmp = tmp/max(tmp);

patchSize = 200*tmp;
colorList = generateColorList(NClass);
colorPlot = colorList(testLabel,:);
figure(3);
scatter(data2D(:,1),data2D(:,2),patchSize, colorPlot,'filled'); hold on;

% plot the predicted labels for the test set
patchSize = patchSize/2;
colorPlot = colorList(predictedLabel,:);
scatter(data2D(:,1),data2D(:,2),patchSize, colorPlot,'filled');

%%
% #######################
% Get the weights and bias from the trained model
% #######################

% Prepare helpful matrices for support vector matrix
numSV = model.nSV; % size = the number of classes
numSV_end = cumsum(numSV);
numSV_begin = numSV_end-numSV+1;

% Prepare useful matrices
cnt = 1;
W = zeros(NClass*(NClass-1)/2,D);
B = zeros(NClass*(NClass-1)/2,1);
for c1 = 1:NClass
    for c2 = (c1+1):NClass
        % the weight of class c1 vs class c2
        coef = [model.sv_coef(numSV_begin(c1):numSV_end(c1),c2-(c1<c2));...
            model.sv_coef(numSV_begin(c2):numSV_end(c2),c1-(c2<c1))];
        SVs = [model.SVs(numSV_begin(c1):numSV_end(c1),:);...
            model.SVs(numSV_begin(c2):numSV_end(c2),:)];
        w = SVs'*coef;
        
        % This is how to convert (c1,c2) to the order of (1,2) (1,3) (1,4)
        % (2,3) (2,4)
        tmp = zeros(NClass,NClass);
        tmp(c1,c2) = 1;
        tmp(c2,c1) = 1;
        
        % Get the bias term
        b = -model.rho(squareform(tmp)==1);
        
        % Store the weight matrix W and the bias matrix B
        W(cnt,:) = w(:)';
        B(cnt,:) = b;
        
        cnt = cnt + 1;
        
    end
end

%%
% #######################
% Plot the decision boundary on top of the predicted labels
% #######################
% plot the predicted labels for the test set
figure(4);
patchSize = 2*patchSize;
colorPlot = colorList(predictedLabel,:);
scatter(data2D(:,1),data2D(:,2),patchSize, colorPlot,'filled'); hold on;
daspect([1 1 1]);

% Plot the decision boundary
x_plot = [min(data2D(:,1)) max(data2D(:,1))];
for c = 1:size(W,1)
    w = W(c,:);
    b = B(c,:);
    y_plot = (1/w(2))*(-w(1)*x_plot+0-b);
    plot(x_plot,y_plot,'k-');
end
axis( [min(data2D(:,1)), max(data2D(:,1)), min(data2D(:,2)), max(data2D(:,2))]);

Test_label(:,1) = test_label;
Test_label(:,2) = predictedLabel;
error = zeros(size(test_label,1),1); a = 1;
for q = 1:size(test_label,1)
    
    if (test_label(q,1) ~= Test_label(q,2))
        
        error(q,1) = 1;
%        error_sub{a,1} = test_data2{q,1};
a = a + 1;

    end
    
end


[X_svm,Y_svm,T_svm,AUC_svm] = perfcurve(test_label,decisValueWinner(:,1),1);
Xsvm(:,n) = X_svm;
Ysvm(:,n) = Y_svm;
AUCsvm(:,n) = AUC_svm;

CA_test(n,2) = confusionMatrixAll(2,2)/(confusionMatrixAll(2,2)+confusionMatrixAll(2,1))*100; % SE = (TP/(TP+FN))
CA_test(n,3) = confusionMatrixAll(1,1)/(confusionMatrixAll(1,1)+confusionMatrixAll(1,2))*100; % SP = (TN/(TN+FP)) 

% TN = size(find(test_label == 1),1); % class same as training set (negative = same as training set)
% TP = size(find(test_label == 2),1); % class different from training set (positive = different from training set)
% FN = size(find(test_label - predictedLabel == 1),1); % should be same class as training set but falsely classified as different class (should be 2 but predicted as 1)
% FP = size(find(test_label - predictedLabel == -1),1); % should be different class from training set but falsely classified as same class as training set (should be 1 but predicted as 2)

clear B D NClass Ncv_param Ntrain SVs W a acc accuracy accuracyAll b bestc bestcv bestg c c1 c2 cmd cnt coef colorList colorPlot data data2D;
clear distanceMatrix error evalData evalIndex evalLabel labelList model n numSV numSV_begin numSV_end o optionCV orderAll outputLabel outputValue p patchSize;
clear q run tmp w x_plot y_plot Test_label;



%% additional code (Eo-Jin Hwang)

% 1. all slices 
    
% test_label(:,2) = predictedLabel;
% error = zeros(size(test_label,1),1); a = 1;
% for q = 1:size(test_label,1)
%     
%     if (test_label(q,1) ~= test_label(q,2))
%         
%         error(q,1) = 1;
%         error_sub{a,1} = test_data2{q,1};
%         a = a + 1;
%         
%     end
%     
% end
% 
% meanCA = mean(CA(:));
% stdCA = std(CA(:));

% 2. slice 

% for n = 1:size(predictedLabel,1)
%     
%     temp(n,1) =  ceil(n/9);
%     
% end
% templist = unique(temp);
% for n = 1:size(predictedLabel,1)
%     
%     for m = 1:size(templist,1)
%         
%         eval(m,1) = mean(predictedLabel(temp == m,1));
%         test_label_sub(m,1) = mean(test_label(temp == m,1));
%         
%     end
%     
% end
% 
% predictedLabel_sub = round(eval);
% test_label_sub(:,2) = predictedLabel_sub;
% error = zeros(size(test_label_sub,1),1); a = 1;
% for q = 1:size(test_label_sub,1)
%     
%     if (test_label_sub(q,1) ~= test_label_sub(q,2))
%         
%         error(q,1) = 1;
%         error_sub{a,1} = test_ind_2{q,1};
%         a = a + 1;
%         
%     end
%     
% end


end

% CA_test = mean(CA_test); 
% CA_training = mean(CA_training); 
% 
% Xsvm_mean = mean(Xsvm,2);
% Ysvm_mean = mean(Ysvm,2);
% figure(5); plot(Xsvm_mean,Ysvm_mean);
% xlabel('False positive rate'); ylabel('True positive rate');
% title('ROC curve for one class SVM'); 
% hold off; 
% 
% 
% x(:,1) = decisValueWinner(:,1);
% x(:,2) = testLabel-1; 
% ROCout=roc(x,0,0.05,1);

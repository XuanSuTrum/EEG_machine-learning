clear;
clc;

data = load('E:\数据集\SEED_VIG\DE\1.mat');
label = load('E:\数据集\SEED_VIG\perclos_labels\1.mat');

data = data.de_LDS;
data = reshape(data,885,17,25);
label = label.perclos;

for i = 1:885
    if label(i)<=0.35
        label(i)=0;
    else 
        label(i)=1;
    end
end 

%% 划分数据集
c = cvpartition(size(data,1),'Holdout',0.2);
%获取训练集和测试集的索引
train_indices = c.training;
test_indices  = c.test;
%使用索引从数据集中分离出训练数据集和测试数据集
train_data = data(train_indices,:);
train_label = label(train_indices);

test_data = data(test_indices,:);
test_label = label(test_indices,:);

%%
all_samples = train_data;
all_labels = train_label;
K = 10; % K-fold CV
indices = crossvalind('Kfold',all_labels,K); % generate indices for CV

for k = 1:K % K iterations
    cv_test_idx = find(indices == k); % indices for test samples in one trial of validation
    cv_train_idx = find(indices ~= k); % indices for training samples in one trial of validation
    cv_classout = classify(all_samples(cv_test_idx,:),all_samples(cv_train_idx,:),all_labels(cv_train_idx));
    cv_acc(k) = mean(cv_classout==all_labels(cv_test_idx)); % compute accuracy
    TP = sum((cv_classout==all_labels(cv_test_idx))&(cv_classout==1));
    TN = sum((cv_classout==all_labels(cv_test_idx))&(cv_classout==0));
    FP = sum((cv_classout~=all_labels(cv_test_idx))&(cv_classout==1));
    FN = sum((cv_classout~=all_labels(cv_test_idx))&(cv_classout==0));
    cv_sensitivity(k) = TP/(TP+FN); % compute specificity
    cv_specificity(k) = TN/(TN+FP); % compute sensitivity
end
cv_acc_avg = mean(cv_acc); % averaged accuracy
cv_sensitivity_avg = mean(cv_sensitivity);  % averaged sensitivity
cv_specificity_avg = mean(cv_specificity);  % averaged specificity

%%
train_samples = train_data;
train_labels = train_label;
test_samples = test_data;
test_labels = test_label;
N_Test = 177;

classout = classify(test_samples,train_samples,train_labels,'linear');
TP_test = sum((classout==test_labels)&(classout==1));
TN_test = sum((classout==test_labels)&(classout==0));
FP_test = sum((classout~=test_labels)&(classout==1));
FN_test = sum((classout~=test_labels)&(classout==0));
test_acc = sum(classout==test_labels)/N_Test; % compute accuracy
test_sensitivity = TP_test/(TP_test+FN_test); % compute specificity
test_specificity = TN_test/(TN_test+FP_test); % compute sensitivity

clc;
clear;

fs = 200;
time_point = 8;
Epoch = 885;
channel_number = 17;

raw_data = load('E:\数据集\SEED_VIG\Raw_Data\1.mat');
label = load('E:\数据集\SEED_VIG\perclos_labels\1.mat');

raw_data = raw_data.EEG.data;
label = label.perclos;

for i = 1:885
    if label(i)<=0.35
        label(i)=0;
    elseif (label(i) >= 0.35) && (label(i) < 0.7)
        label(i)=0;
    else
        label(i)=1;
    end

end 

windowSize = fs*time_point; % 窗口大小
stepSize = windowSize; % 步长

% 计算窗口数
numWindows = floor((size(raw_data, 1) - windowSize) / stepSize) + 1;
% 创建一个矩阵来存储窗口数据
B = zeros(fs*time_point, numWindows);
data = zeros(fs*time_point,Epoch,channel_number);

for j = 1:channel_number
x = raw_data(:,j);
for i = 1:numWindows
    idx = (i-1)*windowSize + (1:stepSize);
    B(:,i) = x(idx);
end 
    data(:,:,j) = B;
end 

fs = 200;
fs2=fs/2;                          % 设置奈奎斯特频率
W0=50/fs2;                         % 陷波器中心频率50Hz
BW=0.1;                            % 陷波器带宽 
[b,a]=iirnotch(W0,BW);             % 设计IIR数字陷波器
[H,w]=freqz(b,a);                  % 求出滤波器的频域响应
y1=filter(b,a,data);           % 对信号滤波

fl = 0.1;
fh = 30;
wp=[fl/(fs/2) fh/(fs/2)];
N=5; 
b=fir1(N,wp,blackman(N+1)); 
y2 = filtfilt(b,1,y1);

N_data = size(y2,2); % number of training trials
nfft = 256; % Point of FFT
for j = 1:885
    y3(:,:) = y2(:,j,:);
    for n=1:17
        [P_x(:,n),f] = pwelch(detrend(y3(:,n)),[],[],nfft,fs); % calculate PSD for ec condition 每个样本的功率谱密度值 129*885
    end
    y4(j,:,:) = P_x;
end 

delta_idx = find((f<4)&(f>=1));  % frequency index of alpha band power
for i = 1:885
    y5(:,:) = y4(i,:,:);
    delta = mean(y5(delta_idx,:)); % extract alpha band power from eo
    delta_all(i,:) = delta;
end

theta_idx = find((f<8)&(f>=5));  % frequency index of alpha band power
for i = 1:885
    y5(:,:) = y4(i,:,:);
    theta = mean(y5(theta_idx,:)); % extract alpha band power from eo
    theta_all(i,:) = theta;
end

alpha_idx = find((f<12)&(f>=8));  % frequency index of alpha band power
for i = 1:885
    y5(:,:) = y4(i,:,:);
    alpha = mean(y5(alpha_idx,:)); % extract alpha band power from eo
    alpha_all(i,:) = alpha;
end

beta_idx = find((f<30)&(f>=12));  % frequency index of alpha band power
for i = 1:885
    y5(:,:) = y4(i,:,:);
    beta = mean(y5(beta_idx,:)); % extract alpha band power from eo
    beta_all(i,:) = beta;
end

gamma_idx = find((f<50)&(f>=30)); % frequency index of alpha band power
for i = 1:885
    y5(:,:) = y4(i,:,:);
    gamma = mean(y5(gamma_idx,:)); % extract alpha band power from eo
    gamma_all(i,:) = gamma;
end

all_rhythm_idx = find((f<50)&(f>=1));
for i = 1:885
    y5(:,:) = y4(i,:,:);
    all_rhythm = mean(y5(all_rhythm_idx,:)); % extract alpha band power from eo
    rhythm_all(i,:) = all_rhythm;
end


%% 划分数据集
data = gamma_all;
c = cvpartition(885,'Holdout',0.3);
%获取训练集和测试集的索引
train_indices = c.training;
test_indices  = c.test;
%使用索引从数据集中分离出训练数据集和测试数据集
train_data = data(train_indices,:);
train_label = label(train_indices);

test_data = data(test_indices,:);
test_label = label(test_indices,:);

Mdl = fitcdiscr(train_data, train_label);

Ypred = predict(Mdl, test_data);

% 计算分类准确率
accuracy = sum(Ypred == test_label)/numel(test_label);
disp(['Classification accuracy: ', num2str(accuracy)]);
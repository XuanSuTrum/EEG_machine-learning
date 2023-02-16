clc;
clear;

fs = 10;
time_point = 2;
Epoch = 5;
channel_number = 1;

raw_data = randn(fs*time_point*Epoch,channel_number);

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



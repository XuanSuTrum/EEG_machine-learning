clc;
clear;

fs = 200;
time_point = 8;
Epoch = 885;
channel_number = 17;

data_folder = 'E:\数据集\SEED_VIG\Raw_Data';   % 设置文件夹路径
data_files = dir(fullfile(data_folder, '*.mat'));  % 获取文件夹中所有的.txt文件
data_num_files = length(data_files);   % 获取文件数量
% 使用cell数组存储每个导入的文件和其对应的名称
data_cells = cell(data_num_files, 2); 

for i = 1:data_num_files
    data_file_path = fullfile(data_folder, data_files(i).name);  % 获取当前文件的完整路径
    raw_data = load(data_file_path).EEG.data;  % 导入当前文件的数据
    % 在这里进行数据分析操作
    windowSize = fs*time_point; % 窗口大小
    stepSize = windowSize; % 步长

    % 计算窗口数
    numWindows = floor((size(raw_data, 1) - windowSize) / stepSize) + 1;
    % 创建一个矩阵来存储窗口数据
    B = zeros(fs*time_point, numWindows);
    data = zeros(fs*time_point,Epoch,channel_number);
    for j = 1:channel_number
    x = raw_data(:,j);
    for m = 1:numWindows
        idx = (m-1)*windowSize + (1:stepSize);
        B(:,m) = x(idx);
    end 
        data(:,:,j) = B;
    end 
    
    fs = 200;
    fs2=fs/2;                          % 设置奈奎斯特频率
    W0=50/fs2;                         % 陷波器中心频率50Hz
    BW=0.1;                            % 陷波器带宽 
    [b,a]=iirnotch(W0,BW);             % 设计IIR数字陷波器
    [H,w]=freqz(b,a);                  % 求出滤波器的频域响应
    y1=filter(b,a,data);               % 对信号滤波

    fl = 0.1;
    fh = 50;
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
        %all_rhythm_idx = find((f<50)&(f>=1));
        delta_idx = find((f<4)&(f>=1));
        theta_idx = find((f<8)&(f>=5));
        alpha_idx = find((f<12)&(f>=8));
        beta_idx = find((f<30)&(f>=12));
        gamma_idx = find((f<50)&(f>=30));

    for h = 1:885
        y5(:,:) = y4(h,:,:);
        delta = mean(y5(delta_idx,:));
        theta = mean(y5(theta_idx,:));
        alpha = mean(y5(alpha_idx,:));
        beta  = mean(y5(beta_idx,:));
        gamma = mean(y5(gamma_idx,:));

        feature0 = (theta+alpha)./beta;            %0
        feature1 = alpha./beta;                    %1
        feature2 = (theta+alpha)./(alpha+beta);    %2
        feature3 = (theta)./(beta);                %3
        feature4 = (beta)./(gamma);                %4
        feature5 = alpha./(delta+theta+alpha+beta+gamma);%5
        feature6 = delta./(delta+theta+alpha+beta+gamma);%6
        feature7 = beta./(delta+theta+alpha+beta+gamma);%7
        
        feature8 = (delta+theta+alpha+beta+gamma)./beta;
        feature9 = delta./beta;
        feature10 = gamma./beta;
        feature11 = (alpha+gamma)./beta;
        
        feature = vertcat(feature0,feature1,feature2,feature3,feature4,feature5,feature6,feature7, ...
            feature8,feature9,feature10,feature11);
        %all_rhythm = mean(y5(all_rhythm_idx,:)); % extract alpha band power from eo
        feature_all(h,:,:) = feature;
    end
    
    data_name = ['data', num2str(i)];
    % 将当前数据和名称存储在cell数组中
    data_cells{i, 1} = feature_all;
    data_cells{i, 2} = data_name;
    
end 
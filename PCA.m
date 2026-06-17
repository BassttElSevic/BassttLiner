% PCA

file_list = dir('D:\Linear_algebra\Face\Face_old\*.jpg');
output_folder = 'D:\Linear_algebra\Face\Face_neo';
eigen_folder = 'D:\Linear_algebra\Face\Eigenfaces';
% ===== 清空输出文件夹，防止重复累积 =====
if exist(output_folder, 'dir')
    % 方法1：删除文件夹内所有文件和子文件夹（速度较快）
    delete(fullfile(output_folder, '*'));   % 删除所有文件
    % 如果有子文件夹也可以一并删除（根据需要取消注释下一行）
    % rmdir(fullfile(output_folder, '*'), 's');
    fprintf('已清空输出文件夹: %s\n', output_folder);
else
    % 如果文件夹不存在，则创建
    mkdir(output_folder);
    fprintf('已创建输出文件夹: %s\n', output_folder);
end

if exist(eigen_folder, 'dir')
    % 方法1：删除文件夹内所有文件和子文件夹（速度较快）
    delete(fullfile(eigen_folder, '*'));   % 删除所有文件
    % 如果有子文件夹也可以一并删除（根据需要取消注释下一行）
    % rmdir(fullfile(output_folder, '*'), 's');
    fprintf('已清空特征脸文件夹: %s\n', eigen_folder);
else
    % 如果文件夹不存在，则创建
    mkdir(eigen_folder);
    fprintf('已创建特征脸文件夹: %s\n', eigen_folder);
end



n = length(file_list);

images = cell(1, n);
PR_of_images = cell(1, n);
PG_of_images = cell(1, n);
PB_of_images = cell(1, n);
RAW_gray_images = cell(1, n);
Gauss_filtered = cell(1, n);
Resized_images = cell(1, n);
Gauss_filtered_Resized_images = cell(1,n);

valid_count = 0;  % 记录成功处理的图片数量

% ---- 循环批量处理 ----
for i = 1:n
    filepath = fullfile(file_list(i).folder, file_list(i).name);
    
    try
        img = imread(filepath);
        
        % 检查是否为彩色图（3通道）
        if size(img, 3) < 3
            fprintf('⚠️ 跳过非彩色图: %s\n', file_list(i).name);
            continue;  % 跳过本次循环，进入下一张
        end
        
        valid_count = valid_count + 1;
        
        images{valid_count} = img;
        
        % ===== 这里是你原来的 bug =====
        % 你之前没有给 PR/PG/PB 赋值就直接用了
        % 需要先提取分量，再做灰度化
        PR_of_images{valid_count} = double(img(:,:,1));
        PG_of_images{valid_count} = double(img(:,:,2));
        PB_of_images{valid_count} = double(img(:,:,3));
        
        % 加权灰度化
        RAW_gray_images{valid_count} = uint8( ...
            0.299 * PR_of_images{valid_count} + ...
            0.587 * PG_of_images{valid_count} + ...
            0.114 * PB_of_images{valid_count});
        
        % 高斯滤波
        Gauss_filtered{valid_count} = imgaussfilt(RAW_gray_images{valid_count}, 0.3);
        
        % 尺寸统一化（去掉第二次高斯滤波，避免过度模糊丢失细节）
        Resized_images{valid_count} = imresize(Gauss_filtered{valid_count}, [60, 60]);
        % 写入D:\Linear_algebra\Face\Face_neo
        [~, name, ~] = fileparts(file_list(i).name);
        out_path = fullfile(output_folder, [name, '.jpg']);
        imwrite(Resized_images{valid_count}, out_path);


        fprintf('成功处理: %s\n', file_list(i).name);
        
    catch e
        fprintf('跳过损坏图片: %s (%s)\n', file_list(i).name, e.message);
        % 不增加 valid_count，直接进入下一张
    end
end

% 清理空 cell
images = images(1:valid_count);
PR_of_images = PR_of_images(1:valid_count);
PG_of_images = PG_of_images(1:valid_count);
PB_of_images = PB_of_images(1:valid_count);
RAW_gray_images = RAW_gray_images(1:valid_count);
Gauss_filtered = Gauss_filtered(1:valid_count);
Resized_images = Resized_images(1:valid_count);

%% ===== 1. 准备中心化数据 X =====
data_matrix = zeros(60*60, valid_count);
for k = 1:valid_count
    data_matrix(:, k) = double(Resized_images{k}(:));
end
mean_face = mean(data_matrix, 2);
X = data_matrix - mean_face;   % X: n×m, 每列已去均值
[n, m] = size(X);              % n = 3600, m = valid_count

fprintf('中心化完成：X 尺寸 = %d × %d\n', n, m);

%% ===== 2. 计算协方差矩阵 C = (1/m) * X * X' =====
C = (X * X') / m;
fprintf('协方差矩阵 C 尺寸 = %d × %d\n', size(C));

%% ===== 3. 手工求正交阵 Q 和对角阵 Lambda（幂迭代 + 收缩 + 正交化）=====
r_max = min(n, m);
tol = 1e-8;
max_iter = 1000;

Q = zeros(n, r_max);
lambda = zeros(r_max, 1);
C_curr = C;

for j = 1:r_max
    % 初始化随机向量
    v = randn(n, 1);
    v = v / norm(v);
    % 与已求得的特征向量正交（保证 Q 最终正交）
    if j > 1
        v = v - Q(:, 1:j-1) * (Q(:, 1:j-1)' * v);
        v = v / norm(v);
    end
    
    lambda_old = 0;
    for iter = 1:max_iter
        w = C_curr * v;
        % 再次正交化
        if j > 1
            w = w - Q(:, 1:j-1) * (Q(:, 1:j-1)' * w);
        end
        lambda_new = v' * w;
        v = w / norm(w);
        if abs(lambda_new - lambda_old) < tol
            break;
        end
        lambda_old = lambda_new;
    end
    
    if lambda_new < 1e-10
        fprintf('特征值趋于零，提前停止（已找到 %d 个非零特征向量）\n', j-1);
        break;
    end
    
    lambda(j) = lambda_new;
    Q(:, j) = v;
    
    % Hotelling 收缩
    C_curr = C_curr - lambda_new * (v * v');
    C_curr = (C_curr + C_curr') / 2;
    fprintf('完成第 %d 张 \n', j);
end

r_actual = j;
lambda = lambda(1:r_actual);
Q = Q(:, 1:r_actual);

% 全局正交化（消去累积误差）
[Q, ~] = qr(Q, 0);
for i = 1:r_actual
    lambda(i) = Q(:, i)' * C * Q(:, i);
end

% 降序排列
[lambda_sorted, idx] = sort(lambda, 'descend');
Q = Q(:, idx);
Lambda = diag(lambda_sorted);

fprintf('手工特征分解完成，前 5 个特征值：\n');
disp(lambda_sorted(1:min(5, r_actual))');

% 验证正交性
I_err = max(abs(Q'*Q - eye(r_actual)), [], 'all');
fprintf('Q 正交性误差 (max|Q''Q - I|) = %e\n', I_err);

% 验证对角化
L_check = Q' * C * Q;
off_diag_err = max(abs(L_check - diag(diag(L_check))), [], 'all');
fprintf('Λ 非对角元最大值 = %e\n', off_diag_err);

%% ===== 4. 选择前 r 个特征向量作为投影矩阵 P =====
explained = lambda_sorted / sum(lambda_sorted) * 100;
cum_explained = cumsum(explained);
r = find(cum_explained >= 95, 1);
if isempty(r)
    r = r_actual;
end
fprintf('保留 95%% 方差所需主成分数 r = %d\n', r);

P = Q(:, 1:r);   % 投影矩阵，列向量即特征脸

%% ===== 5. 降维 Y = P^T X =====
Y = P' * X;
fprintf('降维后数据 Y 尺寸 = %d × %d\n', size(Y));

%% ===== 6. 验证降维后协方差矩阵 D 是对角阵 =====
D = (Y * Y') / m;
D_theory = diag(lambda_sorted(1:r));
fprintf('D 与 diag(λ₁...λᵣ) 的最大差异 = %e\n', max(abs(D - D_theory), [], 'all'));

%% ===== 7. 保存特征脸（即 P 的列向量） =====
img_size = [60, 60];
for i = 1:size(P, 2)
    img = reshape(P(:, i), img_size);
    img_uint8 = uint8(255 * mat2gray(img));
    imwrite(img_uint8, fullfile(eigen_folder, sprintf('eigenface_%03d.jpg', i)));
end
fprintf('已保存 %d 张特征脸到 %s\n', size(P,2), eigen_folder);

%% ===== 8. 可视化（与原来类似） =====
% 显示平均脸
figure('Name', '平均脸', 'NumberTitle', 'off');
mean_face_img = reshape(mean_face, img_size);
imshow(mean_face_img, []);
title('平均脸');

% 显示前 20 张特征脸
num_show = min(20, size(P, 2));
figure('Name', '特征脸 (Eigenfaces)', 'NumberTitle', 'off');
for i = 1:num_show
    subplot(4, 5, i);
    imshow(reshape(P(:, i), img_size), []);
    title(sprintf('特征脸 %d', i));
end

% 平均脸 ± 第1特征脸
alpha = 2;
face_plus  = mean_face + alpha * sqrt(lambda_sorted(1)) * P(:, 1);
face_minus = mean_face - alpha * sqrt(lambda_sorted(1)) * P(:, 1);

figure('Name', '平均脸 ± 第1特征脸', 'NumberTitle', 'off');
subplot(1,3,1); imshow(reshape(mean_face, img_size), []); title('平均脸');
subplot(1,3,2); imshow(reshape(face_plus, img_size), []);  title('平均脸 + 2σ(PC1)');
subplot(1,3,3); imshow(reshape(face_minus, img_size), []); title('平均脸 - 2σ(PC1)');
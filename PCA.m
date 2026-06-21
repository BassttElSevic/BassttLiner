% PCA
% =========================================================================
% 【新方案】预处理已移至 face_align_augment.m 脚本
% 请先运行 face_align_augment.m 完成人脸对齐+数据增强，再运行本脚本。
% 本脚本直接从 Face_neo 读取已经对齐、增强好的 60×60 灰度图。
%
% 如果新方案效果不好，想恢复旧代码：
%   1. 删除本段新代码（到 "旧预处理代码结束" 标记为止）
%   2. 取消下方旧代码的注释
% =========================================================================

%% ===== 新方案：直接从 Face_neo 读取已处理好的图片 =====
output_folder = 'D:\Linear_algebra\Face\Face_neo';
eigen_folder  = 'D:\Linear_algebra\Face\Eigenfaces';

% 清空特征脸文件夹
if exist(eigen_folder, 'dir')
    delete(fullfile(eigen_folder, '*'));
    fprintf('已清空特征脸文件夹: %s\n', eigen_folder);
else
    mkdir(eigen_folder);
    fprintf('已创建特征脸文件夹: %s\n', eigen_folder);
end

% 从 Face_neo 读取所有已处理的图片（face_align_augment.m 的输出）
file_list = dir(fullfile(output_folder, '*.jpg'));
n = length(file_list);
fprintf('从 Face_neo 读取到 %d 张已处理图片\n', n);

if n == 0
    error('Face_neo 文件夹为空！请先运行 face_align_augment.m 进行预处理。');
end

Resized_images = cell(1, n);
valid_count = 0;

for i = 1:n
    filepath = fullfile(file_list(i).folder, file_list(i).name);
    try
        img = imread(filepath);
        % Face_neo 中的图片已经是 60×60 灰度图
        if size(img, 3) == 3
            img = uint8(0.299*double(img(:,:,1)) + ...
                        0.587*double(img(:,:,2)) + ...
                        0.114*double(img(:,:,3)));
        end
        % 确保尺寸正确
        if size(img,1) ~= 60 || size(img,2) ~= 60
            img = imresize(img, [60, 60]);
        end
        valid_count = valid_count + 1;
        Resized_images{valid_count} = img;
    catch e
        fprintf('跳过: %s (%s)\n', file_list(i).name, e.message);
    end
end

Resized_images = Resized_images(1:valid_count);
fprintf('成功加载 %d 张图片\n', valid_count);

% =========================================================================
% 【旧预处理代码 — 已注释保留，如需恢复请取消注释】
% =========================================================================
% % --- 旧方案：从 Face_old 直接读取并处理 ---
% file_list = dir('D:\Linear_algebra\Face\Face_old\*.jpg');
% output_folder = 'D:\Linear_algebra\Face\Face_neo';
% eigen_folder = 'D:\Linear_algebra\Face\Eigenfaces';
% % ===== 清空输出文件夹，防止重复累积 =====
% if exist(output_folder, 'dir')
%     delete(fullfile(output_folder, '*'));
%     fprintf('已清空输出文件夹: %s\n', output_folder);
% else
%     mkdir(output_folder);
%     fprintf('已创建输出文件夹: %s\n', output_folder);
% end
%
% if exist(eigen_folder, 'dir')
%     delete(fullfile(eigen_folder, '*'));
%     fprintf('已清空特征脸文件夹: %s\n', eigen_folder);
% else
%     mkdir(eigen_folder);
%     fprintf('已创建特征脸文件夹: %s\n', eigen_folder);
% end
%
% n = length(file_list);
%
% images = cell(1, n);
% PR_of_images = cell(1, n);
% PG_of_images = cell(1, n);
% PB_of_images = cell(1, n);
% RAW_gray_images = cell(1, n);
% Gauss_filtered = cell(1, n);
% Resized_images = cell(1, n);
% Gauss_filtered_Resized_images = cell(1,n);
%
% valid_count = 0;
%
% for i = 1:n
%     filepath = fullfile(file_list(i).folder, file_list(i).name);
%     try
%         img = imread(filepath);
%         if size(img, 3) < 3
%             fprintf('⚠️ 跳过非彩色图: %s\n', file_list(i).name);
%             continue;
%         end
%         valid_count = valid_count + 1;
%         images{valid_count} = img;
%         PR_of_images{valid_count} = double(img(:,:,1));
%         PG_of_images{valid_count} = double(img(:,:,2));
%         PB_of_images{valid_count} = double(img(:,:,3));
%         RAW_gray_images{valid_count} = uint8( ...
%             0.299 * PR_of_images{valid_count} + ...
%             0.587 * PG_of_images{valid_count} + ...
%             0.114 * PB_of_images{valid_count});
%         Gauss_filtered{valid_count} = imgaussfilt(RAW_gray_images{valid_count}, 0.3);
%         Resized_images{valid_count} = imresize(Gauss_filtered{valid_count}, [60, 60]);
%         [~, name, ~] = fileparts(file_list(i).name);
%         out_path = fullfile(output_folder, [name, '.jpg']);
%         imwrite(Resized_images{valid_count}, out_path);
%         fprintf('成功处理: %s\n', file_list(i).name);
%     catch e
%         fprintf('跳过损坏图片: %s (%s)\n', file_list(i).name, e.message);
%     end
% end
%
% images = images(1:valid_count);
% PR_of_images = PR_of_images(1:valid_count);
% PG_of_images = PG_of_images(1:valid_count);
% PB_of_images = PB_of_images(1:valid_count);
% RAW_gray_images = RAW_gray_images(1:valid_count);
% Gauss_filtered = Gauss_filtered(1:valid_count);
% Resized_images = Resized_images(1:valid_count);
% % --- 旧预处理代码结束 ---

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

%% ===== 4. 去除前3个主成分（光照干扰）并用交叉验证选最优维度 =====
skip_pc = 3;  % 去掉前3个主成分（主要捕获光照变化）
fprintf('去除前 %d 个主成分以消除光照干扰\n', skip_pc);

% --- 交叉验证选择最优保留维度 ---
% 候选维度范围：从第 skip_pc+1 个开始，测试 20~60 维
candidate_dims = [20, 25, 30, 35, 40, 45, 50, 55, 60];
candidate_dims = candidate_dims(candidate_dims + skip_pc <= r_actual);

% 提取文件名标签用于交叉验证
cv_labels = cell(valid_count, 1);
for ci = 1:valid_count
    fname = file_list(ci).name;
    tokens = regexp(fname, '^([\x{4e00}-\x{9fff}]{2,3})', 'tokens', 'once');
    if ~isempty(tokens)
        cv_labels{ci} = tokens{1};
    else
        cv_labels{ci} = sprintf('unknown_%d', ci);
    end
end

% 5折交叉验证
n_folds = 5;
rng(123);
cv_perm = randperm(valid_count);
fold_size = floor(valid_count / n_folds);

fprintf('\n===== 交叉验证选择最优PCA维度 =====\n');
cv_accuracies = zeros(length(candidate_dims), 1);

for di = 1:length(candidate_dims)
    dim = candidate_dims(di);
    fold_acc = zeros(n_folds, 1);
    
    for fold = 1:n_folds
        % 划分验证集和训练集
        if fold < n_folds
            val_idx = cv_perm((fold-1)*fold_size+1 : fold*fold_size);
        else
            val_idx = cv_perm((fold-1)*fold_size+1 : end);
        end
        tr_idx = setdiff(cv_perm, val_idx);
        
        % 投影（跳过前skip_pc个，保留dim个）
        P_cv = Q(:, skip_pc+1 : skip_pc+dim);
        Y_tr = P_cv' * X(:, tr_idx);
        Y_val = P_cv' * X(:, val_idx);
        labels_tr = cv_labels(tr_idx);
        labels_val = cv_labels(val_idx);
        
        % 1-NN 欧几里得距离
        correct = 0;
        for vi = 1:length(val_idx)
            diff_cv = Y_tr - Y_val(:, vi);
            dists = sum(diff_cv.^2, 1);
            [~, best] = min(dists);
            if strcmp(labels_tr{best}, labels_val{vi})
                correct = correct + 1;
            end
        end
        fold_acc(fold) = correct / length(val_idx);
    end
    
    cv_accuracies(di) = mean(fold_acc);
    fprintf('  维度 %d: 交叉验证准确率 = %.2f%%\n', dim, cv_accuracies(di)*100);
end

% 选择最优维度
[best_cv_acc, best_di] = max(cv_accuracies);
r = candidate_dims(best_di);
fprintf('交叉验证最优维度: %d（准确率 %.2f%%）\n', r, best_cv_acc*100);

% 最终投影矩阵：跳过前skip_pc个，保留r个
P = Q(:, skip_pc+1 : skip_pc+r);   % 投影矩阵，列向量即特征脸
fprintf('投影矩阵 P 使用第 %d 到第 %d 个特征向量\n', skip_pc+1, skip_pc+r);

%% ===== 5. 降维 Y = P^T X =====
Y = P' * X;
fprintf('降维后数据 Y 尺寸 = %d × %d\n', size(Y));

%% ===== 6. 验证降维后协方差矩阵 D 是对角阵 =====
D = (Y * Y') / m;
D_theory = diag(lambda_sorted(skip_pc+1 : skip_pc+r));
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
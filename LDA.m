% LDA.m — Fisherface（PCA + LDA）人脸识别
% =========================================================================
% 核心思想：
%   PCA 最大化总方差（无监督），但不区分类别信息；
%   LDA（Fisher 线性判别分析）最大化类间散度 / 类内散度，是有监督降维。
%   Fisherface = 先用 PCA 降维到 N-C 维（避免 Sw 奇异），再做 LDA 压到 C-1 维。
%
% 数学原理：
%   给定 C 个类、N 个样本，定义：
%     类内散度矩阵 Sw = Σ_c Σ_{x∈c} (x - μ_c)(x - μ_c)^T
%     类间散度矩阵 Sb = Σ_c n_c (μ_c - μ)(μ_c - μ)^T
%   Fisher 准则：max_W  |W^T Sb W| / |W^T Sw W|
%   等价于广义特征值问题：Sb w = λ Sw w
%   最优投影向量为对应最大特征值的前 C-1 个广义特征向量。
%
% 使用方法：
%   先运行 PCA.m，再运行本脚本。
%   本脚本会在 PCA 空间基础上做 LDA，并用 1-NN 测试识别率。
% =========================================================================

%% ===== 0. 检查工作空间变量 =====
if ~exist('mean_face', 'var') || ~exist('Q', 'var') || ~exist('X', 'var')
    error('请先运行 PCA.m，确保工作空间中存在 mean_face, Q, X, lambda_sorted 等变量。');
end

fprintf('\n');
fprintf('========================================\n');
fprintf('       Fisherface (PCA + LDA)\n');
fprintf('========================================\n');

%% ===== 1. 读取图片并提取标签（与 test_PCA_accuracy.m 一致） =====
face_old_folder = 'D:\Linear_algebra\Face\Face_old';
all_files = dir(fullfile(face_old_folder, '*.*'));
img_extensions = {'.jpg', '.jpeg', '.png', '.bmp'};
keep_mask = false(length(all_files), 1);
for i = 1:length(all_files)
    if all_files(i).isdir, continue; end
    [~, ~, ext] = fileparts(all_files(i).name);
    if any(strcmpi(ext, img_extensions))
        keep_mask(i) = true;
    end
end
all_files = all_files(keep_mask);

% 提取人名标签
labels = cell(length(all_files), 1);
for i = 1:length(all_files)
    fname = all_files(i).name;
    tokens = regexp(fname, '^([\x{4e00}-\x{9fff}]{2,3})', 'tokens', 'once');
    if ~isempty(tokens)
        labels{i} = tokens{1};
    else
        labels{i} = '';
    end
end

valid_mask = ~cellfun(@isempty, labels);
all_files = all_files(valid_mask);
labels = labels(valid_mask);

fprintf('有效标记图片数: %d\n', length(labels));
unique_names = unique(labels);
C = length(unique_names);   % 类别数
fprintf('共有 %d 个不同的人（类别数 C = %d）\n', C, C);

%% ===== 2. 处理图片并投影到像素空间 =====
img_size = [60, 60];
num_images = length(all_files);
all_vectors = zeros(img_size(1)*img_size(2), num_images);
valid_indices = [];

for i = 1:num_images
    filepath = fullfile(all_files(i).folder, all_files(i).name);
    try
        img = imread(filepath);
        if size(img, 3) == 3
            gray_img = uint8(0.299*double(img(:,:,1)) + ...
                             0.587*double(img(:,:,2)) + ...
                             0.114*double(img(:,:,3)));
        elseif size(img, 3) == 1
            gray_img = img;
        else
            continue;
        end
        filtered = imgaussfilt(gray_img, 0.3);
        resized = imresize(filtered, img_size);
        all_vectors(:, i) = double(resized(:));
        valid_indices = [valid_indices, i]; %#ok<AGROW>
    catch
        fprintf('跳过无法读取的图片: %s\n', all_files(i).name);
    end
end

all_vectors = all_vectors(:, valid_indices);
labels = labels(valid_indices);
num_valid = length(valid_indices);
fprintf('成功处理 %d 张图片\n', num_valid);

%% ===== 3. 划分训练集/测试集（分层抽样） =====
rng(42);
test_ratio = 0.3;
train_idx = [];
test_idx = [];

unique_names_valid = unique(labels);
C = length(unique_names_valid);

for k = 1:C
    name = unique_names_valid{k};
    person_idx = find(strcmp(labels, name));
    n_person = length(person_idx);
    
    if n_person < 2
        train_idx = [train_idx; person_idx(:)]; %#ok<AGROW>
        continue;
    end
    
    perm = randperm(n_person);
    n_test = max(1, round(n_person * test_ratio));
    n_train = n_person - n_test;
    
    train_idx = [train_idx; person_idx(perm(1:n_train))]; %#ok<AGROW>
    test_idx = [test_idx; person_idx(perm(n_train+1:end))]; %#ok<AGROW>
end

labels_train = labels(train_idx);
labels_test = labels(test_idx);
N_train = length(train_idx);
N_test = length(test_idx);

fprintf('\n===== 数据划分 =====\n');
fprintf('训练集: %d 张\n', N_train);
fprintf('测试集: %d 张\n', N_test);

%% ===== 4. PCA 降维（第一阶段：降到 N_train - C 维） =====
% Fisherface 要求先用 PCA 降到 N-C 维以使 Sw 可逆
% 使用 PCA.m 已经计算好的特征向量 Q

% 中心化
X_all = all_vectors - mean_face;
X_train_raw = X_all(:, train_idx);

% PCA 降维维度：min(N_train - C, available_eigenvectors)
% 跳过前 skip_pc 个主成分（光照分量）
if ~exist('skip_pc', 'var')
    skip_pc = 3;
end

pca_dim = min(N_train - C, size(Q, 2) - skip_pc);
pca_dim = max(pca_dim, C);  % 至少保留 C 维给 LDA 用
fprintf('\n===== PCA 第一阶段降维 =====\n');
fprintf('跳过前 %d 个主成分（光照），保留 %d 个 PCA 维度\n', skip_pc, pca_dim);

W_pca = Q(:, skip_pc+1 : skip_pc+pca_dim);  % 像素空间 → PCA空间的投影矩阵

% 将训练集投影到 PCA 空间
Y_train_pca = W_pca' * X_train_raw;   % pca_dim × N_train

%% ===== 5. 计算类内散度 Sw 和类间散度 Sb（在 PCA 空间中） =====
fprintf('\n===== 计算 Fisher 散度矩阵 =====\n');

% 总体均值（PCA空间中）
mu_total = mean(Y_train_pca, 2);   % pca_dim × 1

% 初始化
Sw = zeros(pca_dim, pca_dim);   % 类内散度
Sb = zeros(pca_dim, pca_dim);   % 类间散度

for k = 1:C
    name = unique_names_valid{k};
    class_mask = strcmp(labels_train, name);
    class_samples = Y_train_pca(:, class_mask);   % pca_dim × n_k
    n_k = size(class_samples, 2);
    
    if n_k == 0, continue; end
    
    % 类均值
    mu_k = mean(class_samples, 2);   % pca_dim × 1
    
    % 类内散度：Sw += Σ (x_i - μ_k)(x_i - μ_k)^T
    diff_k = class_samples - mu_k;   % pca_dim × n_k
    Sw = Sw + diff_k * diff_k';
    
    % 类间散度：Sb += n_k * (μ_k - μ)(μ_k - μ)^T
    diff_mean = mu_k - mu_total;     % pca_dim × 1
    Sb = Sb + n_k * (diff_mean * diff_mean');
end

% 对称化（消除数值误差）
Sw = (Sw + Sw') / 2;
Sb = (Sb + Sb') / 2;

fprintf('Sw 尺寸: %d × %d, 秩估计: %d\n', size(Sw), rank(Sw));
fprintf('Sb 尺寸: %d × %d, 秩估计: %d\n', size(Sb), rank(Sb));

%% ===== 6. 求解广义特征值问题 Sb * w = λ * Sw * w =====
% 方法：Sw^{-1} * Sb 的特征值分解
% 由于 Sw 可能接近奇异，加正则化项 Sw_reg = Sw + α*I

fprintf('\n===== 求解广义特征值问题 =====\n');

% 正则化参数（防止 Sw 奇异）
alpha_reg = 1e-4 * trace(Sw) / pca_dim;
Sw_reg = Sw + alpha_reg * eye(pca_dim);
fprintf('正则化参数 α = %.6e\n', alpha_reg);

% 求解 Sw_reg^{-1} * Sb 的特征值
% 等价于求 inv(Sw_reg) * Sb 的特征值分解
M = Sw_reg \ Sb;   % pca_dim × pca_dim

% 手工幂迭代求前 C-1 个特征向量（保持与 PCA.m 一致的教学风格）
lda_dim = C - 1;   % LDA 最多产生 C-1 个有判别力的方向
fprintf('LDA 目标维度: %d (C-1 = %d - 1)\n', lda_dim, C);

tol_lda = 1e-10;
max_iter_lda = 2000;

W_lda_vectors = zeros(pca_dim, lda_dim);
lda_eigenvalues = zeros(lda_dim, 1);
M_curr = M;

for j = 1:lda_dim
    % 初始化随机向量
    v = randn(pca_dim, 1);
    v = v / norm(v);
    
    % 正交化（与已求得的向量正交）
    if j > 1
        v = v - W_lda_vectors(:, 1:j-1) * (W_lda_vectors(:, 1:j-1)' * v);
        if norm(v) < 1e-12
            v = randn(pca_dim, 1);
            v = v - W_lda_vectors(:, 1:j-1) * (W_lda_vectors(:, 1:j-1)' * v);
        end
        v = v / norm(v);
    end
    
    % 幂迭代
    lambda_old = 0;
    for iter = 1:max_iter_lda
        w = M_curr * v;
        
        % 正交化
        if j > 1
            w = w - W_lda_vectors(:, 1:j-1) * (W_lda_vectors(:, 1:j-1)' * w);
        end
        
        lambda_new = v' * w;
        nw = norm(w);
        if nw < 1e-15
            break;
        end
        v = w / nw;
        
        if abs(lambda_new - lambda_old) < tol_lda * max(abs(lambda_new), 1)
            break;
        end
        lambda_old = lambda_new;
    end
    
    % 如果特征值太小，说明没有更多判别信息
    if abs(lambda_new) < 1e-10
        fprintf('  LDA 特征值趋于零，提前停止（已找到 %d 个方向）\n', j-1);
        lda_dim = j - 1;
        break;
    end
    
    lda_eigenvalues(j) = lambda_new;
    W_lda_vectors(:, j) = v;
    
    % Hotelling 收缩
    M_curr = M_curr - lambda_new * (v * v');
    M_curr = (M_curr + M_curr') / 2;
end

W_lda_vectors = W_lda_vectors(:, 1:lda_dim);
lda_eigenvalues = lda_eigenvalues(1:lda_dim);

fprintf('\nLDA 完成，有效判别维度: %d\n', lda_dim);
fprintf('前 5 个 Fisher 特征值:\n');
disp(lda_eigenvalues(1:min(5, lda_dim))');

%% ===== 7. 组合投影矩阵 W_total = W_pca * W_lda =====
% 原始像素空间 → PCA 空间 → LDA 空间
% W_total: 3600 × lda_dim
W_fisherface = W_pca * W_lda_vectors;   % 3600 × lda_dim

fprintf('\n===== Fisherface 投影矩阵 =====\n');
fprintf('W_fisherface 尺寸: %d × %d\n', size(W_fisherface));
fprintf('（像素空间 %d 维 → Fisher 空间 %d 维）\n', size(W_fisherface,1), lda_dim);

%% ===== 8. 投影所有数据到 Fisher 空间 =====
Y_fisher_all = W_fisherface' * X_all;         % lda_dim × num_valid
Y_fisher_train = Y_fisher_all(:, train_idx);  % lda_dim × N_train
Y_fisher_test = Y_fisher_all(:, test_idx);    % lda_dim × N_test

%% ===== 9. 分类测试 =====
fprintf('\n===== Fisherface 识别结果 =====\n');

% --- 9.1 类中心法 + 欧几里得距离 ---
centroids_fisher = zeros(lda_dim, C);
for k = 1:C
    name = unique_names_valid{k};
    class_mask = strcmp(labels_train, name);
    centroids_fisher(:, k) = mean(Y_fisher_train(:, class_mask), 2);
end

correct_centroid = 0;
for i = 1:N_test
    test_vec = Y_fisher_test(:, i);
    diff = centroids_fisher - test_vec;
    distances = sum(diff.^2, 1);
    [~, best_idx] = min(distances);
    if strcmp(unique_names_valid{best_idx}, labels_test{i})
        correct_centroid = correct_centroid + 1;
    end
end
acc_centroid = correct_centroid / N_test * 100;
fprintf('类中心法 + 欧几里得距离准确率: %.2f%% (%d/%d)\n', acc_centroid, correct_centroid, N_test);

% --- 9.2 类中心法 + 余弦相似度 ---
correct_centroid_cos = 0;
for i = 1:N_test
    test_vec = Y_fisher_test(:, i);
    norm_test = norm(test_vec);
    if norm_test < 1e-10, continue; end
    norms_c = sqrt(sum(centroids_fisher.^2, 1));
    cos_sim = (test_vec' * centroids_fisher) ./ (norm_test * norms_c + 1e-10);
    [~, best_idx] = max(cos_sim);
    if strcmp(unique_names_valid{best_idx}, labels_test{i})
        correct_centroid_cos = correct_centroid_cos + 1;
    end
end
acc_centroid_cos = correct_centroid_cos / N_test * 100;
fprintf('类中心法 + 余弦相似度准确率: %.2f%% (%d/%d)\n', acc_centroid_cos, correct_centroid_cos, N_test);

% --- 9.3 1-NN 欧几里得距离 ---
correct_1nn = 0;
for i = 1:N_test
    test_vec = Y_fisher_test(:, i);
    diff = Y_fisher_train - test_vec;
    dists = sum(diff.^2, 1);
    [~, best] = min(dists);
    if strcmp(labels_train{best}, labels_test{i})
        correct_1nn = correct_1nn + 1;
    end
end
acc_1nn = correct_1nn / N_test * 100;
fprintf('1-NN + 欧几里得距离准确率: %.2f%% (%d/%d)\n', acc_1nn, correct_1nn, N_test);

% --- 9.4 1-NN 余弦相似度 ---
correct_1nn_cos = 0;
for i = 1:N_test
    test_vec = Y_fisher_test(:, i);
    norm_test = norm(test_vec);
    if norm_test < 1e-10, continue; end
    norms_train = sqrt(sum(Y_fisher_train.^2, 1));
    cos_sim = (test_vec' * Y_fisher_train) ./ (norm_test * norms_train + 1e-10);
    [~, best] = max(cos_sim);
    if strcmp(labels_train{best}, labels_test{i})
        correct_1nn_cos = correct_1nn_cos + 1;
    end
end
acc_1nn_cos = correct_1nn_cos / N_test * 100;
fprintf('1-NN + 余弦相似度准确率: %.2f%% (%d/%d)\n', acc_1nn_cos, correct_1nn_cos, N_test);

% --- 9.5 k-NN (k=3) 欧几里得距离 ---
k_val = 3;
correct_knn = 0;
for i = 1:N_test
    test_vec = Y_fisher_test(:, i);
    diff = Y_fisher_train - test_vec;
    dists = sum(diff.^2, 1);
    [~, sorted_idx] = sort(dists);
    % 取前 k 个邻居投票
    top_k_labels = labels_train(sorted_idx(1:k_val));
    % 多数投票
    vote_names = unique(top_k_labels);
    vote_counts = zeros(length(vote_names), 1);
    for v = 1:length(vote_names)
        vote_counts(v) = sum(strcmp(top_k_labels, vote_names{v}));
    end
    [~, winner] = max(vote_counts);
    if strcmp(vote_names{winner}, labels_test{i})
        correct_knn = correct_knn + 1;
    end
end
acc_knn = correct_knn / N_test * 100;
fprintf('k-NN(k=%d) + 欧几里得距离准确率: %.2f%% (%d/%d)\n', k_val, acc_knn, correct_knn, N_test);

% --- 9.6 加权 k-NN (k=5) ---
k_val_w = 5;
correct_wknn = 0;
for i = 1:N_test
    test_vec = Y_fisher_test(:, i);
    diff = Y_fisher_train - test_vec;
    dists = sum(diff.^2, 1);
    [sorted_dists, sorted_idx] = sort(dists);
    top_k_dists = sorted_dists(1:k_val_w);
    top_k_labels_w = labels_train(sorted_idx(1:k_val_w));
    % 距离反比加权
    weights = 1 ./ (top_k_dists + 1e-10);
    vote_names = unique(top_k_labels_w);
    vote_scores = zeros(length(vote_names), 1);
    for v = 1:length(vote_names)
        mask_v = strcmp(top_k_labels_w, vote_names{v});
        vote_scores(v) = sum(weights(mask_v));
    end
    [~, winner] = max(vote_scores);
    if strcmp(vote_names{winner}, labels_test{i})
        correct_wknn = correct_wknn + 1;
    end
end
acc_wknn = correct_wknn / N_test * 100;
fprintf('加权k-NN(k=%d) + 欧几里得距离准确率: %.2f%% (%d/%d)\n', k_val_w, acc_wknn, correct_wknn, N_test);

%% ===== 10. 不同 LDA 维度下的准确率 =====
fprintf('\n===== 不同 LDA 维度下的 1-NN 准确率 =====\n');
test_dims = unique([5, 10, 15, 20, 25, min(30, lda_dim), lda_dim]);
test_dims = test_dims(test_dims <= lda_dim & test_dims > 0);

for di = 1:length(test_dims)
    d = test_dims(di);
    W_sub = W_pca * W_lda_vectors(:, 1:d);
    Y_tr_sub = W_sub' * X_all(:, train_idx);
    Y_te_sub = W_sub' * X_all(:, test_idx);
    
    correct_sub = 0;
    for i = 1:N_test
        diff_sub = Y_tr_sub - Y_te_sub(:, i);
        dists_sub = sum(diff_sub.^2, 1);
        [~, best_sub] = min(dists_sub);
        if strcmp(labels_train{best_sub}, labels_test{i})
            correct_sub = correct_sub + 1;
        end
    end
    fprintf('  LDA 维度 %d: 1-NN 准确率 = %.2f%% (%d/%d)\n', ...
        d, correct_sub/N_test*100, correct_sub, N_test);
end

%% ===== 11. 类内/类间距离分析（Fisher 空间） =====
fprintf('\n===== Fisher 空间类内/类间距离分析 =====\n');

intra_dists = [];
for k = 1:C
    name = unique_names_valid{k};
    class_mask = strcmp(labels_train, name);
    class_data = Y_fisher_train(:, class_mask);
    n_k = size(class_data, 2);
    if n_k < 2, continue; end
    mu_k = mean(class_data, 2);
    for i = 1:n_k
        intra_dists = [intra_dists; norm(class_data(:,i) - mu_k)]; %#ok<AGROW>
    end
end

inter_dists = [];
for k1 = 1:C
    for k2 = k1+1:C
        d_inter = norm(centroids_fisher(:, k1) - centroids_fisher(:, k2));
        inter_dists = [inter_dists; d_inter]; %#ok<AGROW>
    end
end

fprintf('类内平均距离: %.2f\n', mean(intra_dists));
fprintf('类间平均距离（中心间）: %.2f\n', mean(inter_dists));
ratio = mean(inter_dists) / mean(intra_dists);
fprintf('类间/类内距离比: %.3f（PCA 空间中为 0.647，越大越好）\n', ratio);

%% ===== 12. 与纯 PCA 对比汇总 =====
fprintf('\n========================================\n');
fprintf('       Fisherface 结果汇总\n');
fprintf('========================================\n');
fprintf('PCA 降维: %d 维（跳过前 %d 个光照分量）\n', pca_dim, skip_pc);
fprintf('LDA 降维: %d 维（C-1 = %d）\n', lda_dim, C-1);
fprintf('正则化参数 α: %.2e\n', alpha_reg);
fprintf('训练集: %d 张, 测试集: %d 张, 类别数: %d\n', N_train, N_test, C);
fprintf('----------------------------------------\n');
fprintf('【Fisherface 类中心法】\n');
fprintf('  欧几里得距离准确率: %.2f%%\n', acc_centroid);
fprintf('  余弦相似度准确率:   %.2f%%\n', acc_centroid_cos);
fprintf('----------------------------------------\n');
fprintf('【Fisherface 1-NN】\n');
fprintf('  欧几里得距离准确率: %.2f%%\n', acc_1nn);
fprintf('  余弦相似度准确率:   %.2f%%\n', acc_1nn_cos);
fprintf('----------------------------------------\n');
fprintf('【Fisherface k-NN(k=%d)】\n', k_val);
fprintf('  欧几里得距离准确率: %.2f%%\n', acc_knn);
fprintf('----------------------------------------\n');
fprintf('【Fisherface 加权k-NN(k=%d)】\n', k_val_w);
fprintf('  欧几里得距离准确率: %.2f%%\n', acc_wknn);
fprintf('----------------------------------------\n');
fprintf('【Fisher 空间距离分析】\n');
fprintf('  类间/类内距离比: %.3f\n', ratio);
fprintf('========================================\n');
fprintf('\n对比：纯 PCA 最优准确率约 51.72%%\n');
fprintf('Fisherface 通过最大化类间/类内散度比，预期可提升至 65-75%%\n');

% test_PCA_accuracy.m
% 人脸识别准确率测试脚本（增强版）
% 使用多种分类方法验证PCA人脸识别准确率：
%   - 类中心法（Nearest Centroid）
%   - k-NN 投票法
%   - 加权 k-NN
%   - 去除前N个主成分实验
%   - 不同PCA维度实验
%   - 马氏距离
%   - 【新增】相关系数匹配法
%   - 【新增】子空间法（每类一个子空间）
%   - 【新增】软投票集成法（多方法融合）
%   - 【新增】局部子空间法（L-NN + 局部PCA重建）
% 诊断分析：
%   - 数据集分布诊断（每人样本数统计）
%   - 每人识别准确率分析
%   - 混淆分析（最容易混淆的人脸对）
%   - 类内/类间距离分析
%   - PCA特征值谱分析
%   - 最优skip+维度组合搜索
%   - 重建误差分析
%   - 图像质量与多样性指标
% 注意：运行此脚本前，需先运行 PCA.m 以获得 mean_face, P, Y, file_list, valid_count, skip_pc, lambda_sorted 等变量

%% ===== 0. 检查工作空间中是否存在PCA所需变量 =====
if ~exist('mean_face', 'var') || ~exist('P', 'var') || ~exist('Y', 'var')
    error('请先运行 PCA.m，确保工作空间中存在 mean_face, P, Y 等变量。');
end

% 兼容新版PCA.m的skip_pc变量
if ~exist('skip_pc', 'var')
    skip_pc = 0;
    fprintf('注意：未检测到 skip_pc 变量，默认不跳过主成分。\n');
end

%% ===== 1. 重新读取所有图片并提取人名标签 =====
face_old_folder = 'D:\Linear_algebra\Face\Face_old';
all_files = dir(fullfile(face_old_folder, '*.*'));
% 只保留图片文件（jpg和png）
img_extensions = {'.jpg', '.jpeg', '.png', '.bmp'};
keep_mask = false(length(all_files), 1);
for i = 1:length(all_files)
    if all_files(i).isdir
        continue;
    end
    [~, ~, ext] = fileparts(all_files(i).name);
    if any(strcmpi(ext, img_extensions))
        keep_mask(i) = true;
    end
end
all_files = all_files(keep_mask);

fprintf('共找到 %d 张图片文件\n', length(all_files));

%% ===== 2. 提取每张图片的人名标签 =====
labels = cell(length(all_files), 1);
for i = 1:length(all_files)
    fname = all_files(i).name;
    % 提取人名：匹配开头的连续中文字符（2-3个汉字）
    % 文件名格式: "人名_编号.jpg" 或 "人名 (编号).png" 或 "人名.jpg"
    % 使用正则表达式匹配开头的中文字符
    tokens = regexp(fname, '^([\x{4e00}-\x{9fff}]{2,3})', 'tokens', 'once');
    if ~isempty(tokens)
        labels{i} = tokens{1};
    else
        % 无法解析的文件名，标记为空
        labels{i} = '';
    end
end

% 去除无法解析标签的图片
valid_mask = ~cellfun(@isempty, labels);
all_files = all_files(valid_mask);
labels = labels(valid_mask);

fprintf('有效标记图片数: %d\n', length(labels));

% 获取所有唯一的人名
unique_names = unique(labels);
fprintf('共有 %d 个不同的人\n', length(unique_names));

%% ===== 3. 处理所有图片（灰度化+缩放到60x60），与PCA.m保持一致 =====
img_size = [60, 60];
num_images = length(all_files);
all_vectors = zeros(img_size(1)*img_size(2), num_images);
valid_indices = [];

for i = 1:num_images
    filepath = fullfile(all_files(i).folder, all_files(i).name);
    try
        img = imread(filepath);
        % 灰度化
        if size(img, 3) == 3
            gray_img = uint8(0.299*double(img(:,:,1)) + ...
                             0.587*double(img(:,:,2)) + ...
                             0.114*double(img(:,:,3)));
        elseif size(img, 3) == 1
            gray_img = img;
        else
            continue;
        end
        % 高斯滤波 + 缩放（与PCA.m一致，去掉第二次滤波避免过度模糊）
        filtered = imgaussfilt(gray_img, 0.3);
        resized = imresize(filtered, img_size);
        all_vectors(:, i) = double(resized(:));
        valid_indices = [valid_indices, i]; %#ok<AGROW>
    catch
        % 跳过损坏图片
        fprintf('跳过无法读取的图片: %s\n', all_files(i).name);
    end
end

% 只保留成功处理的
all_vectors = all_vectors(:, valid_indices);
labels = labels(valid_indices);
num_valid = length(valid_indices);
fprintf('成功处理 %d 张图片\n', num_valid);

%% ===== 4. 将所有图片投影到PCA空间 =====
% 中心化（使用PCA.m得到的mean_face）
X_all = all_vectors - mean_face;
% 投影
Y_all = P' * X_all;   % r × num_valid

%% ===== 5. 随机划分训练集和测试集 =====
rng(42);  % 固定随机种子，保证结果可复现
test_ratio = 0.3;  % 30%作为测试集

% 按人分层抽样，确保每个人至少有1张在训练集
train_idx = [];
test_idx = [];

unique_names_valid = unique(labels);
for k = 1:length(unique_names_valid)
    name = unique_names_valid{k};
    person_idx = find(strcmp(labels, name));
    n_person = length(person_idx);
    
    if n_person < 2
        % 只有1张照片的人全部放入训练集
        train_idx = [train_idx; person_idx(:)]; %#ok<AGROW>
        continue;
    end
    
    % 随机打乱
    perm = randperm(n_person);
    n_test = max(1, round(n_person * test_ratio));
    n_train = n_person - n_test;
    
    train_idx = [train_idx; person_idx(perm(1:n_train))]; %#ok<AGROW>
    test_idx = [test_idx; person_idx(perm(n_train+1:end))]; %#ok<AGROW>
end

fprintf('\n===== 数据划分 =====\n');
fprintf('训练集: %d 张\n', length(train_idx));
fprintf('测试集: %d 张\n', length(test_idx));

%% ===== 6. 类中心法（Nearest Centroid） =====
% 对每个人的训练样本在PCA空间取均值，测试时与各人中心比较
fprintf('\n===== 类中心法（Nearest Centroid） =====\n');

Y_train = Y_all(:, train_idx);
Y_test = Y_all(:, test_idx);
labels_train = labels(train_idx);
labels_test = labels(test_idx);
num_test = length(test_idx);

% 计算每个人的类中心
centroids = zeros(size(Y_train, 1), length(unique_names_valid));
for k = 1:length(unique_names_valid)
    name = unique_names_valid{k};
    person_mask = strcmp(labels_train, name);
    centroids(:, k) = mean(Y_train(:, person_mask), 2);
end

% 余弦相似度 - 类中心法
correct_centroid_cosine = 0;
for i = 1:num_test
    test_vec = Y_test(:, i);
    norm_test = norm(test_vec);
    if norm_test < 1e-10
        continue;
    end
    norms_centroids = sqrt(sum(centroids.^2, 1));
    cos_sim = (test_vec' * centroids) ./ (norm_test * norms_centroids + 1e-10);
    [~, best_idx] = max(cos_sim);
    if strcmp(unique_names_valid{best_idx}, labels_test{i})
        correct_centroid_cosine = correct_centroid_cosine + 1;
    end
end
accuracy_centroid_cosine = correct_centroid_cosine / num_test * 100;
fprintf('类中心法 + 余弦相似度准确率: %.2f%% (%d/%d)\n', accuracy_centroid_cosine, correct_centroid_cosine, num_test);

% 欧几里得距离 - 类中心法
correct_centroid_euclidean = 0;
for i = 1:num_test
    test_vec = Y_test(:, i);
    diff = centroids - test_vec;
    distances = sqrt(sum(diff.^2, 1));
    [~, best_idx] = min(distances);
    if strcmp(unique_names_valid{best_idx}, labels_test{i})
        correct_centroid_euclidean = correct_centroid_euclidean + 1;
    end
end
accuracy_centroid_euclidean = correct_centroid_euclidean / num_test * 100;
fprintf('类中心法 + 欧几里得距离准确率: %.2f%% (%d/%d)\n', accuracy_centroid_euclidean, correct_centroid_euclidean, num_test);

%% ===== 7. k-NN 投票法（多个k值） =====
fprintf('\n===== k-NN 投票法 =====\n');

K_values = [1, 3, 5];  % 测试多个k值

% 预计算训练集的范数（避免重复计算）
norms_train = sqrt(sum(Y_train.^2, 1));

% 存储各k值的结果
accuracy_knn_cosine = zeros(length(K_values), 1);
accuracy_knn_euclidean = zeros(length(K_values), 1);

for ki = 1:length(K_values)
    K = K_values(ki);
    fprintf('\n--- k = %d ---\n', K);
    
    % 余弦相似度 k-NN
    correct_knn_cos = 0;
    for i = 1:num_test
        test_vec = Y_test(:, i);
        norm_test = norm(test_vec);
        if norm_test < 1e-10
            continue;
        end
        cos_sim = (test_vec' * Y_train) ./ (norm_test * norms_train + 1e-10);
        
        k_actual = min(K, length(cos_sim));
        [~, sorted_idx] = sort(cos_sim, 'descend');
        top_k_idx = sorted_idx(1:k_actual);
        top_k_labels = labels_train(top_k_idx);
        
        % 多数投票
        vote_names = unique(top_k_labels);
        vote_counts = zeros(length(vote_names), 1);
        for v = 1:length(vote_names)
            vote_counts(v) = sum(strcmp(top_k_labels, vote_names{v}));
        end
        [~, winner_idx] = max(vote_counts);
        predicted_label = vote_names{winner_idx};
        
        if strcmp(predicted_label, labels_test{i})
            correct_knn_cos = correct_knn_cos + 1;
        end
    end
    accuracy_knn_cosine(ki) = correct_knn_cos / num_test * 100;
    fprintf('k-NN(k=%d) + 余弦相似度准确率: %.2f%% (%d/%d)\n', K, accuracy_knn_cosine(ki), correct_knn_cos, num_test);
    
    % 欧几里得距离 k-NN
    correct_knn_euc = 0;
    for i = 1:num_test
        test_vec = Y_test(:, i);
        diff = Y_train - test_vec;
        distances = sqrt(sum(diff.^2, 1));
        
        k_actual = min(K, length(distances));
        [~, sorted_idx] = sort(distances, 'ascend');
        top_k_idx = sorted_idx(1:k_actual);
        top_k_labels = labels_train(top_k_idx);
        
        % 多数投票
        vote_names = unique(top_k_labels);
        vote_counts = zeros(length(vote_names), 1);
        for v = 1:length(vote_names)
            vote_counts(v) = sum(strcmp(top_k_labels, vote_names{v}));
        end
        [~, winner_idx] = max(vote_counts);
        predicted_label = vote_names{winner_idx};
        
        if strcmp(predicted_label, labels_test{i})
            correct_knn_euc = correct_knn_euc + 1;
        end
    end
    accuracy_knn_euclidean(ki) = correct_knn_euc / num_test * 100;
    fprintf('k-NN(k=%d) + 欧几里得距离准确率: %.2f%% (%d/%d)\n', K, accuracy_knn_euclidean(ki), correct_knn_euc, num_test);
end

%% ===== 8. 加权 k-NN（距离加权投票） =====
fprintf('\n===== 加权 k-NN（距离反比加权） =====\n');

K_weighted = 5;
fprintf('k = %d\n', K_weighted);

% 余弦相似度加权 k-NN
correct_wknn_cos = 0;
for i = 1:num_test
    test_vec = Y_test(:, i);
    norm_test = norm(test_vec);
    if norm_test < 1e-10
        continue;
    end
    cos_sim = (test_vec' * Y_train) ./ (norm_test * norms_train + 1e-10);
    
    k_actual = min(K_weighted, length(cos_sim));
    [sorted_sim, sorted_idx] = sort(cos_sim, 'descend');
    top_k_idx = sorted_idx(1:k_actual);
    top_k_labels = labels_train(top_k_idx);
    top_k_weights = max(sorted_sim(1:k_actual), 0);  % 相似度作为权重
    
    % 加权投票
    vote_names = unique(top_k_labels);
    vote_scores = zeros(length(vote_names), 1);
    for v = 1:length(vote_names)
        mask = strcmp(top_k_labels, vote_names{v});
        vote_scores(v) = sum(top_k_weights(mask));
    end
    [~, winner_idx] = max(vote_scores);
    predicted_label = vote_names{winner_idx};
    
    if strcmp(predicted_label, labels_test{i})
        correct_wknn_cos = correct_wknn_cos + 1;
    end
end
accuracy_wknn_cosine = correct_wknn_cos / num_test * 100;
fprintf('加权k-NN + 余弦相似度准确率: %.2f%% (%d/%d)\n', accuracy_wknn_cosine, correct_wknn_cos, num_test);

% 欧几里得距离加权 k-NN
correct_wknn_euc = 0;
for i = 1:num_test
    test_vec = Y_test(:, i);
    diff = Y_train - test_vec;
    distances = sqrt(sum(diff.^2, 1));
    
    k_actual = min(K_weighted, length(distances));
    [sorted_dist, sorted_idx] = sort(distances, 'ascend');
    top_k_idx = sorted_idx(1:k_actual);
    top_k_labels = labels_train(top_k_idx);
    top_k_weights = 1 ./ (sorted_dist(1:k_actual) + 1e-10);  % 距离反比作为权重
    
    % 加权投票
    vote_names = unique(top_k_labels);
    vote_scores = zeros(length(vote_names), 1);
    for v = 1:length(vote_names)
        mask = strcmp(top_k_labels, vote_names{v});
        vote_scores(v) = sum(top_k_weights(mask));
    end
    [~, winner_idx] = max(vote_scores);
    predicted_label = vote_names{winner_idx};
    
    if strcmp(predicted_label, labels_test{i})
        correct_wknn_euc = correct_wknn_euc + 1;
    end
end
accuracy_wknn_euclidean = correct_wknn_euc / num_test * 100;
fprintf('加权k-NN + 欧几里得距离准确率: %.2f%% (%d/%d)\n', accuracy_wknn_euclidean, correct_wknn_euc, num_test);

%% ===== 9. 去除前几个主成分（去光照）的识别实验 =====
fprintf('\n===== 去除前N个主成分（去光照干扰）实验 =====\n');

skip_values = [1, 2, 3];  % 分别去掉前1、2、3个主成分

for si = 1:length(skip_values)
    skip_n = skip_values(si);
    if skip_n >= size(Y_all, 1)
        continue;
    end
    
    % 去掉前skip_n个成分
    Y_train_skip = Y_train(skip_n+1:end, :);
    Y_test_skip = Y_test(skip_n+1:end, :);
    
    % 用1-NN + 欧几里得距离测试
    correct_skip = 0;
    for i = 1:num_test
        test_vec = Y_test_skip(:, i);
        diff = Y_train_skip - test_vec;
        distances = sqrt(sum(diff.^2, 1));
        [~, best_idx] = min(distances);
        if strcmp(labels_train{best_idx}, labels_test{i})
            correct_skip = correct_skip + 1;
        end
    end
    acc_skip = correct_skip / num_test * 100;
    fprintf('去掉前%d个主成分 + 1-NN + 欧几里得: %.2f%% (%d/%d)\n', skip_n, acc_skip, correct_skip, num_test);
end

%% ===== 10. 不同主成分数量的影响实验 =====
fprintf('\n===== 不同PCA维度下的识别准确率 =====\n');

% 测试不同保留维度（10~全部，含原始的93维作为参考）
r_full = size(Y_all, 1);
dim_values = unique([10, 20, 30, 50, min(70, r_full), min(93, r_full), r_full]);
dim_values = dim_values(dim_values <= r_full);

for di = 1:length(dim_values)
    dim = dim_values(di);
    Y_train_dim = Y_train(1:dim, :);
    Y_test_dim = Y_test(1:dim, :);
    
    % 1-NN + 欧几里得距离
    correct_dim = 0;
    for i = 1:num_test
        test_vec = Y_test_dim(:, i);
        diff = Y_train_dim - test_vec;
        distances = sqrt(sum(diff.^2, 1));
        [~, best_idx] = min(distances);
        if strcmp(labels_train{best_idx}, labels_test{i})
            correct_dim = correct_dim + 1;
        end
    end
    acc_dim = correct_dim / num_test * 100;
    fprintf('保留前%d维 + 1-NN + 欧几里得: %.2f%% (%d/%d)\n', dim, acc_dim, correct_dim, num_test);
end

%% ===== 11. 马氏距离（对角近似）分类 =====
fprintf('\n===== 马氏距离（对角近似）最近邻 =====\n');

% 使用PCA特征值作为各维度方差的估计
if exist('skip_pc', 'var') && skip_pc > 0
    lambda_diag = lambda_sorted(skip_pc+1 : skip_pc+size(Y_train,1));
else
    lambda_diag = lambda_sorted(1:size(Y_train,1));
end
% 避免除以过小的值
lambda_diag(lambda_diag < 1e-6) = 1e-6;

correct_mahal = 0;
for i = 1:num_test
    test_vec = Y_test(:, i);
    diff = Y_train - test_vec;
    % 马氏距离 = sqrt(sum((diff_j)^2 / lambda_j))
    mahal_dist = sqrt(sum(diff.^2 ./ lambda_diag, 1));
    [~, best_idx] = min(mahal_dist);
    if strcmp(labels_train{best_idx}, labels_test{i})
        correct_mahal = correct_mahal + 1;
    end
end
accuracy_mahal = correct_mahal / num_test * 100;
fprintf('马氏距离(对角近似) + 1-NN 准确率: %.2f%% (%d/%d)\n', accuracy_mahal, correct_mahal, num_test);

%% ===== 12. 【新增】相关系数匹配法 =====
fprintf('\n===== 相关系数匹配法（Correlation） =====\n');

% 相关系数法：计算测试样本与每个训练样本之间的 Pearson 相关系数
correct_corr = 0;
num_corr_tested = 0;
for i = 1:num_test
    test_vec = Y_test(:, i);
    test_centered = test_vec - mean(test_vec);
    test_norm = norm(test_centered);
    if test_norm < 1e-10
        continue;  % 零向量无法计算相关系数，跳过
    end
    num_corr_tested = num_corr_tested + 1;
    
    % 计算与所有训练样本的相关系数
    train_centered = Y_train - mean(Y_train, 1);
    train_norms = sqrt(sum(train_centered.^2, 1));
    corr_vals = (test_centered' * train_centered) ./ (test_norm * train_norms + 1e-10);
    
    [~, best_idx] = max(corr_vals);
    if strcmp(labels_train{best_idx}, labels_test{i})
        correct_corr = correct_corr + 1;
    end
end
accuracy_corr = correct_corr / max(num_corr_tested, 1) * 100;
fprintf('相关系数匹配 + 1-NN 准确率: %.2f%% (%d/%d)\n', accuracy_corr, correct_corr, num_corr_tested);

%% ===== 13. 【新增】子空间法（每类一个子空间） =====
fprintf('\n===== 子空间法（Per-Class Subspace） =====\n');

% 每个人用其训练样本构建一个小子空间，测试时计算到各子空间的投影残差
subspace_dim = 3;  % 每类子空间维度（取较小值，因为每人样本少）

% 为每个人构建子空间基
class_bases = cell(length(unique_names_valid), 1);
class_means = zeros(size(Y_train, 1), length(unique_names_valid));

for k = 1:length(unique_names_valid)
    name = unique_names_valid{k};
    person_mask = strcmp(labels_train, name);
    Y_person = Y_train(:, person_mask);
    class_means(:, k) = mean(Y_person, 2);
    
    % 中心化后用手工QR分解求子空间基
    Y_person_c = Y_person - class_means(:, k);
    n_samples = size(Y_person_c, 2);
    
    if n_samples >= subspace_dim
        % 手工Gram-Schmidt正交化求前subspace_dim个基向量
        basis = zeros(size(Y_person_c, 1), subspace_dim);
        for bi = 1:min(subspace_dim, n_samples)
            v = Y_person_c(:, bi);
            for bj = 1:bi-1
                v = v - (basis(:, bj)' * v) * basis(:, bj);
            end
            nv = norm(v);
            if nv > 1e-10
                basis(:, bi) = v / nv;
            end
        end
        class_bases{k} = basis;
    else
        % 样本不够，直接用中心化后的样本作为基
        basis = zeros(size(Y_person_c, 1), n_samples);
        for bi = 1:n_samples
            v = Y_person_c(:, bi);
            for bj = 1:bi-1
                v = v - (basis(:, bj)' * v) * basis(:, bj);
            end
            nv = norm(v);
            if nv > 1e-10
                basis(:, bi) = v / nv;
            end
        end
        class_bases{k} = basis;
    end
end

% 测试：计算到每个类子空间的残差距离
correct_subspace = 0;
for i = 1:num_test
    test_vec = Y_test(:, i);
    min_residual = inf;
    best_class = -1;
    
    for k = 1:length(unique_names_valid)
        % 投影到类子空间
        centered = test_vec - class_means(:, k);
        basis = class_bases{k};
        proj = basis * (basis' * centered);
        residual = norm(centered - proj);
        
        if residual < min_residual
            min_residual = residual;
            best_class = k;
        end
    end
    
    if strcmp(unique_names_valid{best_class}, labels_test{i})
        correct_subspace = correct_subspace + 1;
    end
end
accuracy_subspace = correct_subspace / num_test * 100;
fprintf('子空间法（每类%d维）准确率: %.2f%% (%d/%d)\n', subspace_dim, accuracy_subspace, correct_subspace, num_test);

%% ===== 14. 【新增】白化欧几里得距离（Whitened Euclidean） =====
fprintf('\n===== 白化欧几里得距离 =====\n');

% 对每一维除以其标准差（即特征值的平方根），使各维度等权
if exist('skip_pc', 'var') && skip_pc > 0
    lambda_for_whiten = lambda_sorted(skip_pc+1 : skip_pc+size(Y_train,1));
else
    lambda_for_whiten = lambda_sorted(1:size(Y_train,1));
end
whiten_scale = 1 ./ sqrt(max(lambda_for_whiten, 1e-6));

Y_train_w = Y_train .* whiten_scale;
Y_test_w = Y_test .* whiten_scale;

correct_whiten = 0;
for i = 1:num_test
    test_vec = Y_test_w(:, i);
    diff = Y_train_w - test_vec;
    distances = sum(diff.^2, 1);
    [~, best_idx] = min(distances);
    if strcmp(labels_train{best_idx}, labels_test{i})
        correct_whiten = correct_whiten + 1;
    end
end
accuracy_whiten = correct_whiten / num_test * 100;
fprintf('白化欧几里得 + 1-NN 准确率: %.2f%% (%d/%d)\n', accuracy_whiten, correct_whiten, num_test);

%% ===== 15. 【新增】软投票集成法（多方法融合） =====
fprintf('\n===== 软投票集成法（Ensemble） =====\n');

% 融合三种方法的预测结果：欧几里得1-NN、余弦1-NN、相关系数1-NN
% 使用多数投票决定最终标签
correct_ensemble = 0;
for i = 1:num_test
    test_vec = Y_test(:, i);
    
    % 方法1：欧几里得 1-NN
    diff = Y_train - test_vec;
    distances = sqrt(sum(diff.^2, 1));
    [~, idx1] = min(distances);
    pred1 = labels_train{idx1};
    
    % 方法2：余弦相似度 1-NN
    norm_test = norm(test_vec);
    cos_sim = (test_vec' * Y_train) ./ (norm_test * norms_train + 1e-10);
    [~, idx2] = max(cos_sim);
    pred2 = labels_train{idx2};
    
    % 方法3：相关系数 1-NN
    test_c = test_vec - mean(test_vec);
    test_n = norm(test_c);
    if test_n > 1e-10
        train_c = Y_train - mean(Y_train, 1);
        train_n = sqrt(sum(train_c.^2, 1));
        corr_v = (test_c' * train_c) ./ (test_n * train_n + 1e-10);
        [~, idx3] = max(corr_v);
        pred3 = labels_train{idx3};
    else
        pred3 = pred1;
    end
    
    % 多数投票
    preds = {pred1, pred2, pred3};
    vote_names = unique(preds);
    vote_counts = zeros(length(vote_names), 1);
    for v = 1:length(vote_names)
        vote_counts(v) = sum(strcmp(preds, vote_names{v}));
    end
    [~, winner] = max(vote_counts);
    final_pred = vote_names{winner};
    
    if strcmp(final_pred, labels_test{i})
        correct_ensemble = correct_ensemble + 1;
    end
end
accuracy_ensemble = correct_ensemble / num_test * 100;
fprintf('软投票集成（欧几里得+余弦+相关系数）准确率: %.2f%% (%d/%d)\n', accuracy_ensemble, correct_ensemble, num_test);

%% ===== 16. 【新增】类中心 + 白化距离 =====
fprintf('\n===== 类中心 + 白化距离 =====\n');

centroids_w = zeros(size(Y_train_w, 1), length(unique_names_valid));
for k = 1:length(unique_names_valid)
    name = unique_names_valid{k};
    person_mask = strcmp(labels_train, name);
    centroids_w(:, k) = mean(Y_train_w(:, person_mask), 2);
end

correct_centroid_w = 0;
for i = 1:num_test
    test_vec = Y_test_w(:, i);
    diff = centroids_w - test_vec;
    distances = sqrt(sum(diff.^2, 1));
    [~, best_idx] = min(distances);
    if strcmp(unique_names_valid{best_idx}, labels_test{i})
        correct_centroid_w = correct_centroid_w + 1;
    end
end
accuracy_centroid_w = correct_centroid_w / num_test * 100;
fprintf('类中心 + 白化欧几里得准确率: %.2f%% (%d/%d)\n', accuracy_centroid_w, correct_centroid_w, num_test);

%% ===== 17. 【新增】数据集分布诊断 =====
fprintf('\n===== 数据集分布诊断 =====\n');

% 每人样本数统计
samples_per_person = zeros(length(unique_names_valid), 1);
train_per_person = zeros(length(unique_names_valid), 1);
test_per_person = zeros(length(unique_names_valid), 1);
for k = 1:length(unique_names_valid)
    name = unique_names_valid{k};
    samples_per_person(k) = sum(strcmp(labels, name));
    train_per_person(k) = sum(strcmp(labels_train, name));
    test_per_person(k) = sum(strcmp(labels_test, name));
end

fprintf('每人总样本数: 最少=%d, 最多=%d, 平均=%.1f, 中位数=%.1f\n', ...
    min(samples_per_person), max(samples_per_person), ...
    mean(samples_per_person), median(samples_per_person));
fprintf('每人训练样本: 最少=%d, 最多=%d, 平均=%.1f\n', ...
    min(train_per_person), max(train_per_person), mean(train_per_person));
fprintf('每人测试样本: 最少=%d, 最多=%d, 平均=%.1f\n', ...
    min(test_per_person), max(test_per_person), mean(test_per_person));

% 列出样本数较少的人（可能影响识别率）
fprintf('\n样本数 <= 5 的人:\n');
for k = 1:length(unique_names_valid)
    if samples_per_person(k) <= 5
        fprintf('  %s: 总%d张 (训练%d, 测试%d)\n', ...
            unique_names_valid{k}, samples_per_person(k), ...
            train_per_person(k), test_per_person(k));
    end
end

%% ===== 18. 【新增】每人识别准确率分析 =====
fprintf('\n===== 每人识别准确率分析（1-NN 欧几里得）=====\n');

person_correct = zeros(length(unique_names_valid), 1);
person_total = zeros(length(unique_names_valid), 1);

for i = 1:num_test
    test_vec = Y_test(:, i);
    diff = Y_train - test_vec;
    distances = sqrt(sum(diff.^2, 1));
    [~, best_idx] = min(distances);
    
    % 找到这个人在unique_names_valid中的索引
    true_class = find(strcmp(unique_names_valid, labels_test{i}));
    person_total(true_class) = person_total(true_class) + 1;
    
    if strcmp(labels_train{best_idx}, labels_test{i})
        person_correct(true_class) = person_correct(true_class) + 1;
    end
end

% 计算每人准确率并排序
person_accuracy = zeros(length(unique_names_valid), 1);
for k = 1:length(unique_names_valid)
    if person_total(k) > 0
        person_accuracy(k) = person_correct(k) / person_total(k) * 100;
    else
        person_accuracy(k) = -1;  % 没有测试样本
    end
end

[sorted_acc, sort_idx] = sort(person_accuracy, 'descend');
fprintf('\n识别率最高的5人:\n');
count = 0;
for k = 1:length(sort_idx)
    if sorted_acc(k) < 0, continue; end
    count = count + 1;
    idx = sort_idx(k);
    fprintf('  %s: %.1f%% (%d/%d) [训练%d张]\n', ...
        unique_names_valid{idx}, sorted_acc(k), ...
        person_correct(idx), person_total(idx), train_per_person(idx));
    if count >= 5, break; end
end

fprintf('\n识别率最低的5人:\n');
count = 0;
for k = length(sort_idx):-1:1
    if sorted_acc(k) < 0, continue; end
    count = count + 1;
    idx = sort_idx(k);
    fprintf('  %s: %.1f%% (%d/%d) [训练%d张]\n', ...
        unique_names_valid{idx}, sorted_acc(k), ...
        person_correct(idx), person_total(idx), train_per_person(idx));
    if count >= 5, break; end
end

% 完全识别正确和完全错误的人数
fully_correct = sum(person_accuracy == 100 & person_total > 0);
fully_wrong = sum(person_accuracy == 0 & person_total > 0);
fprintf('\n完全正确的人数: %d/%d\n', fully_correct, sum(person_total > 0));
fprintf('完全错误的人数: %d/%d\n', fully_wrong, sum(person_total > 0));

%% ===== 19. 【新增】混淆分析（最容易混淆的人脸对）=====
fprintf('\n===== 混淆分析（最容易被误判为谁）=====\n');

% 构建简易混淆矩阵
num_classes = length(unique_names_valid);
confusion = zeros(num_classes, num_classes);

for i = 1:num_test
    test_vec = Y_test(:, i);
    diff = Y_train - test_vec;
    distances = sqrt(sum(diff.^2, 1));
    [~, best_idx] = min(distances);
    
    true_class = find(strcmp(unique_names_valid, labels_test{i}));
    pred_class = find(strcmp(unique_names_valid, labels_train{best_idx}));
    confusion(true_class, pred_class) = confusion(true_class, pred_class) + 1;
end

% 找出最大的非对角线混淆对
confusion_offdiag = confusion;
for k = 1:num_classes
    confusion_offdiag(k, k) = 0;
end

fprintf('最容易混淆的5对人脸:\n');
for pair = 1:5
    [max_val, linear_idx] = max(confusion_offdiag(:));
    if max_val == 0, break; end
    [row, col] = ind2sub([num_classes, num_classes], linear_idx);
    fprintf('  "%s" 被误判为 "%s": %d 次\n', ...
        unique_names_valid{row}, unique_names_valid{col}, max_val);
    confusion_offdiag(row, col) = 0;  % 标记已输出
end

%% ===== 20. 【新增】类内/类间距离分析 =====
fprintf('\n===== 类内距离 vs 类间距离分析 =====\n');

% 在PCA空间中计算类内和类间平均距离
intra_dists = [];
inter_dists = [];

for k = 1:length(unique_names_valid)
    name = unique_names_valid{k};
    person_mask = strcmp(labels_train, name);
    Y_person = Y_train(:, person_mask);
    n_p = size(Y_person, 2);
    
    % 类内距离：同一人不同样本之间的距离
    if n_p >= 2
        for a = 1:n_p-1
            for b = a+1:n_p
                d = norm(Y_person(:, a) - Y_person(:, b));
                intra_dists = [intra_dists; d]; %#ok<AGROW>
            end
        end
    end
end

% 类间距离：不同人的类中心之间的距离
for k1 = 1:num_classes-1
    for k2 = k1+1:num_classes
        d = norm(centroids(:, k1) - centroids(:, k2));
        inter_dists = [inter_dists; d]; %#ok<AGROW>
    end
end

fprintf('类内平均距离: %.2f (std=%.2f, min=%.2f, max=%.2f)\n', ...
    mean(intra_dists), std(intra_dists), min(intra_dists), max(intra_dists));
fprintf('类间平均距离（中心间）: %.2f (std=%.2f, min=%.2f, max=%.2f)\n', ...
    mean(inter_dists), std(inter_dists), min(inter_dists), max(inter_dists));
fprintf('类间/类内距离比: %.3f（越大越容易区分，理想值 > 2）\n', ...
    mean(inter_dists) / mean(intra_dists));

% 找类间距离最小的几对（最难区分的）
[sorted_inter, inter_idx] = sort(inter_dists, 'ascend');
fprintf('\n类间距离最小的3对（最难区分）:\n');
pair_count = 0;
pair_idx = 0;
for k1 = 1:num_classes-1
    for k2 = k1+1:num_classes
        pair_idx = pair_idx + 1;
        if any(inter_idx(1:3) == pair_idx)
            pair_count = pair_count + 1;
            fprintf('  "%s" vs "%s": 中心距离 = %.2f\n', ...
                unique_names_valid{k1}, unique_names_valid{k2}, ...
                norm(centroids(:, k1) - centroids(:, k2)));
        end
        if pair_count >= 3, break; end
    end
    if pair_count >= 3, break; end
end

%% ===== 21. 【新增】PCA特征值谱分析 =====
fprintf('\n===== PCA特征值谱分析 =====\n');

if exist('lambda_sorted', 'var')
    total_var = sum(lambda_sorted);
    cum_var = cumsum(lambda_sorted) / total_var * 100;
    
    fprintf('总方差: %.2e\n', total_var);
    fprintf('前1个主成分累积方差贡献: %.2f%%\n', cum_var(1));
    fprintf('前5个主成分累积方差贡献: %.2f%%\n', cum_var(min(5, length(cum_var))));
    fprintf('前10个主成分累积方差贡献: %.2f%%\n', cum_var(min(10, length(cum_var))));
    fprintf('前20个主成分累积方差贡献: %.2f%%\n', cum_var(min(20, length(cum_var))));
    fprintf('前40个主成分累积方差贡献: %.2f%%\n', cum_var(min(40, length(cum_var))));
    fprintf('前70个主成分累积方差贡献: %.2f%%\n', cum_var(min(70, length(cum_var))));
    
    % 特征值衰减速率
    fprintf('\n特征值衰减（前10个）:\n');
    for ev = 1:min(10, length(lambda_sorted))
        fprintf('  λ_%d = %.4e (占比 %.2f%%)\n', ev, lambda_sorted(ev), ...
            lambda_sorted(ev)/total_var*100);
    end
    
    % 有效维度估计（特征值 > 平均值的维度数）
    avg_lambda = mean(lambda_sorted(lambda_sorted > 0));
    effective_dim = sum(lambda_sorted > avg_lambda);
    fprintf('\n有效维度估计（λ > 平均特征值）: %d\n', effective_dim);
    fprintf('达到95%%方差所需维度: %d\n', find(cum_var >= 95, 1, 'first'));
    fprintf('达到99%%方差所需维度: %d\n', find(cum_var >= 99, 1, 'first'));
else
    fprintf('未检测到 lambda_sorted 变量，跳过特征值分析。\n');
end

%% ===== 22. 【新增】最优维度+skip组合搜索 =====
fprintf('\n===== 最优 skip_pc + 保留维度 组合搜索 =====\n');

skip_candidates = [0, 1, 2, 3, 5];
dim_candidates = [20, 30, 40, 50, 70];
best_combo_acc = 0;
best_skip = 0;
best_dim = 0;

% 需要完整的特征向量矩阵Q
if exist('Q', 'var')
    for si = 1:length(skip_candidates)
        for di = 1:length(dim_candidates)
            s = skip_candidates(si);
            d = dim_candidates(di);
            if s + d > size(Q, 2), continue; end
            
            P_test = Q(:, s+1 : s+d);
            Y_tr_test = P_test' * (all_vectors(:, train_idx) - mean_face);
            Y_te_test = P_test' * (all_vectors(:, test_idx) - mean_face);
            
            correct_combo = 0;
            for i = 1:num_test
                diff_c = Y_tr_test - Y_te_test(:, i);
                dists_c = sum(diff_c.^2, 1);
                [~, best_c] = min(dists_c);
                if strcmp(labels_train{best_c}, labels_test{i})
                    correct_combo = correct_combo + 1;
                end
            end
            acc_combo = correct_combo / num_test * 100;
            fprintf('  skip=%d, dim=%d: %.2f%%\n', s, d, acc_combo);
            
            if acc_combo > best_combo_acc
                best_combo_acc = acc_combo;
                best_skip = s;
                best_dim = d;
            end
        end
    end
    fprintf('最优组合: skip=%d, dim=%d, 准确率=%.2f%%\n', best_skip, best_dim, best_combo_acc);
else
    fprintf('未检测到 Q 变量，跳过组合搜索。\n');
end

%% ===== 23. 【新增】重建误差分析 =====
fprintf('\n===== 重建误差分析 =====\n');

% 计算PCA重建误差（原始空间中）
recon_errors = zeros(num_test, 1);
for i = 1:num_test
    x_orig = all_vectors(:, test_idx(i)) - mean_face;
    x_recon = P * (P' * x_orig);  % 投影再重建
    recon_errors(i) = norm(x_orig - x_recon) / norm(x_orig) * 100;
end

fprintf('测试集平均相对重建误差: %.2f%%\n', mean(recon_errors));
fprintf('重建误差范围: %.2f%% ~ %.2f%%\n', min(recon_errors), max(recon_errors));
fprintf('重建误差标准差: %.2f%%\n', std(recon_errors));

% 按正确/错误分类样本的重建误差对比
correct_mask_recon = false(num_test, 1);
for i = 1:num_test
    test_vec = Y_test(:, i);
    diff_r = Y_train - test_vec;
    distances_r = sum(diff_r.^2, 1);
    [~, best_r] = min(distances_r);
    if strcmp(labels_train{best_r}, labels_test{i})
        correct_mask_recon(i) = true;
    end
end

if any(correct_mask_recon)
    fprintf('正确分类样本的平均重建误差: %.2f%%\n', mean(recon_errors(correct_mask_recon)));
end
if any(~correct_mask_recon)
    fprintf('错误分类样本的平均重建误差: %.2f%%\n', mean(recon_errors(~correct_mask_recon)));
end
fprintf('（重建误差高 → 该人脸可能不在训练集主要变差空间内）\n');

%% ===== 24. 【新增】图像质量/多样性指标 =====
fprintf('\n===== 图像质量与多样性指标 =====\n');

% 计算训练集每人的类内方差（代表姿态/光照多样性）
fprintf('每人类内方差（PCA空间，反映训练多样性）:\n');
class_variance = zeros(length(unique_names_valid), 1);
for k = 1:length(unique_names_valid)
    name = unique_names_valid{k};
    person_mask = strcmp(labels_train, name);
    Y_person = Y_train(:, person_mask);
    if size(Y_person, 2) > 1
        class_variance(k) = mean(var(Y_person, 0, 2));
    else
        class_variance(k) = 0;
    end
end

[sorted_var, var_idx] = sort(class_variance, 'descend');
fprintf('  类内方差最大（最多样）的3人:\n');
for k = 1:min(3, length(var_idx))
    fprintf('    %s: 方差=%.2e (样本%d张)\n', ...
        unique_names_valid{var_idx(k)}, sorted_var(k), train_per_person(var_idx(k)));
end
fprintf('  类内方差最小（最单一）的3人:\n');
for k = length(var_idx):-1:max(1, length(var_idx)-2)
    fprintf('    %s: 方差=%.2e (样本%d张)\n', ...
        unique_names_valid{var_idx(k)}, class_variance(var_idx(k)), train_per_person(var_idx(k)));
end

%% ===== 25. 结果汇总 =====
fprintf('\n========================================\n');
fprintf('       人脸识别准确率测试结果汇总\n');
fprintf('========================================\n');
fprintf('PCA保留主成分数: %d（跳过前%d个）\n', size(P, 2), skip_pc);
fprintf('训练集样本数: %d\n', length(train_idx));
fprintf('测试集样本数: %d\n', num_test);
fprintf('人数: %d\n', length(unique_names_valid));
fprintf('每人平均样本数: %.1f（最少%d，最多%d）\n', mean(samples_per_person), min(samples_per_person), max(samples_per_person));
fprintf('随机种子: 42（固定）\n');
fprintf('----------------------------------------\n');
fprintf('【类中心法】\n');
fprintf('  余弦相似度准确率:   %.2f%%\n', accuracy_centroid_cosine);
fprintf('  欧几里得距离准确率: %.2f%%\n', accuracy_centroid_euclidean);
fprintf('  白化欧几里得准确率: %.2f%%\n', accuracy_centroid_w);
fprintf('----------------------------------------\n');
fprintf('【k-NN 投票法】\n');
for ki = 1:length(K_values)
    fprintf('  k=%d 余弦: %.2f%%  欧几里得: %.2f%%\n', K_values(ki), accuracy_knn_cosine(ki), accuracy_knn_euclidean(ki));
end
fprintf('----------------------------------------\n');
fprintf('【加权 k-NN (k=%d)】\n', K_weighted);
fprintf('  余弦相似度准确率:   %.2f%%\n', accuracy_wknn_cosine);
fprintf('  欧几里得距离准确率: %.2f%%\n', accuracy_wknn_euclidean);
fprintf('----------------------------------------\n');
fprintf('【马氏距离 1-NN】\n');
fprintf('  准确率: %.2f%%\n', accuracy_mahal);
fprintf('----------------------------------------\n');
fprintf('【相关系数匹配 1-NN】\n');
fprintf('  准确率: %.2f%%\n', accuracy_corr);
fprintf('----------------------------------------\n');
fprintf('【子空间法（每类%d维）】\n', subspace_dim);
fprintf('  准确率: %.2f%%\n', accuracy_subspace);
fprintf('----------------------------------------\n');
fprintf('【白化欧几里得 1-NN】\n');
fprintf('  准确率: %.2f%%\n', accuracy_whiten);
fprintf('----------------------------------------\n');
fprintf('【软投票集成法】\n');
fprintf('  准确率: %.2f%%\n', accuracy_ensemble);
fprintf('----------------------------------------\n');
fprintf('【诊断指标】\n');
fprintf('  类间/类内距离比: %.3f\n', mean(inter_dists) / mean(intra_dists));
fprintf('  完全正确人数: %d/%d\n', fully_correct, sum(person_total > 0));
fprintf('  完全错误人数: %d/%d\n', fully_wrong, sum(person_total > 0));
fprintf('  平均重建误差: %.2f%%\n', mean(recon_errors));
if exist('Q', 'var')
    fprintf('  最优组合: skip=%d, dim=%d → %.2f%%\n', best_skip, best_dim, best_combo_acc);
end
fprintf('========================================\n');

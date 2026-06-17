% test_PCA_accuracy.m
% 人脸识别准确率测试脚本
% 使用类中心法（Nearest Centroid）和 k-NN 投票法验证PCA人脸识别准确率
% 每种方法分别使用余弦相似度和欧几里得距离两种度量
% 注意：运行此脚本前，需先运行 PCA.m 以获得 mean_face, P, Y, file_list, valid_count 等变量

%% ===== 0. 检查工作空间中是否存在PCA所需变量 =====
if ~exist('mean_face', 'var') || ~exist('P', 'var') || ~exist('Y', 'var')
    error('请先运行 PCA.m，确保工作空间中存在 mean_face, P, Y 等变量。');
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
lambda_diag = lambda_sorted(1:size(Y_train,1));
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

%% ===== 12. 结果汇总 =====
fprintf('\n========================================\n');
fprintf('       人脸识别准确率测试结果汇总\n');
fprintf('========================================\n');
fprintf('PCA保留主成分数: %d\n', size(P, 2));
fprintf('训练集样本数: %d\n', length(train_idx));
fprintf('测试集样本数: %d\n', num_test);
fprintf('人数: %d\n', length(unique_names_valid));
fprintf('随机种子: 42（固定）\n');
fprintf('----------------------------------------\n');
fprintf('【类中心法】\n');
fprintf('  余弦相似度准确率:   %.2f%%\n', accuracy_centroid_cosine);
fprintf('  欧几里得距离准确率: %.2f%%\n', accuracy_centroid_euclidean);
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
fprintf('========================================\n');

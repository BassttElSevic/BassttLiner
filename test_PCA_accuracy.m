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
        % 高斯滤波 + 缩放（与PCA.m一致）
        filtered = imgaussfilt(gray_img, 0.3);
        resized = imgaussfilt(imresize(filtered, img_size));
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
rng('shuffle');  % 随机种子
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

%% ===== 7. k-NN 投票法 =====
% 取k个最近邻，按人名多数投票决定分类结果
fprintf('\n===== k-NN 投票法 =====\n');

K = 5;  % k值，可根据数据量调整
fprintf('k = %d\n', K);

% 余弦相似度 k-NN
correct_knn_cosine = 0;
for i = 1:num_test
    test_vec = Y_test(:, i);
    norm_test = norm(test_vec);
    if norm_test < 1e-10
        continue;
    end
    norms_train = sqrt(sum(Y_train.^2, 1));
    cos_sim = (test_vec' * Y_train) ./ (norm_test * norms_train + 1e-10);
    
    % 取前K个最相似的邻居
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
        correct_knn_cosine = correct_knn_cosine + 1;
    end
end
accuracy_knn_cosine = correct_knn_cosine / num_test * 100;
fprintf('k-NN + 余弦相似度准确率: %.2f%% (%d/%d)\n', accuracy_knn_cosine, correct_knn_cosine, num_test);

% 欧几里得距离 k-NN
correct_knn_euclidean = 0;
for i = 1:num_test
    test_vec = Y_test(:, i);
    diff = Y_train - test_vec;
    distances = sqrt(sum(diff.^2, 1));
    
    % 取前K个最近的邻居
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
        correct_knn_euclidean = correct_knn_euclidean + 1;
    end
end
accuracy_knn_euclidean = correct_knn_euclidean / num_test * 100;
fprintf('k-NN + 欧几里得距离准确率: %.2f%% (%d/%d)\n', accuracy_knn_euclidean, correct_knn_euclidean, num_test);

%% ===== 8. 结果汇总 =====
fprintf('\n========================================\n');
fprintf('       人脸识别准确率测试结果汇总\n');
fprintf('========================================\n');
fprintf('PCA保留主成分数: %d\n', size(P, 2));
fprintf('训练集样本数: %d\n', length(train_idx));
fprintf('测试集样本数: %d\n', num_test);
fprintf('人数: %d\n', length(unique_names_valid));
fprintf('k-NN 的 k 值: %d\n', K);
fprintf('----------------------------------------\n');
fprintf('类中心法 + 余弦相似度准确率:   %.2f%%\n', accuracy_centroid_cosine);
fprintf('类中心法 + 欧几里得距离准确率: %.2f%%\n', accuracy_centroid_euclidean);
fprintf('k-NN + 余弦相似度准确率:       %.2f%%\n', accuracy_knn_cosine);
fprintf('k-NN + 欧几里得距离准确率:     %.2f%%\n', accuracy_knn_euclidean);
fprintf('========================================\n');

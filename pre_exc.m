% 任务一，对人像进行预处理
%  所谓图像配准，就是对同一场景的两幅或者多幅图像进行对准，比如很多人脸自动分析系统中的人
%  脸归一化，作用就是使照片中的人脸大小一致并尽量处于相同的位置。
%  图像配准需要以基准图像为参照，并通过一些基准点找到适当的空间变换关系系数，然后对输入图
%  像进行相应的几何变换，从而实现与基准图像在这些基准点位置的对齐
% example_image = imread('D:\Linear_algebra\HUMANFACE\HUMANFACE\81011EB7D2C1CD813B7C6342A06C5019.jpg');
% % imshow(example_image);
% PR_of_example_image = example_image(:,:,1);
% PG_of_example_image = example_image(:,:,2);
% PB_of_example_image = example_image(:,:,3);
% raw_gray_example_image = 0.299 * PR_of_example_image + 0.587 * PG_of_example_image + 0.114 * PB_of_example_image;

file_list = dir('D:\Linear_algebra\HUMANFACE\HUMANFACE\*.jpg'); %批量化读取
n = length(file_list);

images = cell(1, n);
PR_of_images = cell(1, n);
PG_of_images = cell(1, n);
PB_of_images = cell(1, n);
RAW_gray_images = cell(1, n);
Gauss_filtered = cell(1, n);

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
        Gauss_filtered{valid_count} = imgaussfilt(RAW_gray_images{valid_count}, 1.0);
        
        fprintf('成功处理: %s\n', file_list(i).name);
        
    catch e
        fprintf('跳过损坏图片: %s (%s)\n', file_list(i).name, e.message);
        % 不增加 valid_count，直接进入下一张
    end
end

% 去掉空位
images = images(1:valid_count);
PR_of_images = PR_of_images(1:valid_count);
PG_of_images = PG_of_images(1:valid_count);
PB_of_images = PB_of_images(1:valid_count);
RAW_gray_images = RAW_gray_images(1:valid_count);
Gauss_filtered = Gauss_filtered(1:valid_count);

fprintf('\n共成功处理 %d / %d 张图片\n', valid_count, n);

% % ---- 2. 分页显示（一次性弹出所有页） ----
% cols = 6;
% rows_per_group = 2;
% per_page = cols * rows_per_group;   % 每页 12 组
% total_pages = ceil(valid_count / per_page);
% 
% for page = 1:total_pages
%     start_idx = (page - 1) * per_page + 1;
%     end_idx = min(page * per_page, valid_count);
%     count = end_idx - start_idx + 1;
% 
%     figure('Name', sprintf('第 %d/%d 页 (图片 %d~%d)', page, total_pages, start_idx, end_idx), ...
%            'NumberTitle', 'off');
% 
%     for j = 1:count
%         idx = start_idx + j - 1;
% 
%         subplot(rows_per_group * 2, cols, j);
%         imshow(RAW_gray_images{idx});
%         title(sprintf('#%d', idx), 'FontSize', 7);
% 
%         subplot(rows_per_group * 2, cols, per_page + j);
%         imshow(Gauss_filtered{idx});
%         title(sprintf('#%d 滤波', idx), 'FontSize', 7);
%     end
% end
%% ---- 0. 在滤波代码之后继续执行 ----

%% ---- 1. 创建检测器 ----
faceDetector = vision.CascadeObjectDetector();
eyeDetector  = vision.CascadeObjectDetector('EyePairBig');

%% ---- 2. 选择基准图 ----
% 遍历找到第一张能成功检测到眼睛的图作为基准
ref_idx = -1;
ref_left_eye = [];
ref_right_eye = [];

for i = 1:valid_count
    try
        eye_bbox = step(eyeDetector, Gauss_filtered{i});
        if ~isempty(eye_bbox)
            eye_bbox = eye_bbox(1,:);
            ref_left_eye  = [eye_bbox(1) + eye_bbox(3)*0.25, ...
                             eye_bbox(2) + eye_bbox(4)*0.5];
            ref_right_eye = [eye_bbox(1) + eye_bbox(3)*0.75, ...
                             eye_bbox(2) + eye_bbox(4)*0.5];
            ref_idx = i;
            fprintf('✅ 选择图片 #%d 作为基准图\n', i);
            break;
        end
    catch
        continue;
    end
end

if ref_idx == -1
    error('❌ 所有图片都无法检测到眼睛，无法进行配准');
end

ref_img = Gauss_filtered{ref_idx};
ref_eye_dist   = norm(ref_right_eye - ref_left_eye);
ref_eye_angle  = atan2d(ref_right_eye(2) - ref_left_eye(2), ...
                        ref_right_eye(1) - ref_left_eye(1));
ref_eye_center = (ref_left_eye + ref_right_eye) / 2;
output_size    = size(ref_img);

fprintf('基准图眼睛中心: (%.1f, %.1f), 眼距: %.1f\n', ...
    ref_eye_center(1), ref_eye_center(2), ref_eye_dist);

%% ---- 3. 对所有图片进行配准 ----
aligned_images = cell(1, valid_count);
align_status = zeros(1, valid_count);  % 0=失败 1=成功 2=回退估计

for i = 1:valid_count
    try
        img = Gauss_filtered{i};
        
        % 基准图直接跳过
        if i == ref_idx
            aligned_images{i} = ref_img;
            align_status(i) = 1;
            continue;
        end
        
        % ---- 尝试检测眼睛 ----
        left_eye = [];
        right_eye = [];
        
        try
            eye_bbox = step(eyeDetector, img);
            if ~isempty(eye_bbox)
                eye_bbox = eye_bbox(1,:);
                left_eye  = [eye_bbox(1) + eye_bbox(3)*0.25, ...
                             eye_bbox(2) + eye_bbox(4)*0.5];
                right_eye = [eye_bbox(1) + eye_bbox(3)*0.75, ...
                             eye_bbox(2) + eye_bbox(4)*0.5];
            end
        catch
            % 眼睛检测器出错，继续尝试人脸
        end
        
        % ---- 眼睛没检测到，回退到人脸框估算 ----
        if isempty(left_eye)
            try
                face_bbox = step(faceDetector, img);
                if ~isempty(face_bbox)
                    face_bbox = face_bbox(1,:);
                    left_eye  = [face_bbox(1) + face_bbox(3)*0.3, ...
                                 face_bbox(2) + face_bbox(4)*0.35];
                    right_eye = [face_bbox(1) + face_bbox(3)*0.7, ...
                                 face_bbox(2) + face_bbox(4)*0.35];
                    fprintf('⚠️ 图片 #%d 用人脸框估算眼睛位置\n', i);
                end
            catch
                % 人脸检测也失败了
            end
        end
        
        % ---- 都检测不到，直接缩放到统一尺寸 ----
        if isempty(left_eye)
            fprintf('❌ 图片 #%d 未检测到人脸和眼睛，仅缩放\n', i);
            aligned_images{i} = imresize(img, output_size);
            align_status(i) = 0;
            continue;
        end
        
        % ---- 计算仿射变换 ----
        cur_angle  = atan2d(right_eye(2) - left_eye(2), ...
                            right_eye(1) - left_eye(1));
        rotate_angle = ref_eye_angle - cur_angle;
        
        cur_eye_dist = norm(right_eye - left_eye);
        if cur_eye_dist < 1
            fprintf('❌ 图片 #%d 眼距异常(%.1f)，仅缩放\n', i, cur_eye_dist);
            aligned_images{i} = imresize(img, output_size);
            align_status(i) = 0;
            continue;
        end
        scale = ref_eye_dist / cur_eye_dist;
        
        cur_eye_center = (left_eye + right_eye) / 2;
        
        cos_a = cosd(rotate_angle);
        sin_a = sind(rotate_angle);
        
        tx = ref_eye_center(1) - scale * (cos_a * cur_eye_center(1) - sin_a * cur_eye_center(2));
        ty = ref_eye_center(2) - scale * (sin_a * cur_eye_center(1) + cos_a * cur_eye_center(2));
        
        tform = affine2d([scale*cos_a,  scale*sin_a, 0; ...
                         -scale*sin_a,  scale*cos_a, 0; ...
                          tx,           ty,          1]);
        
        ref_view = imref2d(output_size);
        aligned_images{i} = imwarp(img, tform, 'OutputView', ref_view, ...
                                   'FillValues', 0);
        
        align_status(i) = 1;
        fprintf('✅ 图片 #%d 配准完成 (旋转%.1f° 缩放%.2f)\n', i, rotate_angle, scale);
        
    catch e
        % ---- 最外层兜底：任何意外错误都不会崩溃 ----
        fprintf('❌ 图片 #%d 配准异常: %s\n', i, e.message);
        try
            aligned_images{i} = imresize(Gauss_filtered{i}, output_size);
        catch
            aligned_images{i} = zeros(output_size, 'uint8');
        end
        align_status(i) = 0;
    end
end

% ---- 统计结果 ----
fprintf('\n====== 配准统计 ======\n');
fprintf('成功: %d 张\n', sum(align_status == 1));
fprintf('失败(仅缩放): %d 张\n', sum(align_status == 0));
fprintf('总计: %d 张\n', valid_count);

%% ---- 4. 分页显示 ----
cols = 6;
per_page = cols * 2;
total_pages = ceil(valid_count / per_page);

for page = 1:total_pages
    start_idx = (page - 1) * per_page + 1;
    end_idx = min(page * per_page, valid_count);
    count = end_idx - start_idx + 1;
    
    figure('Name', sprintf('配准结果 第%d/%d页', page, total_pages), ...
           'NumberTitle', 'off');
    
    for j = 1:count
        idx = start_idx + j - 1;
        
        subplot(4, cols, j);
        imshow(Gauss_filtered{idx});
        title(sprintf('#%d 前', idx), 'FontSize', 7);
        
        subplot(4, cols, per_page + j);
        imshow(aligned_images{idx});
        % 根据状态标注颜色
        if align_status(idx) == 1
            title(sprintf('#%d ✅', idx), 'FontSize', 7, 'Color', 'g');
        else
            title(sprintf('#%d ❌', idx), 'FontSize', 7, 'Color', 'r');
        end
    end
end

cols = 6;
rows = 4;
per_page = cols * rows;   % 每页 24 张
total_pages = ceil(valid_count / per_page);

for page = 1:total_pages
    start_idx = (page - 1) * per_page + 1;
    end_idx = min(page * per_page, valid_count);
    count = end_idx - start_idx + 1;
    
    figure('Name', sprintf('配准成品 第%d/%d页 (图片%d~%d)', page, total_pages, start_idx, end_idx), ...
           'NumberTitle', 'off');
    
    for j = 1:count
        idx = start_idx + j - 1;
        
        subplot(rows, cols, j);
        imshow(aligned_images{idx});
        if align_status(idx) == 1
            title(sprintf('#%d ✅', idx), 'FontSize', 7, 'Color', 'g');
        else
            title(sprintf('#%d ❌', idx), 'FontSize', 7, 'Color', 'r');
        end
    end
end
% % 展示三个维度的分量
% figure;
% subplot(1,3,1);
% imshow(PR_of_example_image);
% title('Red Channel');
% 
% subplot(1,3,2);
% imshow(PG_of_example_image);
% title('Green Channel');
% 
% subplot(1,3,3);
% imshow(PB_of_example_image);
% title('Blue Channel');
% 
% figure;
% raw_gray_example_image = 0.299 * PR_of_example_image + 0.587 * PG_of_example_image + 0.114 * PB_of_example_image;
% imshow(raw_gray_example_image);
% title('BT.601 加权灰度化');
% 
% figure;
% 
% Gauss_filtered_3 = medfilt2(raw_gray_example_image, [3 3]);
% filtered_5 = medfilt2(raw_gray_example_image, [5 5]);
% filtered_7 = medfilt2(raw_gray_example_image, [7 7]);
% 
% subplot(1,3,1); imshow(Gauss_filtered_3); title('邻域 3×3');
% subplot(1,3,2); imshow(filtered_5); title('邻域 5×5');
% subplot(1,3,3); imshow(filtered_7); title('邻域 7×7');
% 
% figure;
% % Gauss滤波
% Gauss_filtered_05 = imgaussfilt(raw_gray_example_image, 0.5);   % σ=0.5 轻微模糊
% Gauss_filtered_1  = imgaussfilt(raw_gray_example_image, 1.0);   % σ=1.0 中等模糊
% Gauss_filtered_3  = imgaussfilt(raw_gray_example_image, 3.0);   % σ=3.0 强烈模糊
% 
% subplot(1,3,1); imshow(Gauss_filtered_05); title('邻域 3×3');
% subplot(1,3,2); imshow(Gauss_filtered_1); title('邻域 5×5');
% subplot(1,3,3); imshow(Gauss_filtered_3); title('邻域 7×7');

% face_align_augment.m — 人脸对齐 + 数据增强预处理脚本
% =========================================================================
% 功能：
%   1. 从 Face_old 文件夹读取原始照片
%   2. 使用 MATLAB 内置 CascadeObjectDetector 检测眼睛位置
%   3. 根据眼睛位置进行旋转对齐（使双眼连线水平）
%   4. 裁剪并缩放到统一尺寸 60×60
%   5. 数据增强：水平翻转、轻微旋转（±5°）、亮度微调
%   6. 输出到 Face_neo 文件夹
%
% 使用方式：
%   直接运行此脚本，完成后再运行 PCA.m（PCA.m 从 Face_neo 读取）
%
% 如果效果不好，可以恢复 PCA.m 中被注释的旧预处理代码。
% =========================================================================

%% ===== 参数设置 =====
input_folder  = 'D:\Linear_algebra\Face\Face_old';
output_folder = 'D:\Linear_algebra\Face\Face_neo';
img_size = [60, 60];   % 输出统一尺寸

% 数据增强选项
do_flip        = true;    % 水平翻转
do_rotation    = true;    % 轻微旋转增强
rotation_angles = [-5, 5]; % 增强旋转角度（度）
do_brightness  = true;    % 亮度微调
brightness_factors = [0.85, 1.15]; % 亮度缩放因子

fprintf('\n========================================\n');
fprintf('   人脸对齐 + 数据增强预处理\n');
fprintf('========================================\n');

%% ===== 清空输出文件夹 =====
if exist(output_folder, 'dir')
    delete(fullfile(output_folder, '*'));
    fprintf('已清空输出文件夹: %s\n', output_folder);
else
    mkdir(output_folder);
    fprintf('已创建输出文件夹: %s\n', output_folder);
end

%% ===== 读取文件列表 =====
all_files = dir(fullfile(input_folder, '*.*'));
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
n = length(all_files);
fprintf('共找到 %d 张图片\n', n);

%% ===== 创建眼睛检测器 =====
eyeDetector = vision.CascadeObjectDetector('EyePairBig');
eyeDetector.MinSize = [11, 45];
eyeDetector.MergeThreshold = 3;

% 备用：单眼检测器（当眼对检测失败时）
leftEyeDetector  = vision.CascadeObjectDetector('LeftEye');
leftEyeDetector.MinSize  = [10, 10];
leftEyeDetector.MergeThreshold = 4;
rightEyeDetector = vision.CascadeObjectDetector('RightEye');
rightEyeDetector.MinSize  = [10, 10];
rightEyeDetector.MergeThreshold = 4;

%% ===== 主循环：对齐 + 增强 =====
total_output = 0;
align_success = 0;
align_fallback = 0;

for i = 1:n
    filepath = fullfile(all_files(i).folder, all_files(i).name);
    [~, name_base, ~] = fileparts(all_files(i).name);
    
    try
        img = imread(filepath);
    catch
        fprintf('跳过无法读取: %s\n', all_files(i).name);
        continue;
    end
    
    % 转灰度
    if size(img, 3) == 3
        gray = uint8(0.299*double(img(:,:,1)) + ...
                     0.587*double(img(:,:,2)) + ...
                     0.114*double(img(:,:,3)));
    elseif size(img, 3) == 1
        gray = img;
    else
        continue;
    end
    
    % ----- 眼睛检测与对齐 -----
    aligned_gray = align_face_by_eyes(gray, eyeDetector, ...
                                       leftEyeDetector, rightEyeDetector);
    
    if isempty(aligned_gray)
        % 对齐失败，直接使用原始灰度图（仅缩放）
        aligned_gray = gray;
        align_fallback = align_fallback + 1;
    else
        align_success = align_success + 1;
    end
    
    % 缩放到统一尺寸
    resized = imresize(aligned_gray, img_size);
    
    % ----- 保存原始对齐图 -----
    out_path = fullfile(output_folder, [name_base, '.jpg']);
    imwrite(resized, out_path);
    total_output = total_output + 1;
    
    % ----- 数据增强 -----
    
    % 1) 水平翻转
    if do_flip
        flipped = flip_perm(resized, 'lr');
        out_path = fullfile(output_folder, [name_base, '_flip.jpg']);
        imwrite(flipped, out_path);
        total_output = total_output + 1;
    end
    
    % 2) 轻微旋转增强（使用自己写的 imrotate_matlab_linear）
    if do_rotation
        for ai = 1:length(rotation_angles)
            ang = rotation_angles(ai);
            rotated = imrotate_matlab_linear(resized, ang, 'loose', false, 'fill', 0);
            % 保持 crop 模式，尺寸不变
            out_path = fullfile(output_folder, ...
                [name_base, sprintf('_rot%+d.jpg', ang)]);
            imwrite(rotated, out_path);
            total_output = total_output + 1;
        end
    end
    
    % 3) 亮度微调
    if do_brightness
        for bi = 1:length(brightness_factors)
            bf = brightness_factors(bi);
            bright_img = uint8(min(255, max(0, double(resized) * bf)));
            out_path = fullfile(output_folder, ...
                [name_base, sprintf('_bright%.0f.jpg', bf*100)]);
            imwrite(bright_img, out_path);
            total_output = total_output + 1;
        end
    end
    
    if mod(i, 20) == 0
        fprintf('  进度: %d/%d\n', i, n);
    end
end

%% ===== 统计结果 =====
fprintf('\n===== 预处理统计 =====\n');
fprintf('输入图片数: %d\n', n);
fprintf('眼睛对齐成功: %d 张\n', align_success);
fprintf('对齐失败(仅缩放): %d 张\n', align_fallback);
fprintf('输出图片总数: %d 张（含增强）\n', total_output);
fprintf('数据增强倍率: %.1fx\n', total_output / n);
fprintf('输出文件夹: %s\n', output_folder);
fprintf('========================================\n');
fprintf('预处理完成！接下来请运行 PCA.m\n');

%% =========================================================================
%  局部函数：基于眼睛检测的人脸对齐
%  =========================================================================
function aligned = align_face_by_eyes(gray_img, eyePairDet, leftEyeDet, rightEyeDet)
% ALIGN_FACE_BY_EYES 基于双眼位置进行人脸旋转对齐
%   输入: gray_img — 灰度图 (uint8)
%         eyePairDet — 眼对检测器
%         leftEyeDet, rightEyeDet — 单眼检测器（备用）
%   输出: aligned — 对齐后的灰度图，若检测失败返回 []
%
%   对齐原理：
%     1. 检测双眼位置 (left_eye, right_eye)
%     2. 计算双眼连线与水平方向的夹角 θ
%     3. 将图像旋转 -θ 度，使双眼连线变为水平
%     4. 以双眼中心为基准裁剪人脸区域

    aligned = [];
    [H, W] = size(gray_img);
    
    left_eye = [];
    right_eye = [];
    
    % --- 方法1：检测眼对 ---
    try
        eye_bbox = step(eyePairDet, gray_img);
        if ~isempty(eye_bbox)
            % 取最大的检测框（面积最大的最可能是正确的）
            areas = eye_bbox(:,3) .* eye_bbox(:,4);
            [~, best] = max(areas);
            bbox = eye_bbox(best, :);
            % 估计左右眼中心位置
            left_eye  = [bbox(1) + bbox(3)*0.25, bbox(2) + bbox(4)*0.5];
            right_eye = [bbox(1) + bbox(3)*0.75, bbox(2) + bbox(4)*0.5];
        end
    catch
        % 检测器出错，继续尝试其他方法
    end
    
    % --- 方法2：分别检测左右眼 ---
    if isempty(left_eye)
        try
            lbox = step(leftEyeDet, gray_img);
            rbox = step(rightEyeDet, gray_img);
            if ~isempty(lbox) && ~isempty(rbox)
                % 取面积最大的
                [~, li] = max(lbox(:,3).*lbox(:,4));
                [~, ri] = max(rbox(:,3).*rbox(:,4));
                left_eye  = [lbox(li,1) + lbox(li,3)/2, lbox(li,2) + lbox(li,4)/2];
                right_eye = [rbox(ri,1) + rbox(ri,3)/2, rbox(ri,2) + rbox(ri,4)/2];
                % 确保左眼在左边
                if left_eye(1) > right_eye(1)
                    tmp = left_eye;
                    left_eye = right_eye;
                    right_eye = tmp;
                end
            end
        catch
            % 单眼检测也失败
        end
    end
    
    % 检测失败，返回空
    if isempty(left_eye) || isempty(right_eye)
        return;
    end
    
    % 检查眼距是否合理（至少是图像宽度的 10%）
    eye_dist = norm(right_eye - left_eye);
    if eye_dist < W * 0.10
        return;  % 眼距太小，可能是误检
    end
    
    % --- 计算旋转角度（使双眼连线水平）---
    angle_deg = atan2d(right_eye(2) - left_eye(2), ...
                       right_eye(1) - left_eye(1));
    
    % 如果角度太大（超过30度），可能是误检
    if abs(angle_deg) > 30
        return;
    end
    
    % --- 旋转图像（使用自己写的旋转函数，crop 模式保持尺寸不变）---
    % 旋转 -angle_deg 度使双眼水平
    rotated = imrotate_matlab_linear(gray_img, -angle_deg, 'loose', false, 'fill', 0);
    
    % --- 计算旋转后的眼睛中心位置 ---
    cx = (W + 1) / 2;
    cy = (H + 1) / 2;
    theta = -angle_deg * pi / 180;
    ct = cos(theta);  st = sin(theta);
    
    % 旋转变换后的眼睛中心
    eye_center = (left_eye + right_eye) / 2;
    new_eye_cx = cx + ct*(eye_center(1)-cx) - st*(eye_center(2)-cy);
    new_eye_cy = cy + st*(eye_center(1)-cx) + ct*(eye_center(2)-cy);
    
    % --- 以眼睛中心为基准裁剪人脸区域 ---
    % 人脸框估计：眼睛中心略偏上方（眼睛大约在人脸上 1/3 处）
    new_eye_dist = eye_dist;  % 旋转不改变距离
    face_width  = new_eye_dist * 2.2;   % 人脸宽度约为眼距的 2.2 倍
    face_height = new_eye_dist * 2.8;   % 人脸高度约为眼距的 2.8 倍
    
    % 裁剪框：眼睛中心位于人脸框上方约 40% 处
    x1 = round(new_eye_cx - face_width/2);
    y1 = round(new_eye_cy - face_height * 0.4);
    x2 = round(x1 + face_width - 1);
    y2 = round(y1 + face_height - 1);
    
    % 边界检查
    x1 = max(1, x1);  y1 = max(1, y1);
    x2 = min(W, x2);  y2 = min(H, y2);
    
    % 裁剪区域太小则放弃
    if (x2 - x1) < 20 || (y2 - y1) < 20
        aligned = rotated;  % 裁剪失败，至少返回旋转后的图
        return;
    end
    
    aligned = rotated(y1:y2, x1:x2);
end

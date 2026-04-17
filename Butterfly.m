%% 蝴蝶漫天飞舞合成图像（v3 - 去除白色背景）
clear; clc; close all;

%% 1. 读取素材图片
butterfly_orig = imread('D:\Linear_algebra\Butterfly\780.jpg');  % 彩色蝴蝶+白色背景
background     = imread('D:\Linear_algebra\Butterfly\R.jpg');   % 背景风景

[bg_h, bg_w, ~] = size(background);

figure('Name', '原始素材');
subplot(1,2,1); imshow(butterfly_orig); title('蝴蝶素材');
subplot(1,2,2); imshow(background);     title('背景素材');

%% 2. 色彩通道运算 —— 提取蝴蝶掩膜（去除白色背景）
% ★★★ 关键：蝴蝶是彩色的，背景是白色的 ★★★
% 白色特征：R高、G高、B高，饱和度S低
% 蝴蝶特征：有颜色，饱和度S较高

butterfly_hsv = rgb2hsv(butterfly_orig);
H = butterfly_hsv(:,:,1);
S = butterfly_hsv(:,:,2);
V = butterfly_hsv(:,:,3);

% 分离 RGB 通道
R = double(butterfly_orig(:,:,1));
G = double(butterfly_orig(:,:,2));
B = double(butterfly_orig(:,:,3));

% ★ 方法：白色背景 = R、G、B 都很高且饱和度很低
% 蝴蝶区域 = 饱和度较高 或 亮度不是纯白
% 组合判断：非白色区域就是蝴蝶
is_white = (R > 220) & (G > 220) & (B > 220);  % 接近纯白的像素
mask = ~is_white;  % 取反 = 非白色 = 蝴蝶区域

% 也可以结合饱和度辅助判断（白色饱和度很低）
mask_sat = S > 0.10;  % 饱和度大于0.10的区域
mask = mask | mask_sat;  % 两个条件取并集，更完整

% 形态学优化
se = strel('disk', 3);
mask = imclose(mask, se);      % 闭运算填充小缝隙
mask = imfill(mask, 'holes');  % 填充内部孔洞
mask = imopen(mask, strel('disk', 2));   % 去除小噪点

% 去除边缘残留的小区域（只保留最大连通区域）
mask = bwareaopen(mask, 500);  % 去除小于500像素的区域

% 显示掩膜
figure('Name', '色彩通道运算 - 掩膜');
subplot(1,3,1); imshow(S);    title('饱和度通道 S');
subplot(1,3,2); imshow(V);    title('明度通道 V');
subplot(1,3,3); imshow(mask); title('蝴蝶掩膜（去除白色背景）');

%% 3. 代数运算 —— 提取蝴蝶前景
mask3 = repmat(uint8(mask), [1, 1, 3]);
butterfly_extracted = butterfly_orig .* mask3;

figure('Name', '代数运算 - 提取蝴蝶');
imshow(butterfly_extracted); title('代数乘法提取蝴蝶');

%% 4. 几何运算 + 代数运算 —— 漫天飞舞
result = double(background);
num_butterflies = 15;
rng(42);

for i = 1:num_butterflies
    %% ---- 几何运算 ----

    % (1) 缩放
    scale = 0.06 + rand() * 0.15;
    new_w = round(bg_w * scale);
    new_h = round(size(butterfly_orig,1) / size(butterfly_orig,2) * new_w);
    b_resized = imresize_bilinear_mat(butterfly_extracted, [new_h, new_w], 'fill', 0);

% mask 缩放建议用 double + fill=0，后续再二值化
m_resized = imresize_bilinear_mat(double(mask), [new_h, new_w], 'fill', 0);
m_resized = m_resized > 0.5;
    % (2) 旋转 —— ★ 关键：旋转时填充值设为 0（黑色），而不是白色
    angle = rand() * 60 - 30;  % 旋转角度 -30° ~ 30°，更自然
   b_rotated = imrotate_matlab_linear(b_resized, angle, 'loose', true, 'fill', 0);

% m_resized 是 double(mask) 这种0/1，旋转后仍会变成灰度（插值导致）
m_rotated = imrotate_matlab_linear(uint8(255*m_resized), angle, 'loose', true, 'fill', 0);
m_rotated = double(m_rotated) / 255;  % 回到0~1
m_rotated = m_rotated > 0.5;          % 二值化

    % ★★★ 关键：旋转后新增的区域掩膜值为0，所以不会显示白色边框 ★★★
    % 将掩膜二值化（旋转插值后可能有中间值）
    m_rotated = m_rotated > 0.5;

    % (3) 随机水平翻转
   if rand() > 0.5
    b_rotated = flip_perm(b_rotated, 'lr');
    m_rotated = flip_perm(m_rotated, 'lr');
end

    % (4) 随机位置（平移）
    [bh, bw, ~] = size(b_rotated);
    max_x = bg_w - bw;
    max_y = bg_h - bh;
    if max_x < 1 || max_y < 1
        continue;
    end
    pos_x = randi([1, max_x]);
    pos_y = randi([1, max_y]);

    %% ---- 代数运算 + 色彩通道运算 ----
    roi_y = pos_y : (pos_y + bh - 1);
    roi_x = pos_x : (pos_x + bw - 1);
    roi = result(roi_y, roi_x, :);

    % 色彩通道运算：轻微调整色调增加多样性
    b_double = double(b_rotated) / 255;
    b_hsv_local = rgb2hsv(b_double);
    hue_shift = rand() * 0.06;           % 很轻微的色调偏移
    b_hsv_local(:,:,1) = mod(b_hsv_local(:,:,1) + hue_shift, 1.0);
    sat_factor = 0.9 + rand() * 0.2;
    b_hsv_local(:,:,2) = min(b_hsv_local(:,:,2) * sat_factor, 1.0);
    b_colored = hsv2rgb(b_hsv_local) * 255;

    % ★ 软边缘掩膜：高斯模糊让边缘自然过渡，消除硬边
    m_soft = imgaussfilt(double(m_rotated), 2.0);
    m3 = repmat(m_soft, [1, 1, 3]);

    % 代数运算：Alpha 混合
    % 合成公式: Result = 蝴蝶 × m3 + 背景 × (1 - m3)
    blended = b_colored .* m3 + roi .* (1 - m3);

    result(roi_y, roi_x, :) = blended;
end

%% 5. 显示最终结果
result = uint8(result);

figure('Name', '最终合成效果');
imshow(result);
title('蝴蝶漫天飞舞', 'FontSize', 16);

%% 6. 保存
imwrite(result, 'D:\Linear_algebra\Butterfly\result_butterfly_fly.jpg');
disp('✓ 合成图像已保存');

%% 7. 总结对比图
figure('Name', '处理流程总结', 'Position', [100 100 1200 800]);
subplot(2,3,1); imshow(butterfly_orig);     title('① 原始蝴蝶素材');
subplot(2,3,2); imshow(S);                  title('② 饱和度通道 S');
subplot(2,3,3); imshow(mask);               title('③ 通道运算→掩膜');
subplot(2,3,4); imshow(butterfly_extracted); title('④ 代数乘法抠图');
subplot(2,3,5); imshow(background);         title('⑤ 背景风景');
subplot(2,3,6); imshow(result);             title('⑥ 最终合成');
sgtitle('蝴蝶漫天飞舞 — 数字图像处理', 'FontSize', 14, 'FontWeight', 'bold');
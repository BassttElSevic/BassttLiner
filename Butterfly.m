%% 蝴蝶漫天飞舞合成图像（v3 - 去除白色背景）
% =========================================================================
% 脚本功能概述：
%   本脚本综合运用数字图像处理中的多种矩阵运算技术，实现"蝴蝶漫天飞舞"的
%   合成图像效果。主要演示以下线性代数与数字图像处理核心概念：
%
%   1. 色彩通道运算（Channel Arithmetic）：
%      将图像视为多通道二维矩阵，通过通道间的逻辑运算提取目标区域掩膜。
%
%   2. 代数运算（Algebraic Operations）：
%      用矩阵逐元素乘法（Hadamard乘积）提取前景，实现 Alpha 混合合成。
%
%   3. 几何变换（Geometric Transformations）：
%      - 缩放：逆向映射 + 双线性插值（见 imresize_bilinear_mat）
%      - 旋转：齐次坐标仿射变换矩阵 + 逆向映射（见 imrotate_matlab_linear）
%      - 翻转：置换矩阵左乘/右乘（见 flip_perm）
%      - 平移：通过 ROI（感兴趣区域）索引实现
%
%   4. 形态学运算（Morphological Operations）：
%      膨胀/腐蚀/填孔，用于优化二值掩膜的质量。
%
% 函数依赖：
%   - imresize_bilinear_mat.m  : 教学版双线性插值缩放
%   - imrotate_matlab_linear.m : 教学版仿射变换旋转
%   - flip_perm.m              : 教学版置换矩阵翻转
% =========================================================================
clear; clc; close all;

%% 1. 读取素材图片
% imread 读取图像，返回 uint8 类型的三维矩阵（H × W × 3）
% 第1维 = 行（高度），第2维 = 列（宽度），第3维 = 颜色通道（RGB）
butterfly_orig = imread('D:\Linear_algebra\Butterfly\780.jpg');  % 彩色蝴蝶+白色背景
background     = imread('D:\Linear_algebra\Butterfly\R.jpg');   % 背景风景

% 获取背景图像尺寸（用于后续几何变换的坐标范围计算）
[bg_h, bg_w, ~] = size(background);

figure('Name', '原始素材');
subplot(1,2,1); imshow(butterfly_orig); title('蝴蝶素材');
subplot(1,2,2); imshow(background);     title('背景素材');

%% 2. 色彩通道运算 —— 提取蝴蝶掩膜（去除白色背景）
% =========================================================================
% 【原理：色彩通道分析与掩膜提取】
%
% 蝴蝶图像的像素可以在 HSV 颜色空间中被有效分类：
%   - 白色背景特征：R≈255, G≈255, B≈255（三通道均高），饱和度 S≈0（接近灰轴）
%   - 蝴蝶区域特征：颜色鲜艳，饱和度 S 较高（远离灰轴）
%
% 策略：
%   1. 用 RGB 三通道阈值判断"接近纯白"的像素 → is_white 掩膜
%   2. 用 HSV 饱和度通道判断"有颜色"的像素 → mask_sat 掩膜
%   3. 取两者并集，得到最终蝴蝶区域掩膜 mask
%   4. 用形态学运算优化掩膜（填孔、去噪）
% =========================================================================

% rgb2hsv 将图像从 RGB 颜色空间转换到 HSV 颜色空间
% HSV 是一种更接近人类视觉感知的颜色表示：
%   H（色调 Hue）     : 颜色种类，范围 [0,1]，对应色环 0°~360°
%   S（饱和度 Sat.）  : 颜色纯度，0=灰色，1=纯色
%   V（明度 Value）   : 亮度，0=黑色，1=最亮
butterfly_hsv = rgb2hsv(butterfly_orig);
H = butterfly_hsv(:,:,1);   % 色调通道矩阵（H × W）
S = butterfly_hsv(:,:,2);   % 饱和度通道矩阵（H × W）
V = butterfly_hsv(:,:,3);   % 明度通道矩阵（H × W）

% 分离 RGB 通道，转为 double 类型以支持数值比较（uint8 比较会有截断问题）
% 图像矩阵的第三维切片：img(:,:,1) 取第1通道（Red），依此类推
R = double(butterfly_orig(:,:,1));   % 红色通道（H × W 矩阵）
G = double(butterfly_orig(:,:,2));   % 绿色通道（H × W 矩阵）
B = double(butterfly_orig(:,:,3));   % 蓝色通道（H × W 矩阵）

% ---- 方法1：RGB 三通道阈值判断白色区域 ----
% 白色像素的 RGB 值均接近 255（uint8 最大值）
% 逻辑运算 & 对矩阵逐元素取与，结果为同尺寸 logical 矩阵
is_white = (R > 220) & (G > 220) & (B > 220);  % 接近纯白的像素（logical H×W）
mask = ~is_white;  % 取反：非白色区域 = 蝴蝶区域（逻辑非，逐元素）

% ---- 方法2：饱和度阈值判断有颜色区域 ----
% 白色的饱和度 S 接近 0，蝴蝶区域的 S 通常 > 0.10
% 与方法1取并集（|），使掩膜更完整地覆盖蝴蝶区域
mask_sat = S > 0.10;         % 饱和度大于阈值的区域（蝴蝶颜色区域）
mask = mask | mask_sat;      % 并集：两种方法任一成立即视为蝴蝶区域

% ---- 形态学优化 ----
% 形态学运算本质上是对二值矩阵做基于结构元素的邻域 max/min 操作：
%   膨胀（dilation）≈ 邻域 max：使白色区域扩张，填充细小缝隙
%   腐蚀（erosion） ≈ 邻域 min：使白色区域收缩，去除小噪点
%   闭运算（close） = 先膨胀后腐蚀：填充小孔洞，保持整体形状
%   开运算（open）  = 先腐蚀后膨胀：去除小噪点，保持整体形状
se = strel('disk', 3);                          % 半径为3的圆形结构元素
mask = imclose(mask, se);                        % 闭运算：填充蝴蝶轮廓内的小缝隙
mask = imfill(mask, 'holes');                    % 填充内部封闭孔洞（如翅膀中的空白）
mask = imopen(mask, strel('disk', 2));           % 开运算：去除边缘散点噪声

% 去除面积小于 500 像素的连通区域（保留主体蝴蝶，去除残留碎片）
% bwareaopen 本质是连通域标记（BFS/并查集）后按面积过滤
mask = bwareaopen(mask, 500);

% 显示掩膜（观察色彩通道运算效果）
figure('Name', '色彩通道运算 - 掩膜');
subplot(1,3,1); imshow(S);    title('饱和度通道 S');
subplot(1,3,2); imshow(V);    title('明度通道 V');
subplot(1,3,3); imshow(mask); title('蝴蝶掩膜（去除白色背景）');

%% 3. 代数运算 —— 提取蝴蝶前景
% =========================================================================
% 【原理：Hadamard 乘积（逐元素乘法）提取前景】
%
% 掩膜 mask 是一个 logical 矩阵（0/1），将其扩展为三通道后
% 与原图像做逐元素乘法（Hadamard 乘积），实现"抠图"：
%   - 蝴蝶区域（mask=1）：像素值保留（乘1）
%   - 背景区域（mask=0）：像素值清零（乘0），变为黑色
%
% repmat 将二维掩膜沿第三维复制3份，变成与图像同尺寸的 H×W×3 矩阵
% 这等价于对每个颜色通道分别做逐元素乘法
% =========================================================================

% 将 logical 掩膜转为 uint8（0/1），再扩展为三通道（H×W×3）
mask3 = repmat(uint8(mask), [1, 1, 3]);

% 逐元素乘法（.*）：图像矩阵与掩膜矩阵的 Hadamard 乘积
% butterfly_orig 是 H×W×3 的 uint8 矩阵，mask3 也是同尺寸，
% 结果 butterfly_extracted 中蝴蝶区域保留原色，背景变为黑色（0）
butterfly_extracted = butterfly_orig .* mask3;

figure('Name', '代数运算 - 提取蝴蝶');
imshow(butterfly_extracted); title('代数乘法提取蝴蝶');

%% 4. 几何运算 + 代数运算 —— 漫天飞舞
% =========================================================================
% 【原理：仿射几何变换 + Alpha 混合合成】
%
% 对每只蝴蝶依次执行以下变换（均通过矩阵运算实现）：
%   (1) 缩放：输出网格 → 逆向映射到输入 → 双线性插值
%   (2) 旋转：齐次仿射矩阵 T = T_c * R(θ) * T_{-c} → 逆向映射 → 双线性插值
%   (3) 翻转：置换矩阵 P（水平翻转：右乘 A*P，垂直翻转：左乘 Q*A）
%   (4) 平移：通过 ROI 索引直接指定合成位置（平移变换的直接实现）
%
% 合成方法（Alpha 混合）：
%   result = 蝴蝶 × α + 背景 × (1 - α)
%   其中 α 为软边缘掩膜（高斯模糊后的掩膜），使边缘自然过渡
% =========================================================================

result = double(background);   % 转为 double，便于后续浮点数运算（避免 uint8 溢出）
num_butterflies = 15;          % 合成的蝴蝶数量
rng(42);                       % 固定随机数种子，保证每次运行结果可复现

for i = 1:num_butterflies
    %% ---- 几何运算 ----

    % (1) 缩放 —— 逆向映射 + 双线性插值（矩阵坐标变换）
    % 随机确定缩放比例（相对于背景图像宽度），模拟远近不同的蝴蝶
    scale = 0.06 + rand() * 0.15;                                     % 缩放比例 6%~21%
    new_w = round(bg_w * scale);                                       % 目标宽度（像素）
    new_h = round(size(butterfly_orig,1) / size(butterfly_orig,2) * new_w); % 保持宽高比
    % imresize_bilinear_mat：通过线性坐标映射 + 双线性插值实现缩放（详见该函数注释）
    % fill=0：超出边界的区域填充为0（黑色），与掩膜一致，避免白边
    b_resized = imresize_bilinear_mat(butterfly_extracted, [new_h, new_w], 'fill', 0);

    % mask 缩放：先转为 double（0~1），缩放后再二值化
    % 注意：双线性插值后边缘像素值为小数（介于0和1之间），需阈值化回 logical
    m_resized = imresize_bilinear_mat(double(mask), [new_h, new_w], 'fill', 0);
    m_resized = m_resized > 0.5;   % 以 0.5 为阈值二值化：>0.5 视为蝴蝶区域

    % (2) 旋转 —— 齐次仿射变换矩阵 + 逆向映射（详见 imrotate_matlab_linear 注释）
    % 关键：旋转时填充值设为 0（黑色），而不是白色
    % 这样旋转后新增区域（padding）与掩膜的0值区域一致，不会出现白色边框
    angle = rand() * 60 - 30;   % 旋转角度随机选取 -30° ~ +30°（正值=逆时针）
    % 旋转蝴蝶图像：loose 模式自动扩展画布以包含完整旋转结果
    b_rotated = imrotate_matlab_linear(b_resized, angle, 'loose', true, 'fill', 0);

    % 旋转掩膜：先将 logical 转为 uint8（0/255）再旋转，最后转回归一化 double
    % 原因：imrotate_matlab_linear 内部转 double，掩膜若为 logical 直接旋转
    %        双线性插值后会有 0~1 之间的连续值，转 uint8(255*mask) 再÷255 可提高精度
    m_rotated = imrotate_matlab_linear(uint8(255*m_resized), angle, 'loose', true, 'fill', 0);
    m_rotated = double(m_rotated) / 255;   % 归一化回 [0, 1]
    m_rotated = m_rotated > 0.5;           % 二值化：>0.5 视为蝴蝶区域

    % ★★★ 旋转后新增的填充区域（padding）掩膜值为0，因此不会显示白色边框 ★★★
    % 再次二值化，确保旋转插值引入的中间值被正确处理
    m_rotated = m_rotated > 0.5;

    % (3) 随机水平翻转 —— 置换矩阵右乘（A * P）
    % flip_perm 内部构造 W×W 的列置换矩阵 P，对每个通道执行 A(:,:,k) * P
    % 效果：图像左右镜像，模拟蝴蝶朝不同方向飞行（增加多样性）
   if rand() > 0.5
        b_rotated = flip_perm(b_rotated, 'lr');   % 水平翻转蝴蝶图像
        m_rotated = flip_perm(m_rotated, 'lr');   % 同步翻转掩膜（保持对齐）
    end

    % (4) 随机位置（平移）—— ROI 索引实现平移变换
    % 平移在矩阵运算中对应：将蝴蝶图像放置到背景的特定行列范围
    % 等价于平移仿射矩阵：T_translate = [1 0 tx; 0 1 ty; 0 0 1]
    [bh, bw, ~] = size(b_rotated);   % 旋转后蝴蝶图像的实际尺寸
    max_x = bg_w - bw;               % 水平方向最大起始位置（保证蝴蝶不超出背景右边界）
    max_y = bg_h - bh;               % 垂直方向最大起始位置（保证蝴蝶不超出背景下边界）
    if max_x < 1 || max_y < 1
        continue;   % 若蝴蝶旋转后尺寸超出背景，跳过本次（避免越界错误）
    end
    pos_x = randi([1, max_x]);   % 随机确定蝴蝶左上角的列坐标
    pos_y = randi([1, max_y]);   % 随机确定蝴蝶左上角的行坐标

    %% ---- 代数运算 + 色彩通道运算 ----

    % 确定本次蝴蝶在背景图像中的目标区域（ROI：Region of Interest）
    roi_y = pos_y : (pos_y + bh - 1);   % 行范围（垂直方向）
    roi_x = pos_x : (pos_x + bw - 1);   % 列范围（水平方向）
    roi = result(roi_y, roi_x, :);       % 取出对应区域的背景像素（bh × bw × 3 矩阵）

    % ---- 色彩通道运算：轻微调整 HSV 色调和饱和度，增加蝴蝶颜色多样性 ----
    % 将蝴蝶图像从 uint8 转为归一化 double [0,1]，以便 rgb2hsv 处理
    b_double = double(b_rotated) / 255;
    b_hsv_local = rgb2hsv(b_double);          % 转换到 HSV 颜色空间

    % 对 H 通道（色调）施加微小偏移（mod 1.0 保证色调值循环在 [0,1] 内）
    % 不同蝴蝶的色调偏移不同，模拟自然界中蝴蝶翅膀颜色的细微差异
    hue_shift = rand() * 0.06;                % 随机色调偏移量（约 0°~21.6°）
    b_hsv_local(:,:,1) = mod(b_hsv_local(:,:,1) + hue_shift, 1.0);

    % 对 S 通道（饱和度）施加随机缩放（乘以系数，再 clamp 到 [0,1]）
    sat_factor = 0.9 + rand() * 0.2;          % 饱和度缩放因子 0.9~1.1（轻微调整）
    b_hsv_local(:,:,2) = min(b_hsv_local(:,:,2) * sat_factor, 1.0);

    % 转换回 RGB 空间，并缩放回 [0,255] 范围
    b_colored = hsv2rgb(b_hsv_local) * 255;

    % ---- 软边缘掩膜：高斯模糊让边缘自然过渡（消除硬边） ----
    % 卷积（convolution）是线性算子，可以写成矩阵乘法形式：
    %   vec(output) = A_gaussian * vec(input)
    %   其中 A_gaussian 是块 Toeplitz 结构的稀疏矩阵（高斯核对应的卷积矩阵）
    % imgaussfilt 等价于用高斯核对掩膜做 2D 卷积，使边缘从 0→1 平滑过渡
    % sigma=2.0 控制高斯核的宽度（标准差），值越大边缘越柔和
    m_soft = imgaussfilt(double(m_rotated), 2.0);   % 软边缘掩膜，值域 [0,1]

    % 将软掩膜扩展为三通道（H×W×3），对 RGB 三个通道使用同一个 Alpha 值
    m3 = repmat(m_soft, [1, 1, 3]);

    % ---- 代数运算：Alpha 混合（线性插值合成） ----
    % 合成公式（逐像素、逐通道）：
    %   result = 蝴蝶像素 × α + 背景像素 × (1 - α)
    % 其中 α = m3（软边缘掩膜），范围 [0,1]：
    %   α=1（蝴蝶中心区域）：完全显示蝴蝶颜色
    %   α=0（背景区域）    ：完全显示背景颜色
    %   0<α<1（边缘过渡区）：蝴蝶与背景线性混合，消除硬边
    %
    % 向量化理解：对于每个像素的 RGB 向量 c = [R, G, B]^T，
    %   c_out = α * c_butterfly + (1-α) * c_background
    % 这是一个凸组合（权重之和=1，每个权重在[0,1]内），保证输出值域在 [0,255] 内
    blended = b_colored .* m3 + roi .* (1 - m3);

    % 将合成结果写回背景图像的对应区域（ROI）
    result(roi_y, roi_x, :) = blended;
end

%% 5. 显示最终结果
% 将 double 类型转回 uint8（图像显示标准格式，值域 [0,255]）
% round 确保浮点运算误差不影响最终像素值
result = uint8(result);

figure('Name', '最终合成效果');
imshow(result);
title('蝴蝶漫天飞舞', 'FontSize', 16);

%% 6. 保存
imwrite(result, 'D:\Linear_algebra\Butterfly\result_butterfly_fly.jpg');
disp('✓ 合成图像已保存');

%% 7. 总结对比图
% 用六宫格展示完整的数字图像处理流程：原图 → 通道分析 → 掩膜 → 抠图 → 背景 → 合成
figure('Name', '处理流程总结', 'Position', [100 100 1200 800]);
subplot(2,3,1); imshow(butterfly_orig);      title('① 原始蝴蝶素材');
subplot(2,3,2); imshow(S);                   title('② 饱和度通道 S');
subplot(2,3,3); imshow(mask);                title('③ 通道运算→掩膜');
subplot(2,3,4); imshow(butterfly_extracted); title('④ 代数乘法抠图');
subplot(2,3,5); imshow(background);          title('⑤ 背景风景');
subplot(2,3,6); imshow(result);              title('⑥ 最终合成');
sgtitle('蝴蝶漫天飞舞 — 数字图像处理', 'FontSize', 14, 'FontWeight', 'bold');
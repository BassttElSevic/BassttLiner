function out = imresize_bilinear_mat(img, outSize, varargin)
%IMRESIZE_BILINEAR_MAT  用矩阵映射实现图像缩放的教学版函数（替代 MATLAB 内置 imresize）
%
% ===== 函数用途 =====
%   使用逆向坐标映射 + 双线性插值对图像进行缩放，功能上等价于
%   MATLAB 内置 imresize(img, [outH outW], 'bilinear')，
%   但完全通过矩阵/向量化运算实现，便于理解缩放变换的线性代数本质。
%
% ===== 调用方式 =====
%   out = imresize_bilinear_mat(img, [outH outW])
%   out = imresize_bilinear_mat(img, [outH outW], 'fill', 0)
%
% ===== 输入参数 =====
%   img     : 输入图像，支持灰度（H×W）或彩色（H×W×C），可为 uint8/double 等类型
%   outSize : 目标输出尺寸 [outH, outW]，outH 为输出行数，outW 为输出列数
%   'fill'  : (可选，默认 0) 超出输入范围时的填充值（正常缩放不会用到，
%              此参数保留用于一致性/边界安全）
%
% ===== 输出参数 =====
%   out : 缩放后的图像，类型与输入一致（通过 cast 转换回原始数值类型）
%
% ===== 核心数学原理 =====
%
%   【缩放变换的线性代数表达】
%   图像缩放本质上是一个线性（仿射）变换。对于将 [inH × inW] 图像缩放为
%   [outH × outW]，可以用以下缩放矩阵描述（齐次坐标）：
%
%       S = [sx   0   0 ]     sx = (inW-1)/(outW-1)（x方向缩放因子）
%           [ 0  sy   0 ]     sy = (inH-1)/(outH-1)（y方向缩放因子）
%           [ 0   0   1 ]
%
%   这里使用"像素中心对齐"映射约定（pixel center-to-center mapping），
%   使得输出图像第一个像素对应输入第一个像素，最后一个像素对应输入最后一个像素。
%
%   【逆向映射（Inverse Mapping）】
%   与 imrotate_matlab_linear 相同，采用逆向映射策略：
%   对输出图像的每个像素 (xo, yo)，计算其在输入图像中的对应坐标 (xin, yin)：
%
%       xin = 1 + (xo - 1) * (inW - 1) / (outW - 1)
%       yin = 1 + (yo - 1) * (inH - 1) / (outH - 1)
%
%   这是一个线性函数（仿射映射），确保：
%   - 输出第 1 列 (xo=1) 映射到输入第 1 列 (xin=1)
%   - 输出最后一列 (xo=outW) 映射到输入最后一列 (xin=inW)
%   中间的输出像素均匀分布在输入坐标范围内。
%
%   【双线性插值（Bilinear Interpolation）】
%   逆向映射得到的 (xin, yin) 通常为非整数，需要对 4 个邻近整数像素做插值：
%
%       设 x1 = floor(xin), x2 = x1+1, y1 = floor(yin), y2 = y1+1
%       dx = xin - x1,  dy = yin - y1
%
%   插值公式（面积权重）：
%       V = (1-dx)*(1-dy)*I(y1,x1) + dx*(1-dy)*I(y1,x2)
%         + (1-dx)*dy   *I(y2,x1) + dx*dy     *I(y2,x2)
%
%   权重之和 = 1，保证插值结果在原始像素值范围内，不会溢出。
%
%   【为什么使用双线性插值而不是最近邻】
%   最近邻插值（nearest neighbor）在缩小图像时会产生锯齿（aliasing），
%   双线性插值通过加权平均使结果更平滑，视觉效果更好。
%
% ===== 特殊情况处理 =====
%   - 若输出尺寸与输入相同，直接返回原图（避免不必要的计算）
%   - 若输出某方向尺寸为 1，则该方向所有像素都映射到输入图像的中心列/行
%   - 若输入某方向尺寸为 1，则该方向的输出坐标全为 1
%
% ===== 坐标约定 =====
%   x = 列索引（水平方向），y = 行索引（垂直方向），均为 1-based（从1开始）

% ---- 解析可选参数 ----
p = inputParser;
addParameter(p, 'fill', 0, @(x)isnumeric(x) && isscalar(x));
parse(p, varargin{:});
fillVal = p.Results.fill;   % 超出范围的填充值（正常缩放不会触发）

% 转为 double 类型进行浮点数运算
img_in = double(img);
[inH, inW, C] = size(img_in);   % inH=输入高度, inW=输入宽度, C=通道数

outH = outSize(1);   % 目标输出高度（行数）
outW = outSize(2);   % 目标输出宽度（列数）

% 如果目标尺寸与输入相同，直接返回原图，避免不必要的计算
if outH == inH && outW == inW
    out = img;
    return;
end

% ---- 生成输出图像的像素坐标网格 ----
% xo 为列索引，yo 为行索引，均为 1-based，形状 outH × outW
[xo, yo] = meshgrid(1:outW, 1:outH);

% ---- 逆向映射：将输出坐标映射回输入坐标（缩放变换的逆变换） ----
% 采用"像素中心对齐"映射：输出第1个像素 ↔ 输入第1个像素，
%                          输出最后1个像素 ↔ 输入最后1个像素。
% 公式：xin = 1 + (xo - 1) * (inW - 1) / (outW - 1)
% 这是线性缩放：将 [1, outW] 均匀线性映射到 [1, inW]

% 水平方向（列方向）映射
if outW == 1
    % 输出只有 1 列时，映射到输入图像的水平中心
    xin = ones(outH, outW) * ((inW + 1) / 2);
elseif inW == 1
    % 输入只有 1 列时，所有输出列都映射到输入第 1 列
    xin = ones(outH, outW);
else
    % 标准线性映射：输出坐标均匀分布在输入坐标范围内
    xin = 1 + (xo - 1) * (inW - 1) / (outW - 1);
end

% 垂直方向（行方向）映射（同上，方向换为 y）
if outH == 1
    yin = ones(outH, outW) * ((inH + 1) / 2);
elseif inH == 1
    yin = ones(outH, outW);
else
    yin = 1 + (yo - 1) * (inH - 1) / (outH - 1);
end

% ---- 计算双线性插值的 4 个邻居坐标 ----
x1 = floor(xin); x2 = x1 + 1;   % 左/右邻居的列索引
y1 = floor(yin); y2 = y1 + 1;   % 上/下邻居的行索引

% 将邻居坐标 clamp 到输入图像的合法范围（防止越界访问）
% 对于恰好在边界处的像素（xin == inW），x2 = inW+1 需要 clamp 为 inW
x1c = min(max(x1, 1), inW);
x2c = min(max(x2, 1), inW);
y1c = min(max(y1, 1), inH);
y2c = min(max(y2, 1), inH);

% 计算双线性插值权重（dx, dy 为小数部分，表示在 4 邻居间的相对位置）
dx = xin - x1;   % 水平方向小数偏移，范围 [0, 1)
dy = yin - y1;   % 垂直方向小数偏移，范围 [0, 1)

% 4 个邻居的权重（面积权重）：
%   w11 = (1-dx)*(1-dy)  → 左上角 (x1, y1)
%   w21 = dx    *(1-dy)  → 右上角 (x2, y1)
%   w12 = (1-dx)*dy      → 左下角 (x1, y2)
%   w22 = dx    *dy      → 右下角 (x2, y2)
% 4 个权重之和恒等于 1，这是双线性插值保证值域不溢出的关键
w11 = (1 - dx) .* (1 - dy);
w21 = dx       .* (1 - dy);
w12 = (1 - dx) .* dy;
w22 = dx       .* dy;

% ---- 对每个颜色通道独立执行双线性插值 ----
out_d = zeros(outH, outW, C) + fillVal;   % 初始化输出（默认填充值）

% 有效采样掩码：正常缩放下所有点均有效，此处为保证通用性而保留
valid = (xin >= 1) & (xin <= inW) & (yin >= 1) & (yin <= inH);

for k = 1:C
    Ik = img_in(:,:,k);   % 取第 k 通道的二维图像矩阵

    % 将 2D 行列坐标转换为线性索引（高效向量化访问，无需循环）
    ind11 = sub2ind([inH, inW], y1c, x1c);   % 左上邻居
    ind21 = sub2ind([inH, inW], y1c, x2c);   % 右上邻居
    ind12 = sub2ind([inH, inW], y2c, x1c);   % 左下邻居
    ind22 = sub2ind([inH, inW], y2c, x2c);   % 右下邻居

    % 取 4 个邻居的像素值
    v11 = Ik(ind11);
    v21 = Ik(ind21);
    v12 = Ik(ind12);
    v22 = Ik(ind22);

    % 双线性插值：4 邻居像素值的加权线性组合
    Vk = w11 .* v11 + w21 .* v21 + w12 .* v12 + w22 .* v22;

    % 将有效区域的插值结果写入输出（无效区域保持 fillVal）
    tmp = out_d(:,:,k);
    tmp(valid) = Vk(valid);
    out_d(:,:,k) = tmp;
end

% ---- 转回与输入相同的数值类型（uint8/double 等） ----
% cast(..., 'like', img) 使输出类型与输入完全一致，
% round 用于在整数类型（如 uint8）下进行正确的四舍五入
out = cast(round(out_d), 'like', img);
end
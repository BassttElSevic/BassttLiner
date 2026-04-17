function out = imrotate_matlab_linear(img, angleDeg, varargin)
%IMROTATE_MATLAB_LINEAR  用矩阵运算实现图像旋转的教学版函数（替代 MATLAB 内置 imrotate）
%
% ===== 函数用途 =====
%   使用齐次坐标 3×3 仿射变换矩阵对图像执行旋转，并通过逆向映射 + 双线性插值
%   完成像素重采样。功能上等价于 MATLAB 内置的 imrotate(img, angle, 'bilinear', 'loose')，
%   但完全用矩阵运算实现，便于理解线性代数在图像几何变换中的作用。
%
% ===== 调用方式 =====
%   out = imrotate_matlab_linear(img, angleDeg)
%   out = imrotate_matlab_linear(img, angleDeg, 'loose', true, 'fill', 0)
%
% ===== 输入参数 =====
%   img      : 输入图像，支持灰度（H×W）或彩色（H×W×C），可为 uint8/double 等类型
%   angleDeg : 旋转角度（单位：度），正值表示逆时针旋转（与 MATLAB imrotate 一致）
%   'loose'  : (可选，默认 true) 是否扩展画布以包含完整旋转后图像（'loose' 模式）
%              true  = 扩展画布（输出图像包含完整旋转结果，尺寸可能比输入大）
%              false = 保持原始画布大小（'crop' 模式，旋转后超出边界部分被裁剪）
%   'fill'   : (可选，默认 0) 旋转后输出图像中超出输入边界的区域所填充的像素值
%              设为 0（黑色）可避免图像合成时出现白色边框
%
% ===== 输出参数 =====
%   out : 旋转后的图像，类型为 uint8，尺寸根据 'loose' 参数决定
%
% ===== 核心数学原理 =====
%
%   【齐次坐标与仿射变换矩阵】
%   在二维图像中，每个像素坐标用齐次坐标表示为列向量：
%       p = [x; y; 1]
%   其中 x 为列索引（水平方向），y 为行索引（垂直方向），均从 1 开始（1-based）。
%
%   【绕图像中心旋转的复合变换】
%   直接旋转会绕坐标原点旋转，而我们希望绕图像中心旋转，因此需要三步复合变换：
%       1. 平移到原点：将图像中心平移至坐标原点
%       2. 执行旋转：绕原点旋转角度 θ
%       3. 平移回来：将原点平移回图像中心
%
%   写成矩阵形式为：
%       T = T_center * R(θ) * T_{-center}
%
%   其中各矩阵定义如下：
%
%   平移矩阵（将中心移到原点）：
%       T_{-center} = [1  0  -cx]
%                     [0  1  -cy]
%                     [0  0   1 ]
%
%   旋转矩阵（绕原点旋转 θ 度）：
%       R(θ) = [cos θ  -sin θ  0]
%              [sin θ   cos θ  0]
%              [0        0     1]
%
%   平移矩阵（将原点移回图像中心）：
%       T_center = [1  0  cx]
%                  [0  1  cy]
%                  [0  0   1]
%
%   最终正向变换：p_out = T * p_in
%
%   【逆向映射（Inverse Mapping）】
%   实现中采用"逆向映射"策略：对输出图像中的每个像素 (xo, yo)，
%   通过逆变换矩阵 T^{-1} 求其在输入图像中的对应坐标 (xin, yin)：
%       [xin; yin; 1] = T^{-1} * [xo; yo; 1]
%   这样可以保证输出图像中每个像素都有值（不会出现"空洞"），
%   而前向映射（正向把输入像素投影到输出）会导致输出中部分像素无值（有洞）。
%
%   【双线性插值（Bilinear Interpolation）】
%   由于逆向映射求得的 (xin, yin) 通常是非整数坐标，不能直接取像素值，
%   需要对其邻近的 4 个整数坐标像素进行加权平均（双线性插值）：
%
%       设 x1 = floor(xin), x2 = x1+1, y1 = floor(yin), y2 = y1+1
%       dx = xin - x1,  dy = yin - y1
%
%   权重分配（面积权重）：
%       w11 = (1-dx)*(1-dy)   对应左上角像素 (x1, y1)
%       w21 =    dx *(1-dy)   对应右上角像素 (x2, y1)
%       w12 = (1-dx)*   dy    对应左下角像素 (x1, y2)
%       w22 =    dx *   dy    对应右下角像素 (x2, y2)
%
%   插值结果：
%       V = w11*I(y1,x1) + w21*I(y1,x2) + w12*I(y2,x1) + w22*I(y2,x2)
%
%   【loose 模式（画布扩展）】
%   旋转后图像的尺寸通常会变化（例如 45° 旋转使对角线变为新的边长）。
%   'loose' 模式通过将输入图像的 4 个角点坐标用正向变换矩阵 T 投影，
%   取其 x/y 方向的 min/max 得到输出图像的包围盒，从而自动确定输出尺寸。
%
% ===== 与 MATLAB 内置 imrotate 的差异 =====
%   MATLAB 内置 imrotate 内部有更多边界和数值处理策略；
%   本函数仅实现最典型的"线性代数 + 逆向映射 + 双线性插值"流程，便于教学演示。
%
% ===== 坐标约定 =====
%   x = 列索引（水平方向），y = 行索引（垂直方向），均为 1-based（从1开始）

% ---- 解析可选参数 ----
p = inputParser;
addParameter(p, 'loose', true, @(x)islogical(x) && isscalar(x));
addParameter(p, 'fill', 0, @(x)isnumeric(x) && isscalar(x));
parse(p, varargin{:});
doLoose  = p.Results.loose;   % 是否使用 loose 画布扩展模式
fillVal  = p.Results.fill;    % 超出边界区域的填充值（默认0=黑色）

% 转为 double 类型进行矩阵运算（避免整数溢出）
img = double(img);
[H, W, C] = size(img);   % H=高（行数），W=宽（列数），C=通道数

% ---- 角度转换：角度 → 弧度 ----
% MATLAB imrotate 正值为逆时针旋转，因此 theta 直接使用正值
theta = angleDeg * pi / 180;
ct = cos(theta);
st = sin(theta);

% ---- 确定旋转中心（1-based 坐标系下的图像中心） ----
% MATLAB 风格：中心坐标为 ((W+1)/2, (H+1)/2)
% 例如 5×5 图像的中心为 (3, 3)，6×6 图像的中心为 (3.5, 3.5)
cx = (W + 1) / 2;   % 列方向中心（x方向）
cy = (H + 1) / 2;   % 行方向中心（y方向）

% ---- 构造齐次仿射变换矩阵 T = T_center * R(θ) * T_{-center} ----
%
% 第一步：将图像中心平移至坐标原点
%   T_{-center} = [1 0 -cx; 0 1 -cy; 0 0 1]
Tneg = [1 0 -cx;
        0 1 -cy;
        0 0  1];

% 第二步：绕原点旋转 θ 度（逆时针）
%   R(θ) = [cos θ  -sin θ  0; sin θ  cos θ  0; 0  0  1]
% 注意：由于坐标系 y 轴向下，逆时针旋转对应矩阵中 sin 的符号与数学课本一致
R = [ ct -st 0;
      st  ct 0;
      0   0  1];

% 第三步：将原点平移回图像中心
%   T_center = [1 0 cx; 0 1 cy; 0 0 1]
Tpos = [1 0 cx;
        0 1 cy;
        0 0  1];

% 复合变换矩阵（矩阵乘法顺序：先 T_{-center}，再 R，再 T_center）
T = Tpos * R * Tneg;

% ---- 决定输出画布尺寸 ----
if doLoose
    % loose 模式：将输入图像的 4 个角点坐标用正向变换 T 映射到输出坐标系，
    % 取 x/y 方向的 min/max 求包围盒，从而确定输出图像尺寸。
    % 4 个角点（1-based 坐标）：左上、右上、右下、左下
    corners = [ 1   W   W   1;   % x 坐标（列）
                1   1   H   H;   % y 坐标（行）
                1   1   1   1];  % 齐次坐标第三分量，恒为 1
    tc = T * corners;             % 将 4 个角点用正向变换映射到输出坐标系
    x = tc(1, :);
    y = tc(2, :);

    % 计算输出坐标的包围盒（向外取整，保证不截断像素）
    xmin = floor(min(x));
    xmax = ceil(max(x));
    ymin = floor(min(y));
    ymax = ceil(max(y));

    % 输出图像的宽度和高度
    outW = xmax - xmin + 1;
    outH = ymax - ymin + 1;

    % xminW/yminW：输出像素坐标 1 对应的"世界坐标"偏移量
    % 输出像素 (xo, yo) 对应世界坐标：xw = xo + xminW - 1, yw = yo + yminW - 1
    xminW = xmin;
    yminW = ymin;
else
    % crop 模式：保持与输入相同的画布大小，旋转后超出边界的部分被裁剪
    outW = W;
    outH = H;
    xminW = 1;
    yminW = 1;
end

% ---- 生成输出图像的像素坐标网格 ----
% xo, yo 分别是输出图像所有像素的列索引和行索引（1-based），形状为 outH × outW
[xo, yo] = meshgrid(1:outW, 1:outH);

% 将输出像素索引转换为"世界坐标"（与输入图像同一坐标系下的坐标）
xw = xo + (xminW - 1);
yw = yo + (yminW - 1);

% ---- 逆向映射：用 T^{-1} 将输出坐标映射回输入坐标 ----
% 原理：若 p_out = T * p_in，则 p_in = T^{-1} * p_out
% 这里 MATLAB 的 inv(T) 对于旋转矩阵等价于转置（正交矩阵性质），
% 也可以用 T' 代替，但用 inv 更通用、更清晰。
Ti = inv(T);

% 将所有输出像素坐标拼成齐次坐标矩阵（3 × N，N = outH*outW）
Pout = [xw(:)'; yw(:)'; ones(1, numel(xw))];

% 一次矩阵乘法完成所有像素的逆向映射（这是"矩阵运算处理整幅图像"的核心）
Pin  = Ti * Pout;

% 将结果 reshape 回 outH × outW 的网格形式
xin = reshape(Pin(1, :), outH, outW);   % 在输入图像中对应的列坐标（可为小数）
yin = reshape(Pin(2, :), outH, outW);   % 在输入图像中对应的行坐标（可为小数）

% ---- 双线性插值：对每个通道进行采样 ----
% 初始化输出图像，默认用 fillVal 填充（超出输入边界的区域保持该值）
out = zeros(outH, outW, C) + fillVal;

% 有效采样区域：逆向映射后的坐标落在输入图像范围内的像素
% （x 方向：1 ≤ xin ≤ W；y 方向：1 ≤ yin ≤ H）
valid = (xin >= 1) & (xin <= W) & (yin >= 1) & (yin <= H);

% 计算双线性插值的 4 个邻居像素坐标（向下取整）
x1 = floor(xin);  x2 = x1 + 1;   % 左/右邻居的列索引
y1 = floor(yin);  y2 = y1 + 1;   % 上/下邻居的行索引

% 将邻居坐标 clamp 到输入图像范围内（防止越界）
% 注意：越界的点已被 valid 掩码排除，这里 clamp 只是为了安全地计算 sub2ind
x1c = min(max(x1, 1), W);
x2c = min(max(x2, 1), W);
y1c = min(max(y1, 1), H);
y2c = min(max(y2, 1), H);

% 计算双线性插值权重（基于小数部分）
% dx, dy 分别是 xin, yin 的小数部分，表示在 4 个邻居像素之间的相对位置
dx = xin - x1;
dy = yin - y1;

% 4 个邻居的权重（面积权重，4 个权重之和 = 1）：
%   w11：左上角像素 (x1, y1) 的权重，距离当前点越远权重越小
%   w21：右上角像素 (x2, y1) 的权重
%   w12：左下角像素 (x1, y2) 的权重
%   w22：右下角像素 (x2, y2) 的权重
w11 = (1 - dx) .* (1 - dy);
w21 = dx       .* (1 - dy);
w12 = (1 - dx) .* dy;
w22 = dx       .* dy;

% 对每个颜色通道分别执行双线性插值
for k = 1:C
    Ik = img(:,:,k);   % 取第 k 个通道的二维灰度图

    % 将 2D 行列索引转换为线性索引（sub2ind 用于高效的矩阵索引）
    % 注意 sub2ind 的参数顺序是 [nRows, nCols], rowIdx, colIdx
    ind11 = sub2ind([H, W], y1c, x1c);   % 左上邻居的线性索引
    ind21 = sub2ind([H, W], y1c, x2c);   % 右上邻居的线性索引
    ind12 = sub2ind([H, W], y2c, x1c);   % 左下邻居的线性索引
    ind22 = sub2ind([H, W], y2c, x2c);   % 右下邻居的线性索引

    % 取 4 个邻居的像素值
    v11 = Ik(ind11);
    v21 = Ik(ind21);
    v12 = Ik(ind12);
    v22 = Ik(ind22);

    % 双线性插值：4 个邻居像素值的加权平均
    % 这是一个线性组合，权重之和为 1，保证插值结果在原像素值范围内
    Vk = w11 .* v11 + w21 .* v21 + w12 .* v12 + w22 .* v22;

    % 只将有效区域（valid）的插值结果写入输出，超出边界的区域保持 fillVal
    tmp = out(:,:,k);
    tmp(valid) = Vk(valid);
    out(:,:,k) = tmp;
end

% 转回 uint8 类型（与图像处理流程中的类型约定一致），并四舍五入
out = uint8(round(out));
end
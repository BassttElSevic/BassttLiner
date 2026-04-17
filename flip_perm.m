function [out, Perm] = flip_perm(img, mode)
%FLIP_PERM  用置换矩阵实现图像翻转的教学版函数（支持水平翻转与垂直翻转）
%
% ===== 函数用途 =====
%   使用显式构造的置换矩阵（Permutation Matrix）对图像执行翻转操作，
%   功能上等价于 MATLAB 内置的 fliplr（水平翻转）和 flipud（垂直翻转），
%   但通过矩阵乘法实现，便于展示"翻转 = 置换矩阵作用于图像矩阵"的线性代数本质。
%
% ===== 调用方式 =====
%   out = flip_perm(img, mode)
%   [out, Perm] = flip_perm(img, mode)
%
% ===== 输入参数 =====
%   img  : 输入图像，支持灰度（H×W）或彩色（H×W×C），可为 uint8/double/logical 等类型
%   mode : 翻转方向字符串，支持以下选项：
%          'lr'  或 'fliplr' → 水平翻转（左右镜像，列顺序反转）
%          'ud'  或 'flipud' → 垂直翻转（上下镜像，行顺序反转）
%
% ===== 输出参数 =====
%   out  : 翻转后的图像，类型与输入相同
%   Perm : （可选）本次翻转所使用的置换矩阵
%          水平翻转时为 W×W 矩阵 P，垂直翻转时为 H×H 矩阵 Q
%          可用于报告/展示中展示"矩阵形式的翻转"
%
% ===== 核心数学原理 =====
%
%   【置换矩阵（Permutation Matrix）】
%   置换矩阵是一种每行、每列恰好有一个 1、其余元素均为 0 的方阵。
%   它的作用是对向量/矩阵中的元素进行重排（置换）。
%   置换矩阵是正交矩阵，满足 P^T = P^{-1}，即其转置即为逆矩阵。
%
%   【水平翻转（左右镜像）——右乘置换矩阵 P】
%   对于图像矩阵 A（形状 H×W），水平翻转等价于将列顺序反转，
%   即将第 j 列移动到第 (W+1-j) 列，可以用 W×W 的置换矩阵 P 实现：
%
%       A_flip = A * P
%
%   P 的构造方式：将单位矩阵 I_W 的列顺序反转
%       P = I_W(:, W:-1:1)
%
%   P 的直观理解：
%       P = [0 ... 0 1]     第 1 列的 1 在最后一行 → 原第 W 列变成新第 1 列
%           [0 ... 1 0]     第 2 列的 1 在倒数第 2 行 → 原第 W-1 列变成新第 2 列
%           [ ⋮       ]     ...
%           [1 0 ... 0]     第 W 列的 1 在第 1 行 → 原第 1 列变成新第 W 列
%
%   【垂直翻转（上下镜像）——左乘置换矩阵 Q】
%   垂直翻转等价于将行顺序反转，即将第 i 行移动到第 (H+1-i) 行，
%   可以用 H×H 的置换矩阵 Q 实现：
%
%       A_flip = Q * A
%
%   Q 的构造方式：将单位矩阵 I_H 的行顺序反转
%       Q = I_H(H:-1:1, :)
%
%   【左乘 vs 右乘的直觉】
%   - 右乘 P（A * P）：对矩阵的"列"进行置换 → 水平翻转（列顺序反转）
%   - 左乘 Q（Q * A）：对矩阵的"行"进行置换 → 垂直翻转（行顺序反转）
%
%   对于 RGB 彩色图像（H×W×C），每个通道独立执行上述矩阵乘法。
%
%   【与 Kronecker 积的关系（扩展阅读）】
%   若将整幅图像向量化为 vec(A)（按列展开成列向量），则翻转操作可以写成：
%       vec(A_flip) = (P ⊗ I_H) * vec(A)  （水平翻转）
%       vec(A_flip) = (I_W ⊗ Q) * vec(A)  （垂直翻转）
%   其中 ⊗ 表示 Kronecker 积（张量积）。
%   这说明即使是最简单的翻转操作，也可以用严格的矩阵线性代数框架来描述。
%
%   【效率说明】
%   本函数显式构造 W×W 或 H×H 的置换矩阵，用于教学展示"矩阵形式的翻转"。
%   在实际工程中，直接使用 img(:, end:-1:1, :) 或 img(end:-1:1, :, :)
%   效率更高，内存占用也更少（置换矩阵对大图像会占用较大内存）。

if nargin < 2
    error('flip_perm requires (img, mode).');
end

mode = lower(string(mode));
origClass = class(img);   % 记录原始数据类型，用于最后转换回来

% 转为 double 类型以支持矩阵乘法（MATLAB 对整数类型不支持 * 矩阵乘法）
A = double(img);

[H, W, C] = size(A);   % H=高度（行数），W=宽度（列数），C=通道数

switch mode
    case {"lr","fliplr"}
        % ---- 水平翻转：右乘置换矩阵 P ----
        % P 是 W×W 置换矩阵，将单位矩阵的列顺序反转
        % P(:,j) = e_{W-j+1}，即第 j 列变为标准基向量 e_{W-j+1}
        % 效果：A * P 的第 j 列 = A 的第 (W+1-j) 列，实现列顺序反转（水平镜像）
        Perm = eye(W);
        Perm = Perm(:, W:-1:1);

        outD = zeros(size(A));
        for k = 1:C
            % 每个通道独立右乘置换矩阵 P，实现该通道的水平翻转
            outD(:,:,k) = A(:,:,k) * Perm;
        end

    case {"ud","flipud"}
        % ---- 垂直翻转：左乘置换矩阵 Q ----
        % Q 是 H×H 置换矩阵，将单位矩阵的行顺序反转
        % Q(i,:) = e_{H-i+1}^T，即第 i 行变为标准基向量 e_{H-i+1} 的转置
        % 效果：Q * A 的第 i 行 = A 的第 (H+1-i) 行，实现行顺序反转（垂直镜像）
        Perm = eye(H);
        Perm = Perm(H:-1:1, :);

        outD = zeros(size(A));
        for k = 1:C
            % 每个通道独立左乘置换矩阵 Q，实现该通道的垂直翻转
            outD(:,:,k) = Perm * A(:,:,k);
        end

    otherwise
        error("Unknown mode '%s'. Use 'lr'/'fliplr' or 'ud'/'flipud'.", mode);
end

% ---- 转回原始数值类型 ----
if isinteger(img)
    % 对于整数类型（如 uint8），需要先 clamp 到合法范围再转换
    % （置换矩阵不改变元素值，clamp 主要是为了安全）
    lo = double(intmin(origClass));   % 该整数类型的最小值（如 uint8 为 0）
    hi = double(intmax(origClass));   % 该整数类型的最大值（如 uint8 为 255）
    outD = min(max(outD, lo), hi);
    out = cast(round(outD), origClass);
else
    % 对于 double/single/logical 等非整数类型，直接转换
    out = cast(outD, origClass);
end
end
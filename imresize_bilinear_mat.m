function out = imresize_bilinear_mat(img, outSize, varargin)
%IMRESIZE_BILINEAR_MAT  Teaching version of imresize using matrix mapping.
%
% out = imresize_bilinear_mat(img, [outH outW])
% out = imresize_bilinear_mat(img, [outH outW], 'fill', 0)
%
% Core idea:
%   - Build output grid (xo, yo)
%   - Map to input coordinates (xin, yin) with a scaling transform
%   - Bilinear interpolate
%
% Coordinate convention:
%   x = column index, y = row index (1-based)

p = inputParser;
addParameter(p, 'fill', 0, @(x)isnumeric(x) && isscalar(x));
parse(p, varargin{:});
fillVal = p.Results.fill;

img_in = double(img);
[inH, inW, C] = size(img_in);

outH = outSize(1);
outW = outSize(2);

% If size is unchanged
if outH == inH && outW == inW
    out = img;
    return;
end

% ---- Output grid ----
[xo, yo] = meshgrid(1:outW, 1:outH);

% ---- Inverse mapping (scale) ----
% We map pixel centers to pixel centers.
% Common choice: (x - 1)/(W-1) mapping when W>1; handle W==1 separately.
if outW == 1
    xin = ones(outH, outW) * ((inW + 1) / 2);
elseif inW == 1
    xin = ones(outH, outW);
else
    xin = 1 + (xo - 1) * (inW - 1) / (outW - 1);
end

if outH == 1
    yin = ones(outH, outW) * ((inH + 1) / 2);
elseif inH == 1
    yin = ones(outH, outW);
else
    yin = 1 + (yo - 1) * (inH - 1) / (outH - 1);
end

% ---- Bilinear neighbors ----
x1 = floor(xin); x2 = x1 + 1;
y1 = floor(yin); y2 = y1 + 1;

% clamp to bounds (for safety at borders)
x1c = min(max(x1, 1), inW);
x2c = min(max(x2, 1), inW);
y1c = min(max(y1, 1), inH);
y2c = min(max(y2, 1), inH);

dx = xin - x1;
dy = yin - y1;

w11 = (1 - dx) .* (1 - dy);
w21 = dx       .* (1 - dy);
w12 = (1 - dx) .* dy;
w22 = dx       .* dy;

% ---- Sample per channel ----
out_d = zeros(outH, outW, C) + fillVal;

% All points are within bounds due to mapping; still keep a valid mask for generality
valid = (xin >= 1) & (xin <= inW) & (yin >= 1) & (yin <= inH);

for k = 1:C
    Ik = img_in(:,:,k);

    ind11 = sub2ind([inH, inW], y1c, x1c);
    ind21 = sub2ind([inH, inW], y1c, x2c);
    ind12 = sub2ind([inH, inW], y2c, x1c);
    ind22 = sub2ind([inH, inW], y2c, x2c);

    v11 = Ik(ind11);
    v21 = Ik(ind21);
    v12 = Ik(ind12);
    v22 = Ik(ind22);

    Vk = w11 .* v11 + w21 .* v21 + w12 .* v12 + w22 .* v22;

    tmp = out_d(:,:,k);
    tmp(valid) = Vk(valid);
    out_d(:,:,k) = tmp;
end

% ---- Cast back to input class ----
out = cast(round(out_d), 'like', img);
end
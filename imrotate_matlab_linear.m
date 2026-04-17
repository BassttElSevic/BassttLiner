function out = imrotate_matlab_linear(img, angleDeg, varargin)
%IMROTATE_MATLAB_LINEAR  Teaching version of imrotate using matrix ops.
%
% out = imrotate_matlab_linear(img, angleDeg)
% out = imrotate_matlab_linear(img, angleDeg, 'loose', true, 'fill', 0)
%
% - Uses homogeneous 3x3 transform matrices
% - Uses inverse mapping + bilinear interpolation
% - Supports "loose" canvas (like imrotate(...,'loose'))
%
% Notes:
% - Coordinate convention: x = column index, y = row index (1-based)
% - Rotation around image center

% ---- parse args ----
p = inputParser;
addParameter(p, 'loose', true, @(x)islogical(x) && isscalar(x));
addParameter(p, 'fill', 0, @(x)isnumeric(x) && isscalar(x));
parse(p, varargin{:});
doLoose  = p.Results.loose;
fillVal  = p.Results.fill;

img = double(img);  % compute in double
[H, W, C] = size(img);

% ---- angle ----
theta = angleDeg * pi / 180;
ct = cos(theta);
st = sin(theta);

% ---- define center (consistent with MATLAB style) ----
% Use center at ( (W+1)/2, (H+1)/2 ) in 1-based coordinates
cx = (W + 1) / 2;
cy = (H + 1) / 2;

% ---- build homogeneous transform T = Tc * R * T(-c) ----
Tneg = [1 0 -cx;
        0 1 -cy;
        0 0  1];

R = [ ct -st 0;
      st  ct 0;
      0   0  1];

Tpos = [1 0 cx;
        0 1 cy;
        0 0  1];

T = Tpos * R * Tneg;

% ---- decide output canvas size ----
if doLoose
    % transform the 4 corners to find bounding box
    corners = [ 1   W   W   1;
                1   1   H   H;
                1   1   1   1];
    tc = T * corners;
    x = tc(1, :);
    y = tc(2, :);

    xmin = floor(min(x));
    xmax = ceil(max(x));
    ymin = floor(min(y));
    ymax = ceil(max(y));

    outW = xmax - xmin + 1;
    outH = ymax - ymin + 1;

    % offset to shift output coords into 1..outW / 1..outH
    % output pixel (xo,yo) corresponds to world coord:
    % xw = xo + xmin - 1, yw = yo + ymin - 1
    xminW = xmin;
    yminW = ymin;
else
    outW = W;
    outH = H;
    xminW = 1;
    yminW = 1;
end

% ---- create grid of output coordinates ----
% output indices: xo=1..outW, yo=1..outH
[xo, yo] = meshgrid(1:outW, 1:outH);

% map to "world" coordinates in original reference
xw = xo + (xminW - 1);
yw = yo + (yminW - 1);

% ---- inverse mapping: [xin; yin; 1] = inv(T) * [xw; yw; 1] ----
Ti = inv(T);

Pout = [xw(:)'; yw(:)'; ones(1, numel(xw))];
Pin  = Ti * Pout;

xin = reshape(Pin(1, :), outH, outW);
yin = reshape(Pin(2, :), outH, outW);

% ---- bilinear sampling for each channel ----
out = zeros(outH, outW, C) + fillVal;

% valid sampling area (need xin in [1,W], yin in [1,H])
valid = (xin >= 1) & (xin <= W) & (yin >= 1) & (yin <= H);

% compute neighbors
x1 = floor(xin);  x2 = x1 + 1;
y1 = floor(yin);  y2 = y1 + 1;

% clamp neighbors to image bounds (for points near border)
x1c = min(max(x1, 1), W);
x2c = min(max(x2, 1), W);
y1c = min(max(y1, 1), H);
y2c = min(max(y2, 1), H);

dx = xin - x1;
dy = yin - y1;

w11 = (1 - dx) .* (1 - dy);
w21 = dx       .* (1 - dy);
w12 = (1 - dx) .* dy;
w22 = dx       .* dy;

for k = 1:C
    Ik = img(:,:,k);

    % linear indices for the 4 neighbors
    ind11 = sub2ind([H, W], y1c, x1c);
    ind21 = sub2ind([H, W], y1c, x2c);
    ind12 = sub2ind([H, W], y2c, x1c);
    ind22 = sub2ind([H, W], y2c, x2c);

    v11 = Ik(ind11);
    v21 = Ik(ind21);
    v12 = Ik(ind12);
    v22 = Ik(ind22);

    Vk = w11 .* v11 + w21 .* v21 + w12 .* v12 + w22 .* v22;

    tmp = out(:,:,k);
    tmp(valid) = Vk(valid);     % only write valid pixels; invalid keeps fillVal
    out(:,:,k) = tmp;
end

% restore class similar to imrotate usage in your pipeline
out = uint8(round(out));
end
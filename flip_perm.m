function [out, Perm] = flip_perm(img, mode)
%FLIP_PERM  Flip image using permutation matrices (linear algebra style).
%
%   out = flip_perm(img, mode)
%   [out, Perm] = flip_perm(img, mode)
%
% mode:
%   'lr'  or 'fliplr'  -> out = A * P
%   'ud'  or 'flipud'  -> out = Q * A
%
% Notes:
%   - For integer images (uint8/uint16/int...), convert to double for mtimes,
%     then cast back to original type.

if nargin < 2
    error('flip_perm requires (img, mode).');
end

mode = lower(string(mode));
origClass = class(img);

% Convert to double for matrix multiplication safety
A = double(img);

[H, W, C] = size(A);

switch mode
    case {"lr","fliplr"}
        % P reverses columns
        Perm = eye(W);
        Perm = Perm(:, W:-1:1);

        outD = zeros(size(A));
        for k = 1:C
            outD(:,:,k) = A(:,:,k) * Perm;
        end

    case {"ud","flipud"}
        % Q reverses rows
        Perm = eye(H);
        Perm = Perm(H:-1:1, :);

        outD = zeros(size(A));
        for k = 1:C
            outD(:,:,k) = Perm * A(:,:,k);
        end

    otherwise
        error("Unknown mode '%s'. Use 'lr'/'fliplr' or 'ud'/'flipud'.", mode);
end

% Cast back (clamp for uint8-like types)
if isinteger(img)
    % clamp to valid range for the integer type
    lo = double(intmin(origClass));
    hi = double(intmax(origClass));
    outD = min(max(outD, lo), hi);
    out = cast(round(outD), origClass);
else
    % for double/single/logical: keep numeric type behavior reasonable
    out = cast(outD, origClass);
end
end
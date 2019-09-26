%% Extract ROI
function [EDES_MDx,EDES_MD,EDES_C0x,EDES_C0xD,EDES_C0] = extractROI(ED,ES,T,D)

for i=1:size(ED,3)                                        % #Slices/Frame varies across all patients
    EDES_MDx(:,:,i) = abs(double(ED(:,:,i))...            % Computing the Mean Absolute Difference between ED & ES
        -double(ES(:,:,i)));                              % frames, where the heart is the only moving object
end

EDES_MDx = mean(EDES_MDx,3);

% % Filtering
% EDES_MD = wiener2(EDES_MDx,[5 5]);
% % EDES_MD = filter2(fspecial('average',3),EDES_MDx)/255;
% % EDES_MD = medfilt2(EDES_MDx);
% % EDES_MD = imgaussfilt(EDES_MDx,0.7);
% 
% % Contrast Enhancement
% EDES_MD = im2double(EDES_MD) .^ 0.95;
EDES_MD=EDES_MDx;

threshold = (T/100)*(max(max(EDES_MD)));                  % Thresholding
EDES_C0x = zeros(size(EDES_MD));
for i=1:size(EDES_C0x,1)
    for j=1:size(EDES_C0x,2)
        if(EDES_MD(i,j)>threshold)
            EDES_C0x(i,j) = 1;
        end
    end
end

if D>0                                                   % Dilation
    SE = strel('disk',D);
    EDES_C0xD = imopen(EDES_C0x,SE);
    EDES_C0 = EDES_C0xD;
else
    %NoDilation
    EDES_C0xD = 0;
    EDES_C0 = EDES_C0x;
end

CC = bwconncomp(EDES_C0);                                % Extracting largest connected component, C0
numPixels = cellfun(@numel,CC.PixelIdxList);
[biggest,~] = max(numPixels);
for i=1:size(numPixels,2)
    if (numPixels(1,i) ~= biggest)
        EDES_C0(CC.PixelIdxList{i}) = 0;                 % Removing all side-components
    end
end
end
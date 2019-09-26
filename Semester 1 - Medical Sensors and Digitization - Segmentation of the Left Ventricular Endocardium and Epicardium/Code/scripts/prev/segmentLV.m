function [ED3_M] = segmentLV(ED,C0,F)

ED3 = ED(:,:,3);                                % Selecting Mid-Basal ED Slice
ED3_0 = zeros(size(ED3));
for i = 1:size(ED3,1)
    for j = 1:size(ED3,2)
        if(C0(i,j) == 1)                        % Slice is masked with C0
            ED3_0(i,j)= ED3(i,j);               
        end
    end
end

nrows = size(ED3_0,1);
ncols = size(ED3_0,2);
ab = reshape(ED3_0,nrows*ncols,1);
[cluster_idx,~] = kmeans(ab,3,'distance','cityblock','Replicates',5);                           
ED3_M = reshape(cluster_idx,nrows,ncols);       % K-Means Quantization

% ED3_M1(:,:) = (ED3_M == 1);                     % Mask derived from Level 1
% ED3_M2(:,:) = (ED3_M == 2);                     % Mask derived from Level 2
% ED3_M3(:,:) = (ED3_M == 3);                     % Mask derived from Level 3

% ED3_L1 = activecontour(ED3,ED3_M1);                 % Active Contour to segment LV
% ED3_L2 = activecontour(ED3,ED3_M2);                 % Active Contour to segment myocardium
% ED3_L3 = activecontour(ED3,ED3_M3);   

% max(double(ED3_M-1))
ED3_M = uint8((double(ED3_M)-1)./2.*255);


if F==1
    figure,
    set(gcf,'color','w')
    subplot(141),imshow(ED3,[]),title("Mid-Basal ED Slice")
    subplot(142),imshow(ED3_0,[]),title("Post-Masking using C0")
    subplot(143),imshow(ED3_M1,[]),title("Post-KMeans Quantization")
    subplot(144),imshow(LV,[]),title("Segmented LV")
else
    % No Display
end

end
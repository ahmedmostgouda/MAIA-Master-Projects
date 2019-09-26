function [EDES_C0,EDES_C0x] = extractROI(ED,ES,T,F)

for i=1:size(ED,3)                                        % #Slices/Frame varies across all patients
    EDES_MD(:,:,i) = abs(double(ED(:,:,i))...             % Computing the Mean Absolute Difference between ED & ES
        -double(ES(:,:,i)));                              % frames, where the heart is the only moving object
end
EDES_MD = mean(EDES_MD,3);

threshold = (T/100)*(max(max(EDES_MD)));                  % Thresholding
EDES_C0x = zeros(size(EDES_MD));
for i=1:size(EDES_C0x,1)
    for j=1:size(EDES_C0x,2)
        if(EDES_MD(i,j)>threshold)
            EDES_C0x(i,j) = 1;
        end
    end
end

EDES_C0 = EDES_C0x;

CC = bwconncomp(EDES_C0);                                % Extracting largest connected component, C0
numPixels = cellfun(@numel,CC.PixelIdxList);
[biggest,~] = max(numPixels);
for i=1:size(numPixels,2)
    if (numPixels(1,i) ~= biggest)
        EDES_C0(CC.PixelIdxList{i}) = 0;                 % Removing all side-components
    end
end

if F==1
    figure,
    set(gcf,'color','w')
    subplot(131),imshow(EDES_MD,[]),title("MD between ED-ES Phases")
    subplot(132),imshow(EDES_C0x,[]),title(['Post-Threshold at ',num2str(T),'%'])
    subplot(133),imshow(EDES_C0,[]),title("Largest Connected Component")
else
    % No Display
end

end
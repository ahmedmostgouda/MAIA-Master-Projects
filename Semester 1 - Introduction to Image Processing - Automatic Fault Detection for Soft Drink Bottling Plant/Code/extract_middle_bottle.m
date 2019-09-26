function [I_out] =  extract_middle_bottle(I_in)
I_in_size = size(I_in,2);

%Binarize image using Utsu algorithm
level = graythresh(I_in);
BW = not(imbinarize(I_in,level));

%Find the connected black
CC = bwconncomp(not(BW));
numPixels = cellfun(@numel,CC.PixelIdxList);
[biggest,idx] = max(numPixels);

%Keep the bigest connected area(background)
for k=1:size(numPixels,2)
    if (numPixels(1,k) ~= biggest)
         BW(CC.PixelIdxList{k}) = 1;
    end
end

%Flip the image
IrBW = flipdim(BW,2);

%Convole the  image with it self
for k = 1:I_in_size
    conv_BW(1,k) = sum(sum(BW(:,1:k).*IrBW(:,I_in_size-k+1:I_in_size)))/k;
end

for k = 1:I_in_size
    conv_BW(1,k+352) = sum(sum(BW(:,k:I_in_size).*IrBW(:,1:I_in_size-k+1)))/(I_in_size-k+1);
end

[pks,loc] = findpeaks(conv_BW,'MinPeakDistance',110);

if ((round(loc(1,3)-loc(1,2)/2)-round(loc(1,2)/2))>200)
    disp('No bottle')
    I_out = 0;
else
    I_out = I_in(:,round(loc(1,3)/2)-64:round(loc(1,3)/2)+65);

end
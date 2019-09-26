function [I_out_l,I_out_m,I_out_r] =  extract_all_bottles(I_in)
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


if (size(loc)<5)
    disp('No bottle')
    I_out_m = 0;
    I_out_l = 0;
    I_out_r = 0;
   
else
    diff_value = 130 - (round(loc(1,4)/2)-round(loc(1,2)/2) +1);
    I_out_m = I_in(:,round(loc(1,3)/2)-64:round(loc(1,3)/2)+65);
    I_out_l_mid_forward = I_in(:,1:loc(1,1));
    I_out_l_mid_reverse = flipdim(I_out_l_mid_forward,2);
    I_out_l_mid = uint8((double(I_out_l_mid_forward)+double(I_out_l_mid_reverse))./2);
    I_out_l_right = I_in(:,loc(1,1)+1:round(loc(1,2)/2));
    I_out_l_right_reverse = flipdim(I_out_l_right,2);
    I_out_l_init = [I_out_l_right_reverse I_out_l_mid I_out_l_right];
    diff_value = 130 - size(I_out_l_init,2); 
    if(diff_value>0)
        I_out_l =[flipdim(I_in(:,(round(loc(1,2)/2)+1):round(loc(1,2)/2)+floor(diff_value/2)),2) I_out_l_init I_in(:,(round(loc(1,2)/2)+1):round(loc(1,2)/2)+ceil(diff_value/2))];
    else
       I_out_l = I_out_l_init(:,1-ceil(diff_value/2):size(I_out_l_init,2)+floor(diff_value/2));
    end

    I_out_l_mid_forward = I_in(:,1:loc(1,1));
 
    I_in_mirror = flipdim(I_in,2);
    conv_BW_mirror = flipdim(conv_BW,2); 
    [pks1_mirror,loc_mirror] = findpeaks(conv_BW_mirror,'MinPeakDistance',110);

    if(loc_mirror(1,1)<10)
        loc_mirror(1)=[];
    end
    
    I_out_r_mid_forward = I_in_mirror(:,1:loc_mirror(1,1));
    I_out_r_mid_reverse = flipdim(I_out_r_mid_forward,2);
    I_out_r_mid = uint8((double(I_out_r_mid_forward)+double(I_out_r_mid_reverse))./2);
    I_out_r_right = I_in_mirror(:,loc_mirror(1,1)+1:round(loc_mirror(1,2)/2));
    I_out_r_right_reverse = flipdim(I_out_r_right,2);
    I_out_r_init = [I_out_r_right_reverse I_out_r_mid I_out_r_right];
    
    diff_value = 130 - size(I_out_r_init,2); 
    if(diff_value>0)
        I_out_r =[flipdim(I_in_mirror(:,(round(loc_mirror(1,2)/2)+1):round(loc_mirror(1,2)/2)+floor(diff_value/2)),2) I_out_r_init I_in_mirror(:,(round(loc_mirror(1,2)/2)+1):round(loc_mirror(1,2)/2)+ceil(diff_value/2))];
    else
       I_out_r = I_out_r_init(:,1-ceil(diff_value/2):size(I_out_r_init,2)+floor(diff_value/2));
    end
end
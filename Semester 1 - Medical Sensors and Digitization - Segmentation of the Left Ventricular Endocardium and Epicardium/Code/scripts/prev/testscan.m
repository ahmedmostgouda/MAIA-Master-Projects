
addpath ./aux_pack/
patients = get_data('../data/training_set/');
for i=1:numel(patients)
    %% Computation
    ED = patients(i).ED;
    ES = patients(i).ES;
    [MD,T0,C0] = extractROI(ED,ES,30);
    [M] = segmentLV(ED(:,:,3),C0,2);
    
    %% Display
    if F==1
        figure,
        set(gcf,'color','w')
        subplot(151),imshow(ED3,[]),title("Mid-Basal ED Slice")
        subplot(152),imshow(EDES_MD,[]),title("Absolute Mean Difference")
        subplot(153),imshow(PT,[]),title("Post-Threshold")
        subplot(154),imshow(C0,[]),title("C0")
        subplot(155),imshow(ED3_M,[]),title("Post-KMeans Quantization")
    else
        % No Display
    end
    
    %% Export
%     imwrite(C0,strcat('..\results\C0\',...           % Save C0 images
%             patients(i).name,'.png'));
%     imwrite(LV,strcat('..\results\LV\',...           % Save LV images
%             patients(i).name,'.png'));
end

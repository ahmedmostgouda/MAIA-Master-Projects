%% Training Environment
clear all
close all
clc
addpath ./aux_pack/
load patients   

% i=37;
%for i=1:numel(patients)
for i=1:numel(patients)
%for i=1:1
    %% Computation
    ED = patients(i).ED;
    ES = patients(i).ES;
    T1=30; T2=20; L=2; F=0; D=3; A=240;                  % Threshold,Level,Figure,Dilation,Eccentricity,Area
    ED_image = ED(:,:,3);                                       % Target Slice      
    ED_image_norm = (double(ED_image)./double(max(max(ED_image))))*255;
    [MDx,MD,T0,~,C01] = extractROI(ED,ES,T1,0);
    [M1] = maskKM(ED_image_norm,C01,L);
    
    [~,~,D0,C02] = extractROI(ED,ES,T2,1);
    [M2] = maskKM(ED_image_norm,C02,L);
    
    %% Segmentation Detection/Cleanup
%   [centers,radii] = imfindcircles(M2,[20 25],'ObjectPolarity','bright','Sensitivity',0.98);       %Hough Transform
    LV1=segmentLV(M1,A);    
    LV2=segmentLV(M2,A);
    
    LV_mask=selectLV(LV1,LV2);
    
    ES_image = ES(:,:,3);
    
    figure
    subplot(2,3,1),imshow(ES_image,[])
    
    ED_image_norm_inten = ((double(ED_image)./double(max(max(ED_image)))).^0.3)*255;
    ES_image_norm_inten = ((double(ES_image)./double(max(max(ES_image)))).^0.3)*255;
    LV_ED_image = activecontour(ED_image_norm,LV_mask,300,'Chan-Vese');
    %LV_ES_image = activecontour(ES_image_norm,LV_mask,300,'Chan-Vese');
    
    LV_ES_image_loc = LV_ED_image .* ES_image_norm_inten;
    
    subplot(2,3,2),imshow(LV_ES_image_loc,[])
    
    nrows = size(LV_ES_image_loc,1);
    ncols = size(LV_ES_image_loc,2);
    ab = reshape(LV_ES_image_loc,nrows*ncols,1);
    [cluster_idx, cluster_center] = kmeans(ab,3,'distance','cityblock','Replicates',5);
    LV_ES_image_loc_masks = reshape(cluster_idx,nrows,ncols);
    
    subplot(2,3,3),imshow(LV_ES_image_loc_masks,[])
    
    for k = 1 :3       
        LV_ES_image_loc_masks_mean(k) =  sum(sum(double(LV_ES_image_loc_masks == k).*LV_ES_image_loc))./sum(sum(double(LV_ES_image_loc_masks == k)));
    end
    [LV_ES_image_loc_masks_mean_max,LV_ES_image_loc_masks_mean_maxid] = max(LV_ES_image_loc_masks_mean);
    
    LV_ES_image_loc_mask_max = (LV_ES_image_loc_masks == LV_ES_image_loc_masks_mean_maxid);

    subplot(2,3,4),imshow(LV_ES_image_loc_mask_max,[])
    
    CC = bwconncomp(LV_ES_image_loc_mask_max);
    numPixels = cellfun(@numel,CC.PixelIdxList);
    [biggest,idx] = max(numPixels);

    for k=1:size(numPixels,2)
        if (numPixels(1,k) ~= biggest)
             LV_ES_image_loc_mask_max(CC.PixelIdxList{k}) = 0;
        end
    end
    
    subplot(2,3,5),imshow(LV_ES_image_loc_mask_max,[])
    
    %LV_ES_image = activecontour(LV_ES_image_loc,LV_ES_image_loc_mask_max,300,'Chan-Vese');
    
    %subplot(3,3,6),imshow(LV_ES_image,[])
    

    %% Display
    if F==1
        figure,
        set(gcf,'color','w')
        set(gca,'FontName','Monospaced','FontSize',7)
        subplot(2,6,1),imshow(S,[]),title("Target Slice",'FontName','Monospaced','FontSize',7),
        subplot(2,6,2),imshow(MDx,[]),title("Absolute Mean Difference",'FontName','Monospaced','FontSize',7)
        subplot(2,6,3),imshow(T0,[]),title(['Post-Threshold at ',num2str(T1),'%'],'FontName','Monospaced','FontSize',7)
        subplot(2,6,4),imshow(C01,[]),title("Largest Connected Component",'FontName','Monospaced','FontSize',7)
        subplot(2,6,5),imshow(M1,[]),title(['Post-KMeans Quantization at Level ',num2str(L)],'FontName','Monospaced','FontSize',7);
        
        subplot(2,6,9),imshow(D0,[]),title(['Post-Dilation+Threshold at ',num2str(T2),'%'],'FontName','Monospaced','FontSize',7);
        subplot(2,6,10),imshow(C02,[]),title("Largest Connected Component",'FontName','Monospaced','FontSize',7);
        subplot(2,6,11),imshow(M2,[]),title(['Post-KMeans Quantization at Level ',num2str(L)],'FontName','Monospaced','FontSize',7);
        subplot(2,6,6),imshow(LV1,[]),title("Segmented LV",'FontName','Monospaced','FontSize',7);
        subplot(2,6,12),imshow(LV2,[]),title("Segmented LV",'FontName','Monospaced','FontSize',7);
    elseif F==2
        figure
        subplot(1,2,1),imshow(ED_image,[])
        subplot(1,2,2),imshow(LV_ED_image,[])

    elseif F==3
        figure
        subplot(1,4,1),imshow(ES_image,[])
        subplot(1,4,2),imshow(LV_ES_image_loc,[])
        subplot(1,4,3),imshow(LV_ES_image_loc_masks,[])
        subplot(1,4,4),imshow(LV_ES_image_loc_mask_max,[])
        
    end
    
    %% Export
%     imwrite(LV,strcat('..\results\LVtest3\',...           % Save LV masks
%             patients(i).name,'.png'));

     imwrite(LV_ES_image_loc_mask_max,strcat('..\results\LV_ES_test1\',...           % Save LV masks
             patients(i).name,'.png'));
         
     imwrite(LV_ED_image,strcat('..\results\LV_ED_test1\',...           % Save LV masks
         patients(i).name,'.png'));
end

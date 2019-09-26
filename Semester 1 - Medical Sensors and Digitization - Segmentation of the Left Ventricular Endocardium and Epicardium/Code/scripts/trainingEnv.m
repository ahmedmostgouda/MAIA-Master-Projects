%% Training Environment
clear all
close all
clc
addpath ./aux_pack/
load patients   

% i=37;
for i=1:numel(patients)
    %% Computation
    ED = patients(i).ED;
    ES = patients(i).ES;
    T1=30; T2=20; L=2; F=1; D=3; A=240;                  % Threshold,Level,Figure,Dilation,Eccentricity,Area
    S = ED(:,:,3);                                       % Target Slice      
    
    [MDx,MD,T0,~,C01] = extractROI(ED,ES,T1,0);
    [M1] = maskKM(S,C01,L);
    
    [~,~,D0,C02] = extractROI(ED,ES,T2,1);
    [M2] = maskKM(S,C02,L);
    
    %% Segmentation Detection/Cleanup
%   [centers,radii] = imfindcircles(M2,[20 25],'ObjectPolarity','bright','Sensitivity',0.98);       %Hough Transform
    LV1=segmentLV(M1,A);    
    LV2=segmentLV(M2,A);
    
    LV=selectLV(LV1,LV2);

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
    else
        % No Display
    end
    
    %% Export
%     imwrite(LV,strcat('..\results\LVtest3\',...           % Save LV masks
%             patients(i).name,'.png'));
end

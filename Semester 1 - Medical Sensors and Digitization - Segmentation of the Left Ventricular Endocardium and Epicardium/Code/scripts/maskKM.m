%% K-Means Masking
function [ED3_M] = maskKM(ED3,C0,L)              % ED3=Slice; C0=Mask; L=Level

C0x = activecontour(ED3,C0);

ED3_0 = zeros(size(ED3));
for i = 1:size(ED3,1)
    for j = 1:size(ED3,2)
        if(C0x(i,j) == 1)                        % Slice is masked with C0
            ED3_0(i,j)= ED3(i,j);               
        end
    end
end

nrows = size(ED3_0,1);
ncols = size(ED3_0,2);
ab = reshape(ED3_0,nrows*ncols,1);
[cluster_idx,~] = kmeans(ab,L,'distance','cityblock','Replicates',5);                           
ED3_Mx = reshape(cluster_idx,nrows,ncols);       % K-Means Quantization

if L==3
    ED3_Mx = uint8((double(ED3_Mx)-1)./2.*255);       % For Level-3 Display
else
    ED3_Mx = uint8((double(ED3_Mx)-1).*255);      	  % For Level-2 Display
end

if mean(mean(ED3_Mx))>127                                    % Swap FG/BG
    ED3_M = imcomplement(ED3_Mx);
else
    %NoInversion
    ED3_M = ED3_Mx;
end
end
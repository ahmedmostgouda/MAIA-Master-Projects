%% Select Best LV Result
function [LV] = selectLV(LV1,LV2)

[~,L] = bwboundaries(LV1,'noholes');
stats1 = regionprops(L,'Area','Perimeter','Eccentricity','MajorAxisLength','MinorAxisLength','Solidity');
[~,L] = bwboundaries(LV2,'noholes');
stats2 = regionprops(L,'Area','Perimeter','Eccentricity','MajorAxisLength','MinorAxisLength','Solidity');

LV1_A = [stats1.MajorAxisLength]-[stats1.MinorAxisLength];                     % Axial Difference
LV2_A = [stats2.MajorAxisLength]-[stats2.MinorAxisLength];

if abs(LV1_A - LV2_A) >= 4
    if LV1_A > LV2_A
        LV = LV2;
    else
        LV = LV1;
    end
else
    LV1_C = abs(1-([stats1.Perimeter].^2) ./ (4*pi.*[stats1.Area]));           % Circularity Index
    LV2_C = abs(1-([stats2.Perimeter].^2) ./ (4*pi.*[stats2.Area]));
    
    LV1_S = abs(1-([stats1.Solidity]));                                        % Solidity Index
    LV2_S = abs(1-([stats2.Solidity]));
    
    if abs(LV1_C - LV2_C) > abs(LV1_S - LV2_S)
        LV1_EA = LV1_C;
        LV2_EA = LV2_C;
    else
        LV1_EA = LV1_S;
        LV2_EA = LV2_S;
    end
    
    if isempty(LV1_EA)||(isempty(LV1_EA) && isempty(LV2_EA))                    % Empty/Equal
        LV = LV2;
    elseif isempty(LV2_EA)||(LV1_EA == LV2_EA)
        LV = LV1;
    elseif abs(LV1_EA-LV2_EA)<8e-04                                             % Equal Circularity
        if (stats1.Area) >= (stats2.Area)
            LV = LV1;
        else
            LV = LV2;
        end
    else                                                                        % Different Circularity
        if (LV1_EA < LV2_EA)
            LV = LV1;
        else
            LV = LV2;
        end
    end
end
end
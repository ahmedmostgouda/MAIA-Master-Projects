%% Segment LV
function [LV] = segmentLV(O,minArea)
%% Region Properties Extraction - Small LV
Os = bwareaopen(O,60);                                                        % Remove Noise (<60px)   
Os = imfill(Os,'holes');                                                      % Fill Holes

labeledOs = bwlabel(Os, 8);                                                   % Generate 4-Neighbor Labeled Image
[~,L] = bwboundaries(Os,'noholes');
statss = regionprops(L,'Area','Perimeter','Eccentricity','MajorAxisLength','MinorAxisLength');
[R,C] = size(statss);

if (R*C)==1
    flag=1;
    LV = ismember(labeledOs,1);
else
    flag=0;
end

%% Region Properties Extraction - Regular LV
if flag==0
    O = bwareaopen(O,minArea);                                                  % Remove Side-Components
    O = imfill(O,'holes');                                                      % Fill Holes
    
    labeledO = bwlabel(O, 8);                                                   % Generate 4-Neighbor Labeled Image
    [~,L] = bwboundaries(O,'noholes');
    stats = regionprops(L,'Area','Perimeter','Eccentricity','MajorAxisLength','MinorAxisLength','Solidity');
    
    %% Circularity Detection
    allEccs = [stats.Eccentricity];
    allAxDif = ([stats.MajorAxisLength]-[stats.MinorAxisLength]);
    
    lowE1 = find(allEccs == min(allEccs));                                      % Object Index w/ Lowest Ecc
    lowE2 = find(allEccs == min(setdiff(allEccs(:),min(allEccs(:)))));          % Object Index w/ Second-Lowest Ecc
    
    if abs([stats(lowE1).Eccentricity]-[stats(lowE2).Eccentricity])...
            < abs([stats(lowE1).Solidity]-[stats(lowE2).Solidity])
        
        keepIndexes = find([stats.Solidity] == max([stats.Solidity]));          % Solidity Index
        
    else
        if isempty(lowE2)
            mostCirc = (allEccs == min(allEccs));
        else
            mostCirc = (allAxDif == (min(...                                            
                [stats(lowE1).MajorAxisLength]-[stats(lowE1).MinorAxisLength],...       % Highest Axial Difference and Low Ecc
                [stats(lowE2).MajorAxisLength]-[stats(lowE2).MinorAxisLength])));
        end
        
        keepIndexes = find(mostCirc);
        
    end
    LV = ismember(labeledO,keepIndexes);
end

SE = strel('disk',4);
LV = imclose(LV,SE);                                                        % Close Gaps

end
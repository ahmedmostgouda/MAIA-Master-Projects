function [patients]= get_data(folder_name)

listFolders = dir(strcat('../data/training_set/'));
% listFolders = dir(strcat(folder_name));

%% Remove Unuseful Paths
remove=[];
for i=1:numel(listFolders)
    if( strcmp(listFolders(i).name,'.') || strcmp(listFolders(i).name,'..'))
        remove=[remove,i];
    end
end
listFolders(remove)=[];

%% Import Data
for i=1:numel(listFolders)
    patients(i).name = listFolders(i).name;
 
    F = dir(strcat(strcat(listFolders(i).folder,...                     % #Frame is not constant across
        '\',listFolders(i).name,'\')));                                 % all patient directories 
    
    name1 = erase(F(5).name,'.nii.gz');
    name1 = erase(name1,'_gt.nii.gz');
    name2 = erase(F(7).name,'.nii.gz');
    name2 = erase(name2,'_gt.nii.gz');
    
    patients(i).ED = niftiread(strcat(listFolders(i).folder,...
        '\',listFolders(i).name,'\',name1));
    
    patients(i).ES = niftiread(strcat(listFolders(i).folder,...
        '\',listFolders(i).name,'\',name2));
end
end
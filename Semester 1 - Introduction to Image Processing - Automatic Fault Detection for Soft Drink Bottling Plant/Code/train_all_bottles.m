function [I_training,I_training_avg,eig_vect_extract,proj_eigvect] =  train_all_bottles(rescale_factor)

Trainingset = 'TrainingData';
folders = dir(Trainingset);
training_images = cell(7*6,1);
for i = 4:10
    files = dir(join([Trainingset,'/',folders(i).name]));
    for j = 4:12
        training_images{(i-4)*9+j-3,1} = join([files(j).folder,'\',files(j).name]);
    end
end

files = dir(join([Trainingset,'/',folders(12).name]));
for j = 4:27
    training_images{7*9+j-3,1} = join([files(j).folder,'\',files(j).name]);
end

 for i = 1:size(training_images,1)
        I = rgb2gray(imread(training_images{i,1}));
        I_training(i,:,:) = im2double(imresize(extract_middle_bottle(I),rescale_factor));
        D(i,:) = reshape(I_training(i,:,:),1,size(I_training,2)*size(I_training,3));
 end

I_training_avg = mean(D);
for i = 1:size(D,1)
    D(i,:) = D(i,:)-I_training_avg;
end
A = D*D';

[eig_vect,eig_val] = eigs(A,36,'lm');
eig_val_sum = 0;
total_eig_val_sum = trace(eig_val);

% for i = 1 : 87
%     eig_val_sum = eig_val_sum + eig_val(i,i);
%     if (eig_val_sum/total_eig_val_sum>0.95)
%          i
%          break;
%     end
% end

eig_vect_extract = D'*eig_vect;
proj_eigvect = D * eig_vect_extract;





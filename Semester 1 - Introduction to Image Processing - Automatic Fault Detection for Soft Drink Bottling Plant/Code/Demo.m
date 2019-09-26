clc
clear all
close all

rescale_factor = 1;
[I_training,I_training_avg,eig_vect_extract,proj_eigvect] =  train_all_bottles(rescale_factor);

Trainingset = 'TrainingData';
folders = dir(Trainingset);

files = dir(join([Trainingset,'/',folders(13).name]));
for j = 4:18
    testing_images{j-3,1} = join([files(j).folder,'\',files(j).name]);
end

score_ref(:,1) = [4 8 8 5 1 8 6 6 8 3 6 6 8 3 3]';
score_ref(:,2) = [8 8 8 8 8 8 8 8 8 8 8 8 8 8 8]';
score_ref(:,3) = [8 4 5 8 8 1 8 1 3 2 7 4 2 8 4]';

for i = 1:size(testing_images,1)
    I_in = rgb2gray(imread(testing_images{i,1}));
    figure()
    imshow(I_in,[])
    [I_testing_l,I_testing_m,I_testing_r,min_index_l,min_index_m,min_index_r] =  test_all_bottles(I_in,rescale_factor,I_training_avg,eig_vect_extract,proj_eigvect);
    
    figure()
    subplot(2,3,1)
    imshow(I_testing_l)
    subplot(2,3,4)
    imshow(squeeze(I_training(min_index_l,:,:)))
    subplot(2,3,2)
    imshow(I_testing_m)
    subplot(2,3,5)
    imshow(squeeze(I_training(min_index_m,:,:)))
    subplot(2,3,3)
    imshow(I_testing_r)
    subplot(2,3,6)
    imshow(squeeze(I_training(min_index_r,:,:)))
   
    if(min_index_l>63)
        score(i,1) = 8;
    else
        score(i,1) = ceil(min_index_l/9);
    end
    if(min_index_m>63)
        score(i,2) = 8;
    else
        score(i,2) = ceil(min_index_m/9);
    end
    if(min_index_r>63)
        score(i,3) = 8;
    else
        score(i,3) = ceil(min_index_r/9);
    end
end

total_score = score == score_ref;
accuracy = (sum(sum(total_score))/(size(total_score,1)*size(total_score,2)))*100
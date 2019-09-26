function [I_testing_l,I_testing_m,I_testing_r,min_index_l,min_index_m,min_index_r] =  test_all_bottles(I_in,rescale_factor,I_training_avg,eig_vect_extract,proj_eigvect)

[l_bottle,m_bottle,r_bottle] = extract_all_bottles(I_in);
I_testing_l = im2double(imresize(l_bottle,rescale_factor));
I_testing_m = im2double(imresize(m_bottle,rescale_factor));
I_testing_r = im2double(imresize(r_bottle,rescale_factor));
I_testing_l_vect = reshape(I_testing_l,1,size(I_testing_l,1)*size(I_testing_l,2))-I_training_avg; 
I_testing_m_vect = reshape(I_testing_m,1,size(I_testing_m,1)*size(I_testing_m,2))-I_training_avg; 
I_testing_r_vect = reshape(I_testing_r,1,size(I_testing_r,1)*size(I_testing_r,2))-I_training_avg; 

%%%%% calculating & comparing the euclidian distance of all projected trained images from the projected test image %%%%%
I_testing_l_proj =  I_testing_l_vect * eig_vect_extract;
I_testing_m_proj =  I_testing_m_vect * eig_vect_extract;
I_testing_r_proj =  I_testing_r_vect * eig_vect_extract;

euclide_dist_l = [ ];
euclide_dist_m = [ ];
euclide_dist_r = [ ];

for j = 1 : size(proj_eigvect,1)
temp_l = (norm(I_testing_l_proj-proj_eigvect(j,:,:)));  
euclide_dist_l = [euclide_dist_l temp_l];
temp_m = (norm(I_testing_m_proj-proj_eigvect(j,:,:)));  
euclide_dist_m = [euclide_dist_m temp_m];
temp_r = (norm(I_testing_r_proj-proj_eigvect(j,:,:)));  
euclide_dist_r = [euclide_dist_r temp_r];
end
[min_val_l,min_index_l] = min(euclide_dist_l);
[min_val_m,min_index_m] = min(euclide_dist_m);
[min_val_r,min_index_r] = min(euclide_dist_r);

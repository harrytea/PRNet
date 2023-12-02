% compute RMSE(MAE)
clear;close all;clc
% 1`modify the following directories 2`run����
tic;

% GT mask result��
gtdir = 'D:\Removal\data4\others\video\free\'; GTlist = dir([gtdir '/*.png']);
maskdir = 'D:\Removal\data4\others\video\mask\'; masklist = dir([maskdir '/*.png']);
resultsdir = 'D:\Removal\data4\others\video\55aistd\'; resultslist = dir([resultsdir '/*.png']);

total_dist_all    = 0;
total_dist_non    = 0;
total_dist_shadow = 0;
total_num_all     = 0;
total_num_non     = 0;
total_num_shadow  = 0;

mae_all     = zeros(1,size(resultslist,1)); 
mae_non     = zeros(1,size(resultslist,1)); 
mae_shadow  = zeros(1,size(resultslist,1)); 

psnr_all    = zeros(1,size(resultslist,1));
psnr_non    = zeros(1,size(resultslist,1));
psnr_shadow = zeros(1,size(resultslist,1));

ssim_all    = zeros(1,size(resultslist,1));
ssim_non    = zeros(1,size(resultslist,1));
ssim_shadow = zeros(1,size(resultslist,1));

cform  = makecform('srgb2lab');
for i=1:size(resultslist)
    % read file
    gt = imread(strcat(gtdir, GTlist(i).name)); 
    mask = imread(strcat(maskdir, masklist(i).name)); 
    result = imread(strcat(resultsdir, resultslist(i).name));
    gt = double(gt)/255;
    result = double(result)/255;
    % resize
    
%     mask = double(mask);                %%%%%%
    
    
    
    result = imresize(result, [256 256]);
    gt = imresize(gt, [256 256]);
    mask = imresize(mask, [256 256]);

    mask_non = ~mask;       
    mask_shadow = ~mask_non;
    
    
%     mask_non = double(mask_non);         %%%%%%
%     mask_shadow = double(mask_shadow);   %%%%%%
    
    % psnr && ssim
    psnr_all(i)    = psnr(result, gt);
    psnr_shadow(i) = psnr(result.*repmat(mask_shadow,[1 1 1]), gt.*repmat(mask_shadow,[1 1 1]));
    psnr_non(i)    = psnr(result.*repmat(mask_non,[1 1 1]), gt.*repmat(mask_non,[1 1 1]));
    ssim_all(i)    = ssim(result, gt);
    ssim_shadow(i) = ssim(result.*repmat(mask_shadow,[1 1 1]), gt.*repmat(mask_shadow,[1 1 1]));
    ssim_non(i)    = ssim(result.*repmat(mask_non,[1 1 1]), gt.*repmat(mask_non,[1 1 1]));

    % mae
    % rgb to lab
    gt     = applycform(gt,cform);    
    result = applycform(result,cform);
    
    
    % per image
    dist            = abs((gt - result));  % all
    dist_sum_all    = sum(dist(:));
    dist_shadow     = dist.*repmat(mask_shadow, [1 1 1]); % shadow
    dist_sum_shadow = sum(dist_shadow(:));
    dist_non        = dist.*repmat(mask_non, [1 1 1]); % non-shadow
    dist_sum_non    = sum(dist_non(:));
    
    mask_num_all    = size(gt,1)*size(gt,2);
    mask_num_shadow = sum(mask_shadow(:));
    mask_num_non    = sum(mask_non(:));
    % Process
    % Per Image Process and Average
    mae_all(i)    = dist_sum_all/mask_num_all;
    mae_shadow(i) = dist_sum_shadow/mask_num_shadow;
    mae_non(i)    = dist_sum_non/mask_num_non;
    % Accumulate all Pixels and Average
    total_dist_all = total_dist_all + dist_sum_all;
    total_num_all  = total_num_all + mask_num_all;
    total_dist_shadow = total_dist_shadow + dist_sum_shadow;
    total_num_shadow = total_num_shadow + mask_num_shadow;
    total_dist_non = total_dist_non + dist_sum_non;
    total_num_non = total_num_non + mask_num_non;  

    fprintf('%s -- PSNR: %.4f, SSIM: %.4f, MAE: %.4f\n', GTlist(i).name, psnr_all(i), ssim_all(i), mae_all(i));
end
% PSNR Average
psnr_all_ave = mean(psnr_all);
psnr_non_ave = mean(psnr_non);
psnr_sha_ave = mean(psnr_shadow);
% SSIM Average
ssim_all_ave = mean(ssim_all);
ssim_non_ave = mean(ssim_non);
ssim_sha_ave = mean(ssim_shadow);
% Per Image Average Results
imae_all_ave = mean(mae_all); % image
imae_non_ave = mean(mae_non); % image
imae_sha_ave = mean(mae_shadow); % image
% All Pixels Average Results
pmae_all_ave = mean(total_dist_all/total_num_all); % pixel
pmae_non_ave = mean(total_dist_non/total_num_non); % pixel
pmae_sha_ave = mean(total_dist_shadow/total_num_shadow); % pixel

toc;
fprintf('       all,     non,     shadow \n');
fprintf('PSNR:  %.4f  %.4f  %.4f\n',   psnr_all_ave, psnr_non_ave, psnr_sha_ave);
fprintf('SSIM:  %.4f   %.4f   %.4f\n', ssim_all_ave, ssim_non_ave, ssim_sha_ave);
fprintf('Image: %.4f   %.4f   %.4f\n', imae_all_ave, imae_non_ave, imae_sha_ave); % use this
% fprintf('Pixel: %.4f   %.4f   %.4f\n', pmae_all_ave, pmae_non_ave, pmae_sha_ave);

fprintf('             PSNR,  SSIM,  RMSE \n');
fprintf('Shadow:      %.3f   %.3f   %.2f\n',   psnr_sha_ave, ssim_sha_ave, imae_sha_ave);
fprintf('Non-shadow:  %.3f   %.3f   %.2f\n', psnr_non_ave, ssim_non_ave, imae_non_ave);
fprintf('ALL-image:   %.3f   %.3f   %.2f\n', psnr_all_ave, ssim_all_ave, imae_all_ave);




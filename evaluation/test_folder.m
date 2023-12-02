% 测试多个epoch

% gtdir = 'D:\ShadowData\ISTD\test\test_C\'; GTlist = dir([gtdir '/*.png']);
% gtdir = 'D:\ShadowData\ISTD\test\test_C_fixed\'; GTlist = dir([gtdir '/*.png']);
% maskdir = 'D:\ShadowData\ISTD\test\test_B\'; masklist = dir([maskdir '/*.png']);
% resultsdir = 'D:\Removal\data4\exp2_recur\istd\'; resultslist = dir([resultsdir '/*.png']);
gtdir = 'D:\ShadowData\SRD\test\test_C\'; GTlist = dir([gtdir '/*.jpg']);
maskdir = 'D:\ShadowData\SRD\test\test_B\'; masklist = dir([maskdir '/*.jpg']);
resultsdir = 'D:\Removal\data4\exp2_recur\64_recur\srd_ori\'; resultslist = dir([resultsdir '/*.jpg']);



for iter=10:10:300
    pred_path_num = sprintf("%s%s%d%s", resultsdir, '\', iter, 'img\');  % corresponding scan image path
    pred_path_num = char(pred_path_num);
    resultslist = dir([pred_path_num '/*.jpg']);  %%%%% jpg or png
    [imae_all, imae_non, imae_sha, pmae_all, pmae_non, pmae_sha] = compute_mae(gtdir, GTlist, maskdir, masklist, pred_path_num, resultslist);
    [psnr_all_ave, psnr_non_ave, psnr_sha_ave, ssim_all_ave, ssim_non_ave, ssim_sha_ave] = compute_psnr(gtdir, GTlist, maskdir, masklist, pred_path_num, resultslist);
    fprintf('%s-- Image - all:%.4f  non:%.4f  shadow:%.4f\n', pred_path_num, imae_all, imae_non, imae_sha);
    % fprintf('%s-- Pixel - all:%.4f  non:%.4f  shadow:%.4f\n', pred_path_num, pmae_all, pmae_non, pmae_sha); % no
    fprintf('%s-- PSNR  - all:%.4f  non:%.4f  shadow:%.4f\n', pred_path_num, psnr_all_ave, psnr_non_ave, psnr_sha_ave);
    fprintf('%s-- SSIM  - all:%.4f  non:%.4f  shadow:%.4f\n', pred_path_num, ssim_all_ave, ssim_non_ave, ssim_sha_ave);
end
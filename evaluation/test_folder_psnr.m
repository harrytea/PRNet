% gtdir = 'D:\ShadowData\ISTD\test\test_C\'; GTlist = dir([gtdir '/*.png']);
gtdir = 'D:\ShadowData\ISTD\test\test_C_fixed\'; GTlist = dir([gtdir '/*.png']);
maskdir = 'D:\ShadowData\ISTD\test\test_B\'; masklist = dir([maskdir '/*.png']);
resultsdir = 'D:\Removal\data4\shadow_test\10\istd\'; resultslist = dir([resultsdir '/*.png']);

for iter=40:40:960
    pred_path_num = sprintf("%s%s%d%s", resultsdir, '\', iter, '\');  % corresponding scan image path
    pred_path_num = char(pred_path_num);
    resultslist = dir([pred_path_num '/*.png']);
    [psnr_all_ave, psnr_non_ave, psnr_sha_ave, ssim_all_ave, ssim_non_ave, ssim_sha_ave] = compute_psnr(gtdir, GTlist, maskdir, masklist, pred_path_num, resultslist);
    fprintf('%s-- PSNR - all:%.4f  non:%.4f  shadow:%.4f\n', pred_path_num, psnr_all_ave, psnr_non_ave, psnr_sha_ave);
    fprintf('%s-- SSIM - all:%.4f  non:%.4f  shadow:%.4f\n', pred_path_num, ssim_all_ave, ssim_non_ave, ssim_sha_ave);
end
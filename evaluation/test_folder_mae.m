% gtdir = 'D:\ShadowData\ISTD\test\test_C\'; GTlist = dir([gtdir '/*.png']);
gtdir = 'D:\ShadowData\ISTD\test\test_C_fixed\'; GTlist = dir([gtdir '/*.png']);
maskdir = 'D:\ShadowData\ISTD\test\test_B\'; masklist = dir([maskdir '/*.png']);
resultsdir = 'D:\Removal\data4\shadow_test\10\istd\'; resultslist = dir([resultsdir '/*.png']);

for iter=40:40:960
    pred_path_num = sprintf("%s%s%d%s", resultsdir, '\', iter, '\');  % corresponding scan image path
    pred_path_num = char(pred_path_num);
    resultslist = dir([pred_path_num '/*.png']);
    [imae_all, imae_non, imae_sha, pmae_all, pmae_non, pmae_sha] = compute_mae(gtdir, GTlist, maskdir, masklist, pred_path_num, resultslist);
    fprintf('%s-- Image - all:%.4f  non:%.4f  shadow:%.4f\n', pred_path_num, imae_all, imae_non, imae_sha);
    fprintf('%s-- Pixel - all:%.4f  non:%.4f  shadow:%.4f\n', pred_path_num, pmae_all, pmae_non, pmae_sha);
end
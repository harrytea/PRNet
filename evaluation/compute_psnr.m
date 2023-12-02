function [psnr_all_ave, psnr_non_ave, psnr_sha_ave, ssim_all_ave, ssim_non_ave, ssim_sha_ave] ...
    = compute_psnr(gtdir, GTlist, maskdir, masklist, resultsdir, resultslist)
    psnr_all    = zeros(1,size(resultslist,1));
    psnr_non    = zeros(1,size(resultslist,1));
    psnr_shadow = zeros(1,size(resultslist,1));

    ssim_all    = zeros(1,size(resultslist,1));
    ssim_non    = zeros(1,size(resultslist,1));
    ssim_shadow = zeros(1,size(resultslist,1));

    for i=1:size(resultslist)
        % read file
        gt = imread(strcat(gtdir, GTlist(i).name)); 
        mask = imread(strcat(maskdir, masklist(i).name)); 
        result = imread(strcat(resultsdir, resultslist(i).name));
        gt = double(gt)/255;
        result = double(result)/255;
        % resize
        result = imresize(result, [256 256]);
        gt = imresize(gt, [256 256]);
        mask = imresize(mask, [256 256]);

        mask_non = ~mask;       
        mask_shadow = ~mask_non;

        % psnr && ssim
        psnr_all(i)    = psnr(result, gt);
        psnr_shadow(i) = psnr(result.*repmat(mask_shadow,[1 1 3]), gt.*repmat(mask_shadow,[1 1 3]));
        psnr_non(i)    = psnr(result.*repmat(mask_non,[1 1 3]), gt.*repmat(mask_non,[1 1 3]));
        ssim_all(i)    = ssim(result, gt);
        ssim_shadow(i) = ssim(result.*repmat(mask_shadow,[1 1 3]), gt.*repmat(mask_shadow,[1 1 3]));
        ssim_non(i)    = ssim(result.*repmat(mask_non,[1 1 3]), gt.*repmat(mask_non,[1 1 3]));
    end
    % PSNR Average
    psnr_all_ave = mean(psnr_all);
    psnr_non_ave = mean(psnr_non);
    psnr_sha_ave = mean(psnr_shadow);
    % SSIM Average
    ssim_all_ave = mean(ssim_all);
    ssim_non_ave = mean(ssim_non);
    ssim_sha_ave = mean(ssim_shadow);
end

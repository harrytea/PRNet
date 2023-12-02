function [imae_all_ave, imae_non_ave, imae_sha_ave, pmae_all_ave, pmae_non_ave, pmae_sha_ave] ...
    = compute_mae(gtdir, GTlist, maskdir, masklist, resultsdir, resultslist)
    total_dist_all    = 0;
    total_dist_non    = 0;
    total_dist_shadow = 0;
    total_num_all     = 0;
    total_num_non     = 0;
    total_num_shadow  = 0;

    mae_all     = zeros(1,size(resultslist,1)); 
    mae_non     = zeros(1,size(resultslist,1)); 
    mae_shadow  = zeros(1,size(resultslist,1)); 

    cform  = makecform('srgb2lab');
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

        % mae
        % rgb to lab
        gt     = applycform(gt,cform);    
        result = applycform(result,cform);


        % per image
        dist            = abs((gt - result));  % all
        dist_sum_all    = sum(dist(:));
        dist_shadow     = dist.*repmat(mask_shadow, [1 1 3]); % shadow
        dist_sum_shadow = sum(dist_shadow(:));
        dist_non        = dist.*repmat(mask_non, [1 1 3]); % non-shadow
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
    end
    % Per Image Average Results
    imae_all_ave = mean(mae_all); % image
    imae_non_ave = mean(mae_non); % image
    imae_sha_ave = mean(mae_shadow); % image
    % All Pixels Average Results
    pmae_all_ave = mean(total_dist_all/total_num_all); % pixel
    pmae_non_ave = mean(total_dist_non/total_num_non); % pixel
    pmae_sha_ave = mean(total_dist_shadow/total_num_shadow); % pixel
end

#!/bin/bash

root='results/'
out_root='results/metrics'

celeba_array=('self_celeba_v2')

other_array=('lfw_crop' 'wider')

for test_name in "${celeba_array[@]}"
do
    test_image=$test_name'/restored_faces'
    # test_image=$test_name
    out_name=$test_name
    # 0: the name of image does not include 00 and Codeformer
    # 1: otherwise
    need_post=1
    # need_post=0

    CelebAHQ_GT='./data/celeba_512_validation'

    # FID
    python -u scripts/metrics/cal_fid.py \
    $root'/'$test_image \
    --fid_stats 'experiments/pretrained_models/inception_FFHQ_512-f7b384ab.pth' \
    --save_name $out_root'/'$out_name'_fid.txt' \

    if [ -d $CelebAHQ_GT ]
    then
        # PSRN SSIM LPIPS
        python -u scripts/metrics/cal_psnr_ssim.py \
        $root'/'$test_image \
        --gt_folder $CelebAHQ_GT \
        --save_name $out_root'/'$out_name'_psnr_ssim_lpips.txt' \
        --need_post $need_post \

        # # # PSRN SSIM LPIPS
        python -u scripts/metrics/cal_identity_distance.py  \
        $root'/'$test_image \
        --gt_folder $CelebAHQ_GT \
        --save_name $out_root'/'$out_name'_id.txt' \
        --need_post $need_post
    else
        echo 'The path of GT does not exist'
    fi
done

for test_name in "${other_array[@]}"
do
    test_image=$test_name'/restored_faces'
    # test_image=$test_name
    out_name=$test_name

    # FID
    python -u scripts/metrics/cal_fid.py \
    $root'/'$test_image \
    --fid_stats 'experiments/pretrained_models/inception_FFHQ_512-f7b384ab.pth' \
    --save_name $out_root'/'$out_name'_fid.txt' \

done
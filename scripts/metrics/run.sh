#!/bin/bash

### Journal ###
# root='results/'
# out_root='results/metrics'
root='/ssd1/yuju/CodeFormer/results'
out_root='/ssd1/yuju/CodeFormer/results/metrics'
# root='data/'
# out_root='data/metrics'

#test_name='BRIAR_10km'
#test_name='BRIAR_20km'
#test_name='BRIAR_30km'
#test_name='BRIAR_40km'
#test_name='BRIAR_celeba'

#test_name='pretrain_10km'
#test_name='pretrain_20km'
#test_name='pretrain_30km'
#test_name='pretrain_40km'
# test_name='pretrain_celeba'

# test_name='celeba_HQHQ_BCE_test'
# test_name='celeba_HQHQ_L1_test'
# test_name='celeba_LQHQ_BCE_test'
# test_name='celeba_LQHQ_L1_test'

# test_name='child_HQHQ_BCE'
# test_name='child_HQHQ_L1'
# test_name='child_LQHQ_BCE'
# test_name='child_LQHQ_L1'

# test_name='lfw_HQHQ_BCE'
# test_name='lfw_HQHQ_L1'
# test_name='lfw_LQHQ_BCE'
# test_name='lfw_LQHQ_L1'

# test_name='web_HQHQ_BCE'
# test_name='web_HQHQ_L1'
# test_name='web_LQHQ_BCE'
# test_name='web_LQHQ_L1'

# test_name='child_pretrain'
# test_name='lfw_pretrain'
# test_name='web_pretrain'

# test_name='LQHQ_celeba'
# test_name='LQLQ_celeba'
# test_name='LQ_code_lr_1_celeba'

# celeba_array=('LQLQ_celeba' 'LQ_code_lr_1_celeba')
# celeba_array=('celeba_512_validation_lq_0.0')
# celeba_array=('celeba_512_validation_lq_0.5')
celeba_array=('celeba_512_validation_lq_1.0')

# test_name='LQHQ_child'
# test_name='LQLQ_child'
# test_name='LQ_code_lr_1_child'

# test_name='LQHQ_lfw'
# test_name='LQLQ_lfw'
# test_name='LQ_code_lr_1_lfw'

# test_name='LQHQ_web'
# test_name='LQLQ_web'
# test_name='LQ_code_lr_1_web'

# other_array=('LQHQ_child' 'LQLQ_child' 'LQ_code_lr_1_child' 'LQHQ_lfw' 'LQLQ_lfw' 'LQ_code_lr_1_lfw' 'LQHQ_web' 'LQLQ_web' 'LQ_code_lr_1_web')
# other_array=('Child' 'lfw' 'WebPhoto-Test')

for test_name in "${celeba_array[@]}"
do
    test_image=$test_name'/restored_faces'
    # test_image=$test_name
    out_name=$test_name
    # need_post=1
    need_post=0

    #CelebAHQ_GT='/ssd2/yuju/RestoreFormer/data/FFHQ-BRIAR/Yu-Ju/images512x512'
    # CelebAHQ_GT='/ssd2/yuju/RestoreFormer/data/celeba_512_validation'
    CelebAHQ_GT='/ssd1/yuju/CodeFormer/celeba_512_validation'

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

# for test_name in "${other_array[@]}"
# do
#     test_image=$test_name'/restored_faces'
#     # test_image=$test_name
#     out_name=$test_name
#     need_post=1

#     # FID
#     python -u scripts/metrics/cal_fid.py \
#     $root'/'$test_image \
#     --fid_stats 'experiments/pretrained_models/inception_FFHQ_512-f7b384ab.pth' \
#     --save_name $out_root'/'$out_name'_fid.txt' \

# done
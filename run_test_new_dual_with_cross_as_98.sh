#!/bin/bash

# root='results'
# out_root='results/metrics'

root='results_dual_new_ep_98'
out_root='results_dual_new_ep_98/metrics'

# root='/ssd2/yuju/PSFRGAN/results/'
# out_root='/ssd2/yuju/PSFRGAN/results/metrics'

if [ ! -d $out_root ];then
    mkdir -p $out_root
fi

# dataset_name_array=('self_celeba_v2' 'child' 'lfw' 'lfw_crop' 'web' 'wider')
dataset_name_array=('self_celeba_v2' 'lfw' 'lfw_crop' 'wider' 'BRIAR')
# dataset_name_array=('self_celeba')
# dataset_location_array=('self_celeba_512' 'Child' 'lfw' 'lfw_cropped_faces' 'WebPhoto-Test' 'Wider-Test')
dataset_location_array=('self_celeba_512_v2' 'lfw' 'lfw_cropped_faces' 'Wider-Test' 'mix_briar_128')
# dataset_location_array=('self_celeba_512')

# checkpoint=${1}
# config=${2}
# output_name=${3}
# GPU=${4}

checkpoint='/ssd1/yuju/RestoreFormer/experiments/logs/2023-11-29T12-34-27_Merge_feature_512_with_as_and_cross_dual_new_range/checkpoints/epoch=000098-Rec_loss=0.1075163334608078-BCE_loss=3.9325973987579346-L2_loss=0.10661053657531738.ckpt'
config='/ssd1/yuju/RestoreFormer/configs/Merge_feature_512_with_as_and_cross_dual_new_range.yaml'
output_name='DAEFR_dual_new_range_ep_98'
GPU='0'

# echo ${0}
echo ${checkpoint}
echo ${config}
echo ${output_name}
echo $GPU

for i in $(seq 0 4)
do 
    
# echo ${dataset_name_array[${i}]}
# echo ${dataset_location_array[${i}]}

outdir=$root'/'$output_name'_'${dataset_name_array[${i}]}
align_test_path='./data/'${dataset_location_array[${i}]}


# echo ${outdir}
# echo ${align_test_path}

if [ ! -d $outdir ];then
    mkdir $outdir
fi

CUDA_VISIBLE_DEVICES=$GPU python -u scripts/test.py \
--outdir $outdir \
-r $checkpoint \
-c $config \
--test_path $align_test_path \
--aligned

done


# Calculate the FID for real-world datasets
for i in $(seq 1 4)
do
outdir=$output_name'_'${dataset_name_array[${i}]}
# echo $outdir
test_image=$outdir'/restored_faces'
# test_image=$outdir'/hq'
out_name=$outdir
# echo $out_name

# FID
CUDA_VISIBLE_DEVICES=$GPU python -u scripts/metrics/cal_fid.py \
$root'/'$test_image \
--fid_stats 'experiments/pretrained_models/inception_FFHQ_512-f7b384ab.pth' \
--save_name $out_root'/'$out_name'_fid.txt' \

done

# Calculate the FID for Celeba-Test datasets

# For method results
outdir=$output_name'_'${dataset_name_array[0]}
test_image=$outdir'/restored_faces'
# test_image=$outdir'/hq'
# 0: the name of image does not include 00 and Codeformer
# 1: otherwise
need_post=1
# need_post=0

# For original dataset
# test_image=$test_name
# 0: the name of image does not include 00
# 1: otherwise
# need_post=0

out_name=$outdir
# echo $outdir $out_name

#CelebAHQ_GT='/ssd2/yuju/RestoreFormer/data/FFHQ-BRIAR/Yu-Ju/images512x512'
CelebAHQ_GT='/ssd1/yuju/RestoreFormer/data/celeba_512_validation'

# FID
CUDA_VISIBLE_DEVICES=$GPU python -u scripts/metrics/cal_fid.py \
$root'/'$test_image \
--fid_stats 'experiments/pretrained_models/inception_FFHQ_512-f7b384ab.pth' \
--save_name $out_root'/'$out_name'_fid.txt' \

if [ -d $CelebAHQ_GT ]
then
# PSRN SSIM LPIPS
CUDA_VISIBLE_DEVICES=$GPU python -u scripts/metrics/cal_psnr_ssim.py \
$root'/'$test_image \
--gt_folder $CelebAHQ_GT \
--save_name $out_root'/'$out_name'_psnr_ssim_lpips.txt' \
--need_post $need_post \

# # # PSRN SSIM LPIPS
CUDA_VISIBLE_DEVICES=$GPU python -u scripts/metrics/cal_identity_distance.py \
$root'/'$test_image \
--gt_folder $CelebAHQ_GT \
--save_name $out_root'/'$out_name'_id.txt' \
--need_post $need_post
else
    echo 'The path of GT does not exist'
fi
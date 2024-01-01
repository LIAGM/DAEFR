#!/bin/bash

root='results'
out_root='results/metrics'

if [ ! -d $root ];then
    mkdir -p $root
fi

if [ ! -d $out_root ];then
    mkdir -p $out_root
fi


dataset_name_array=('self_celeba_v2' 'lfw' 'lfw_crop' 'wider' 'BRIAR')

dataset_location_array=('self_celeba_512_v2' 'lfw' 'lfw_cropped_faces' 'Wider-Test' 'mix_briar_128')

checkpoint='/ssd1/yuju/DAEFR/experiments/DAEFR_model/DAEFR_model.ckpt'
config='/ssd1/yuju/DAEFR/configs/DAEFR.yaml'
output_name='DAEFR'
GPU='4'

# echo ${0}
echo ${checkpoint}
echo ${config}
echo ${output_name}
echo $GPU

for i in $(seq 0 4)
do 

outdir=$root'/'$output_name'_'${dataset_name_array[${i}]}
align_test_path='/ssd2/yuju/RestoreFormer/data/'${dataset_location_array[${i}]}


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

CelebAHQ_GT='/ssd2/yuju/RestoreFormer/data/celeba_512_validation'

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
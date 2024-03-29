exp_name='DAEFR'

root_path='experiments'
out_root_path='results'
align_test_path='datasets/test'
tag='test'

outdir=$out_root_path'/'$exp_name'_'$tag

if [ ! -d $outdir ];then
    mkdir $outdir
fi

python -u scripts/test.py \
--outdir $outdir \
-r './experiments/DAEFR_model.ckpt' \
-c 'configs/DAEFR.yaml' \
--test_path $align_test_path \
--aligned


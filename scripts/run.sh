#export BASICSR_JIT=True
export CXX=g++

# conf_name='LQ_Dictionary'
# conf_name='HQ_Dictionary_128'
# conf_name='LQ_Dictionary_128'
# conf_name='RestoreFormer'
# conf_name='RestoreFormer-origin'
# conf_name='RestoreFormer_code'
# conf_name='Dual_codebook'
# conf_name='Dual_codebook_with_module_L2'
# conf_name='Dual_codebook_128'
# conf_name='Dual_codebook_for_LQ_128'
# conf_name='Dual_codebook_L1'
# conf_name='Merge_feature_128_v2_lr028'
# conf_name='Merge_feature_128_test'
# conf_name='Merge_feature_512_with_as_and_cross_finetune'
# conf_name='LQHQ_index_v2_128_HQ'
# conf_name='LQLQ_index_v2_128_HQ'
# conf_name='codeformer_512'
# conf_name='codeformer_origin_weight'
# conf_name='codeformer_origin_weight_and_range'
#conf_name='Merge_feature_512_with_cross_and_proj_v2'
conf_name='Merge_feature_512_with_as_and_cross_with_at'

ROOT_PATH='/ssd1/yuju/RestoreFormer/experiments/' # The path for saving model and logs

#gpus='0,'
gpus='0,1,2,3,4,5,6,7'

#P: pretrain SL: soft learning
node_n=1

#python -u main_for_module_association.py \
python -u main_codeformer.py \
--root-path $ROOT_PATH \
--base 'configs/'$conf_name'.yaml' \
-t True \
--gpus $gpus \
--num-nodes $node_n \
# --resume /ssd1/yuju/RestoreFormer/experiments/logs/2023-04-20T09-06-29_codeformer_origin_weight/checkpoints/last.ckpt
# --random-seed True \
# --postfix $conf_name \

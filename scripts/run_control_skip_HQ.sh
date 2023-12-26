# export BASICSR_JIT=True
export CXX=g++

# conf_name='LQ_Dictionary'
# conf_name='HQ_Dictionary_128'
conf_name='Merge_feature_512_with_as_and_cross_control_skip_HQ'
# conf_name='RestoreFormer'
# conf_name='RestoreFormer-origin'
# conf_name='RestoreFormer_code'
# conf_name='Dual_codebook'
# conf_name='Dual_codebook_L1'


ROOT_PATH='/ssd1/yuju/RestoreFormer/experiments/' # The path for saving model and logs

gpus='0,1,2,3'
# gpus='0,'

#P: pretrain SL: soft learning
node_n=1

python -u main_control.py \
--root-path $ROOT_PATH \
--base 'configs/'$conf_name'.yaml' \
-t True \
--gpus $gpus \
--num-nodes $node_n \
--resume /ssd1/yuju/RestoreFormer/experiments/logs/2023-09-24T19-34-29_Merge_feature_512_with_as_and_cross_control_skip_HQ/checkpoints/last.ckpt
# --random-seed True \
# --postfix $conf_name \

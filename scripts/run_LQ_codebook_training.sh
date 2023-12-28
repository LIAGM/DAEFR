# export BASICSR_JIT=True
export CXX=g++

conf_name='LQ_codebook'

ROOT_PATH='/ssd1/yuju/DAEFR/experiments/' # The path for saving model and logs

#gpus='0,1,2,3'
gpus='0,'

#P: pretrain SL: soft learning
node_n=1

python -u main_for_LQ_codebook.py \
--root-path $ROOT_PATH \
--base 'configs/'$conf_name'.yaml' \
-t True \
--gpus $gpus \
--num-nodes $node_n \

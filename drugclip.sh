

data_path="/mnt/goon/benchmark_code/drugclip_data/"


save_dir="savedir"

tmp_save_dir="tmp_save_dir"
tsb_dir="tsb_dir"

n_gpu=1
MASTER_PORT=10055
finetune_mol_model="mol_pre_no_h_220816.pt" # unimol pretrained mol model
finetune_pocket_model="pocket_pre_220816.pt" # unimol pretrained pocket model


batch_size=48
batch_size_valid=48
#batch_size_valid=64
#batch_size_valid=128
epoch=200
dropout=0.0
warmup=0.06
update_freq=1
dist_threshold=8.0
recycling=3
lr=1e-3

# NOTE: CUDA_VISIBLE_DEVICES must be set to all GPUs that we will use - start w/ 0
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=$n_gpu \
       --master_port=$MASTER_PORT $(which unicore-train) $data_path --user-dir ./unimol \
       --train-subset train --valid-subset valid \
       --num-workers 8 --ddp-backend=c10d \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 256 \
       --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-8 --clip-norm 1.0 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $batch_size --batch-size-valid $batch_size_valid \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --update-freq $update_freq --seed 1 \
       --find-unused-parameters \
       --finetune-pocket-model $finetune_pocket_model \
       --finetune-mol-model $finetune_mol_model \
       --tensorboard-logdir $tsb_dir \
       --log-interval 100 --log-format simple \
       --patience 2000 \
       --all-gather-list-size 2048000 \
       --best-checkpoint-metric valid_bedroc \
       --save-dir $save_dir \
       --tmp-save-dir $tmp_save_dir \
       --save-interval-updates 5000 \
       --keep-best-checkpoints 10 \
       --keep-last-epochs 5 \
       --maximize-best-checkpoint-metric \


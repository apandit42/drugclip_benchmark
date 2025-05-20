results_path="./test2"  # replace to your results path
batch_size=32
weight_path="/mnt/goon/benchmark_code/ap_drugclip_benchmark/savedir/checkpoint.best_valid_bedroc_0.37.pt"
MOL_PATH="/mnt/goon/benchmark_code/drugclip_data/full_test_lig_dataset_v2.lmdb" # path to the molecule file
POCKET_PATH="/mnt/goon/benchmark_code/drugclip_data/full_val_target_dataset.lmdb" # path to the pocket file
EMB_DIR="./sims/test_emb_bedroc0.37_V2" # path to the cached mol embedding fille

mkdir $EMB_DIR

CUDA_VISIBLE_DEVICES="0" python ./unimol/retrieval.py --user-dir ./unimol $data_path "./data" --valid-subset test \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 256 \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --path $weight_path \
       --log-interval 100 --log-format simple \
       --mol-path $MOL_PATH \
       --pocket-path $POCKET_PATH \
       --emb-dir $EMB_DIR \

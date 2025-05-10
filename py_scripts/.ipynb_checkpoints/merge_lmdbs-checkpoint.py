import lmdb
import os
import pickle
from glob import glob

def concat_lmdbs_preserve_keys(source_paths, dest_path):
    # os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    env_out = lmdb.open(dest_path, subdir=False,map_size=1 << 40)  # 1 TB

    total_entries = 0
    with env_out.begin(write=True) as txn_out:
        for src_path in source_paths:
            env_in = lmdb.open(src_path, subdir=False,readonly=True, lock=False, readahead=False)
            with env_in.begin() as txn_in:
                cursor = txn_in.cursor()
                for key, val in cursor:
                    if txn_out.get(key) is not None:
                        continue
                        # raise ValueError(f"Duplicate key detected: {key.decode()}")
                    txn_out.put(key, val)
                    total_entries += 1
            env_in.close()

    env_out.close()
    print(f"Merged {len(source_paths)} LMDBs into {dest_path} with {total_entries} entries (original keys preserved).")


paths = sorted(glob("/mnt/goon/benchmark_code/drugclip_data/train/*train_chunk_*.lmdb"))
concat_lmdbs_preserve_keys(paths, "/mnt/goon/benchmark_code/drugclip_data/train/full_lig_dataset.lmdb")
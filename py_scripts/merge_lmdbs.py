import lmdb
import os
import pickle
from glob import glob
from tqdm import tqdm

def concat_lmdbs_preserve_keys(source_paths, dest_path):
    # os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    env_out = lmdb.open(dest_path, subdir=False,map_size=1 << 40)  # 1 TB

    total_entries = 0
    with env_out.begin(write=True) as txn_out:
        for src_path in tqdm(source_paths):
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

# #train
# prot_path = "/mnt/goon/benchmark_code/drugclip_data/full_train_target_dataset.lmdb"
# lig_path = "/mnt/goon/benchmark_code/drugclip_data/full_train_lig_dataset.lmdb"
# out = '/mnt/goon/benchmark_code/drugclip_data/train.lmdb'
# concat_lmdbs_preserve_keys([lig_path, prot_path], out)
# #val
# prot_path = "/mnt/goon/benchmark_code/drugclip_data/full_val_target_dataset.lmdb"
# lig_path= "/mnt/goon/benchmark_code/drugclip_data/full_val_lig_dataset.lmdb"
# out = '/mnt/goon/benchmark_code/drugclip_data/val.lmdb'
# concat_lmdbs_preserve_keys([lig_path, prot_path], out)

# #test
# prot_path = "/mnt/goon/benchmark_code/drugclip_data/full_val_target_dataset.lmdb"
# lig_path= "/mnt/goon/benchmark_code/drugclip_data/full_test_lig_dataset.lmdb"
# out = '/mnt/goon/benchmark_code/drugclip_data/test.lmdb'
# concat_lmdbs_preserve_keys([lig_path, prot_path], out)




# #train ligs:
# train_full_lig_path = "/mnt/goon/benchmark_code/drugclip_data/full_train_lig_dataset.lmdb"
# lmdb_regex ="/mnt/goon/benchmark_code/drugclip_data/train/*train_chunk_*.lmdb"
# print(f"Merging LMDBS for train ligands: {lmdb_regex}\n into: {train_full_lig_path}")
# paths = sorted(glob(lmdb_regex))
# concat_lmdbs_preserve_keys(paths, train_full_lig_path)
# print('done with that!')
# #train targets:
# train_full_target_path = "/mnt/goon/benchmark_code/drugclip_data/full_train_target_dataset.lmdb"
# lmdb_regex ="/mnt/goon/benchmark_code/drugclip_data/protein_data/train/*train_chunk_*.lmdb"
# print(f"Merging LMDBS for train ligands: {lmdb_regex}\n into: {train_full_target_path}")
# paths = sorted(glob(lmdb_regex))
# concat_lmdbs_preserve_keys(paths, train_full_target_path)
# print('done with that!')


# #val ligands:
# train_full_lig_path = "/mnt/goon/benchmark_code/drugclip_data/full_val_lig_dataset_v2.lmdb"
# lmdb_regex ="/mnt/goon/benchmark_code/drugclip_data/ligand_data/val/*train_chunk_*.lmdb"
# print(f"Merging LMDBS for train ligands: {lmdb_regex}\n into: {train_full_lig_path}")
# paths = sorted(glob(lmdb_regex))
# concat_lmdbs_preserve_keys(paths, train_full_lig_path)
# print('done with that!')

# #val target 
# train_full_lig_path = "/mnt/goon/benchmark_code/drugclip_data/full_val_target_dataset.lmdb"
# lmdb_regex ="/mnt/goon/benchmark_code/drugclip_data/protein_data/val/*train_chunk_*.lmdb"
# print(f"Merging LMDBS for train ligands: {lmdb_regex}\n into: {train_full_lig_path}")
# paths = sorted(glob(lmdb_regex))
# concat_lmdbs_preserve_keys(paths, train_full_lig_path)
# print('done with that!')

#test ligands
train_full_lig_path = "/mnt/goon/benchmark_code/drugclip_data/full_test_lig_dataset_v2.lmdb"
lmdb_regex ="/mnt/goon/benchmark_code/drugclip_data/ligand_data/test/*train_chunk_*.lmdb"
print(f"Merging LMDBS for train ligands: {lmdb_regex}\n into: {train_full_lig_path}")
paths = sorted(glob(lmdb_regex))
concat_lmdbs_preserve_keys(paths, train_full_lig_path)
print('done with that!')



#no test targets, reused from val. 
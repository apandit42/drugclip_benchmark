# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import lmdb
import os
import pickle
from functools import lru_cache
import logging
import random
import json

logger = logging.getLogger(__name__)


class LMDBDataset:
    def __init__(self, db_path, lmdb_map_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)
        env = self.connect_db(self.db_path)
        with open(lmdb_map_path, 'r') as file:
            self.lmdb_map = json.load(file)
        self._keys = self.lmdb_map['lmdb_keys'].keys()

        # with env.begin() as txn:
        #     self._keys = list(txn.cursor().iternext(values=False))

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, "env"):
            self.connect_db(self.db_path, save_to_self=True)
        #datapoint_pickled = self.env.begin().get(f"{idx}".encode("ascii"))
        #print(idx)
        uniprot_id, ligand_key = self.lmdb_map['lmdb_keys'][str(idx)]
        prot_key = random.choice(self.lmdb_map['pdb_map'][uniprot_id])
        # prot_key = f"{uniprot_id}_{pdb_id}"
        # print(prot_key)
        # breakpoint()
        datapoint_pickled = self.env.begin().get(prot_key.encode())
        data_pocket = pickle.loads(datapoint_pickled)
        datapoint_pickled = self.env.begin().get(ligand_key.encode())
        data_ligand = pickle.loads(datapoint_pickled)
        
        data = data_pocket | data_ligand
        del data['label']
        return data

# https://github.com/Lyken17/Efficient-PyTorch/blob/master/tools/folder2lmdb.py

import os
import os.path as osp
import os, sys
import os.path as osp
from PIL import Image
import six
import string

import lmdb
import pickle
import msgpack
import tqdm
import pyarrow as pa
import numpy as np
import lz4framed


import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets


def compress_serialize(thing):
    return lz4framed.compress(pa.serialize(thing).to_buffer())

def deserialize_decompress(thing):
    return pa.deserialize(lz4framed.decompress(thing))

class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, targets=None, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length =deserialize_decompress(txn.get(b'__len__'))
            self.keys= deserialize_decompress(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

        self.targets = targets or np.array([self[i][1] for i in range(self.length)])

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = deserialize_decompress(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def folder2lmdb(in_path, out_path, write_frequency=5000, num_workers=8, map_size=1e11):
    directory = osp.expanduser(in_path)
    print("Loading dataset from %s" % directory)
    dataset = ImageFolder(directory, loader=raw_reader)
    data_loader = DataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x)

    lmdb_path = out_path
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=map_size, readonly=False,
                   meminit=False, map_async=True)
    
    labels = []
    print(len(dataset), len(data_loader))
    txn = db.begin(write=True)
    for idx, data in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
        # print(type(data), data)
        image, label = data[0]
        txn.put(u'{}'.format(idx).encode('ascii'), compress_serialize((image, label)))
        if idx % write_frequency == 0:
            txn.commit()
            txn = db.begin(write=True)
        labels.append(label)

    # finish iterating through dataset
    print('Final commit')
    txn.commit()

    print('Writing keys and len')
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', compress_serialize(keys))
        txn.put(b'__len__', compress_serialize(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()

    return labels


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str)
    parser.add_argument('-s', '--split', type=str, default="val")
    parser.add_argument('--out', type=str, default=".")
    parser.add_argument('-p', '--procs', type=int, default=20)

    args = parser.parse_args()

    folder2lmdb(args.folder, num_workers=args.procs, name=args.split)
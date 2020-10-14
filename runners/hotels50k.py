import torch
from torch.utils.data import Dataset
from torchvision import datasets
import numpy as np

import csv, multiprocessing, cv2, os
import numpy as np
import urllib
import urllib.request
import subprocess

from tqdm import tqdm

from powerful_benchmarker.split_managers import ClassDisjointSplitManager, BaseSplitManager, IndexSplitManager
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import pickle
import lmdb

from utils import ImageFolderLMDB, folder2lmdb, raw_reader, compress_serialize


class AppURLopener(urllib.request.FancyURLopener):
    version = "Mozilla/5.0"

opener = AppURLopener()
def url_to_image(url):
    resp = opener.open(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    return image


def make_lmdb(root, out_path, write_frequency=5000, num_workers=8, map_size=1e11):
    idx = 0

    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    for dataset in ['train', 'test']:
        lmdb_path = os.path.join(out_path, f'{dataset}.lmdb')

        print("Generate LMDB to %s" % lmdb_path)
        db = lmdb.open(lmdb_path, 
            subdir=False, 
            map_size=map_size, 
            readonly=False,
            meminit=False,
            map_async=True)

        txn = db.begin(write=True)

        dataset_dir = dataset
        if dataset == 'test':
            dataset_dir = os.path.join('test', 'unoccluded')            

        chains = []
        hotels = []
        for chain in tqdm(os.listdir(os.path.join(root, 'images', dataset_dir))):
            if chain.startswith('.'):
                continue
            for hotel in tqdm(os.listdir(os.path.join(root, 'images', dataset_dir, chain))):
                if hotel.startswith('.'):
                    continue

                try:
                    ds = ImageFolder(os.path.join(root, 'images', dataset_dir, chain, hotel), loader=raw_reader)
                    data_loader = DataLoader(ds, num_workers=num_workers, collate_fn=lambda x: x)
                except RuntimeError as e:
                    print(e)
                    continue

                for data in data_loader:
                    image, im_source = data[0]

                # for im_source in os.listdir(os.path.join(root, 'images', dataset_dir, chain, hotel)):
                #     if im_source.startswith('.'):
                #         continue
                #     for image_name in os.listdir(os.path.join(root, 'images', dataset_dir, chain, hotel, im_source)):
                #         if image_name.startswith('.'):
                #             continue
                #         img_path = os.path.join(root, 'images', dataset_dir, chain, hotel, im_source, image_name)

                #         with open(img_path, 'rb') as f:
                #             image = f.read()

                    txn.put(u'{}'.format(idx).encode('ascii'), compress_serialize((image, chain, hotel, im_source)))

                    chains.append(int(chain))
                    hotels.append(int(hotel))

                    if idx % write_frequency == 0:
                        txn.commit()
                        txn = db.begin(write=True)

                    idx += 1


        txn.commit()

        print('Writing keys and len')
        keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
        with db.begin(write=True) as txn:
            txn.put(b'__keys__', compress_serialize(keys))
            txn.put(b'__len__', compress_serialize(len(keys)))

        print("Flushing database ...")
        db.sync()
        db.close()

        print(f'Dumping targets for {dataset}')

        with open(os.path.join(out_path, f'{dataset}_chains.pkl'), 'wb') as f:
            pickle.dump(chains, f)

        with open(os.path.join(out_path, f'{dataset}_hotels.pkl'), 'wb') as f:
            pickle.dump(hotels, f)


def make_symlinks(root):
    for dataset in ['train', 'test']:
        chains_dir = os.path.join(root, 'symlinks', dataset, 'chains')
        hotels_dir = os.path.join(root, 'symlinks', dataset, 'hotels')

        os.makedirs(chains_dir, exist_ok=True)
        os.makedirs(hotels_dir, exist_ok=True)

        if dataset == 'test':
            dataset = os.path.join('test', 'unoccluded')            

        for chain in os.listdir(os.path.join(root, 'images', dataset)):
            if chain.startswith('.'):
                continue
            for hotel in os.listdir(os.path.join(root, 'images', dataset, chain)):
                if hotel.startswith('.'):
                    continue
                for im_source in os.listdir(os.path.join(root, 'images', dataset, chain, hotel)):
                    if im_source.startswith('.'):
                        continue
                    for image_name in os.listdir(os.path.join(root, 'images', dataset, chain, hotel, im_source)):
                        if image_name.startswith('.'):
                            continue
                        img_path = os.path.join(root, 'images', dataset, chain, hotel, im_source, image_name)

                        os.makedirs(os.path.join(chains_dir, chain), exist_ok=True)
                        os.makedirs(os.path.join(hotels_dir, hotel), exist_ok=True)

                        chain_symlink_path = os.path.join(chains_dir, chain, image_name)
                        if not os.path.exists(chain_symlink_path):
                            os.symlink(img_path, chain_symlink_path)

                        hotel_symlink_path = os.path.join(hotels_dir, hotel, image_name)
                        if not os.path.exists(hotel_symlink_path):
                            os.symlink(img_path, hotel_symlink_path)

                        print('Symlink created:', chain_symlink_path, hotel_symlink_path)
            

def download_and_resize(root, image_list):
    for im in image_list:
        try:
            (chain,hotel,im_source,im_id,im_url) = im
            saveDir = os.path.join(root, 'images/train/', chain, hotel, im_source)
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)

            img_name = str(im_id)+'.'+ im_url.split('.')[-1]
            savePath = os.path.join(saveDir, img_name)

            if not os.path.isfile(savePath):
                img = url_to_image(im[4])
                if img.shape[1] > img.shape[0]:
                    width = 640
                    height = round((640 * img.shape[0]) / img.shape[1])
                    img = cv2.resize(img,(width, height))
                else:
                    height = 640
                    width = round((640 * img.shape[1]) / img.shape[0])
                    img = cv2.resize(img,(width, height))
                cv2.imwrite(savePath, img)

                print('Saved: ' + savePath)
            else:
                print('Already saved: ' + savePath)
        except Exception:
            print('Failed to download image')

def download_train_images(root):
    hotel_f = open(os.path.join(root, 'input/dataset/hotel_info.csv'),'r')
    hotel_reader = csv.reader(hotel_f)
    hotel_headers = next(hotel_reader,None)
    hotel_to_chain = {}
    for row in hotel_reader:
        hotel_to_chain[row[0]] = row[2]

    train_f = open(os.path.join(root, 'input/dataset/train_set.csv'),'r')
    train_reader = csv.reader(train_f)
    train_headers = next(train_reader,None)

    images = []
    for im in train_reader:
        im_id = im[0]
        im_url = im[2]
        im_source = im[3]
        hotel = im[1]
        chain = hotel_to_chain[hotel]
        images.append((chain,hotel,im_source,im_id,im_url))

    NUM_THREADS = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(NUM_THREADS)

    imDict = {}
    for cpu in range(NUM_THREADS):
        pool.apply_async(download_and_resize, [root, images[cpu::NUM_THREADS]])
    print('Started subprocesses')
    pool.close()
    print('Joining threads')
    pool.join()
    print('Done downloading train')




def download_hotels_50k(root):
    print(f'Cloning https://github.com/GWUvision/Hotels-50K into {root}')    
    if not os.path.exists(os.path.join(root, 'input')):
        subprocess.run(['git', 'clone', 'https://github.com/GWUvision/Hotels-50K.git', root],  capture_output=True, check=True)
    else:
        print('Repo already cloned')

    print('Unpacking dataset archive')
    archive_path = os.path.join(root, 'input', 'dataset.tar.gz')
    out_path = os.path.join(root, 'input')    
    if not os.path.exists(out_path):
        subprocess.run(['tar', '-C', out_path, '-zxvf', archive_path],  capture_output=True, check=True)
    else:
        print('Archive already unpacked')

    print('Downloading train images')
    if os.path.exists(os.path.join(root, 'images', 'train')):
        print('Train images already loaded')
    else:
        download_train_images(root)

    print('Downloading test images archive')
    url = 'https://cs.slu.edu/~stylianou/images/hotels-50k/test.tar.lz4'
    archive_path = os.path.join(root, 'images', 'test.tar.lz4')
    if not os.path.exists(archive_path):
        subprocess.run(['wget', url, '-O', archive_path, '--no-check-certificate'],  capture_output=True, check=True)
    else:
        print('Archive already downloaded')

    out_path = os.path.join(root, 'images')
    if not os.path.exists(os.path.join(out_path, 'test')):
        command = ['lz4', '-dc', '--no-sparse',  archive_path, '|',  'tar', '-C', out_path, '-xz']
        print(' '.join(command))
        subprocess.run(command,  capture_output=True, check=True)
    else:
        print('Archive already unpacked')

    print('Creaing LMDB files')
    lmdb_path = os.path.join(root, 'lmdb')
    if os.path.exists(lmdb_path):
        print('LMDB dir already exists')
    else:
        make_lmdb(root, lmdb_path)

    # print('Creating symlinks')
    # make_symlinks(root)

    # print('Creating LMDB image folders')

    # symlinks_dir = os.path.join(root, 'symlinks')

    # for target in ('chains', 'hotels'):
    #     for split in ('test', 'train'):
    #         symlinks_dir = os.path.join(root, 'symlinks', split, target)

    #         lmdb_dir = os.path.join(root, 'lmdb', target)
    #         lmdb_path = os.path.join(lmdb_dir, f'{split}.lmdb')
    #         targets_path = os.path.join(lmdb_dir, f'{split}_targets.pkl')

    #         if os.path.exists(lmdb_path):
    #             print('LMDB file already exists')
    #             continue

    #         os.makedirs(lmdb_dir, exist_ok=True)

    #         print(f'Creating LMDB file for target {target} and split {split}')
    #         targets = folder2lmdb(symlinks_dir, lmdb_path)
    #         print(f'Storing targets at {targets_path}')
    #         with open(targets_path, 'wb') as f:
    #             pickle.dump(targets, f)
            

    print('Done downloading and setting up the dataset.')




class Hotels50kDataset(Dataset):
    def __init__(self, root, target='chains', transform=None, download=False):
        assert target in ('chains', 'hotels')
        self.target = target
        if download:
            download_hotels_50k(root)

        train_path = os.path.join(root, 'lmdb', 'train.lmdb')
        train_chains_path = os.path.join(root, 'lmdb', 'train_chains.pkl')
        train_hotels_path = os.path.join(root, 'lmdb', 'train_hotels.pkl')

        test_path = os.path.join(root, 'lmdb', 'test.lmdb')
        test_chains_path = os.path.join(root, 'lmdb', 'test_chains.pkl')
        test_hotels_path = os.path.join(root, 'lmdb', 'test_hotels.pkl')

        with open(train_chains_path, 'rb') as f:
            self.train_chains = pickle.load(f)

        with open(train_hotels_path, 'rb') as f:
            self.train_hotels = pickle.load(f)

        with open(test_chains_path, 'rb') as f:
            self.test_chains = pickle.load(f)

        with open(test_hotels_path, 'rb') as f:
            self.test_hotels = pickle.load(f)



        self.train_targets = self.train_chains if self.target == 'chains' else self.train_hotels
        self.test_targets = self.test_chains if self.target == 'chains' else self.test_hotels

        print('Loading image folders')
        self.original_train_dataset = ImageFolderLMDB(train_path, transform=transform)
        self.original_test_dataset = ImageFolderLMDB(test_path, transform=transform)
        
        self.train_indices = np.arange(len(self.original_train_dataset))
        self.test_indices = np.arange(len(self.original_train_dataset), len(self.original_train_dataset)+len(self.original_test_dataset))

        print('Concatting dataset')
        self.dataset = torch.utils.data.ConcatDataset([self.original_train_dataset, self.original_test_dataset])

        print('Getting labels')
        # these look useless, but are required by powerful-benchmarker
        self.labels = np.concatenate([self.train_targets, self.test_targets])
        self.transform = transform 

        print('Done loading dataset')
    
    def get_split_indices(self, split_name):
        if split_name == "test":
            return self.test_indices
        return None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, meta = self.dataset[idx]
        if self.target == 'chains':
            label = meta[0]
        elif self.target == 'hotels':
            label = meta[1]
        return dict(data=img, label=label)


class UseOriginalTestSplitManager(BaseSplitManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_names = ["test"]

    def _create_split_schemes(self, datasets):
        output = {}
        for transform_type, v1 in datasets.items():
            output[transform_type] = {}
            for split_name, v2 in v1.items():
                indices = v2.get_split_indices(split_name)
                if indices is not None:
                    output[transform_type][split_name] = torch.utils.data.Subset(v2, indices)
                else:
                    output[transform_type][split_name] = v2
        return {self.get_split_scheme_name(0): output}

    def get_test_set_name(self):
        return 'UsingOriginalTest'

    def get_base_split_scheme_name(self):
        return self.get_test_set_name()

    def get_split_scheme_name(self, partition):
        return self.get_base_split_scheme_name()

    def split_assertions(self):
        pass
    

if __name__ == "__main__":
    root = os.path.join(os.getcwd(), 'hotels50k')
    dataset = Hotels50kDataset(root=root, target='hotels', download=True)

    print('Retrieving example')
    for obs in dataset:
        print(obs['label'])
        print(np.array(obs['data']).shape)
        break

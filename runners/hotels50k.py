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
import torch
import numpy as np
import pickle

from utils import ImageFolderLMDB, folder2lmdb


class AppURLopener(urllib.request.FancyURLopener):
    version = "Mozilla/5.0"

opener = AppURLopener()
def url_to_image(url):
    resp = opener.open(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    return image


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

    print('Creating symlinks')
    #make_symlinks(root)

    print('Creating LMDB image folders')

    symlinks_dir = os.path.join(root, 'symlinks')

    for target in ('chains', 'hotels'):
        for split in ('test', 'train'):
            symlinks_dir = os.path.join(root, 'symlinks', split, target)

            lmdb_dir = os.path.join(root, 'lmdb', target)
            lmdb_path = os.path.join(lmdb_dir, f'{split}.lmdb')
            targets_path = os.path.join(lmdb_dir, f'{split}_targets.pkl')

            if os.path.exists(lmdb_path):
                print('LMDB file already exists')
                continue

            os.makedirs(lmdb_dir, exist_ok=True)

            print(f'Creating LMDB file for target {target} and split {split}')
            targets = folder2lmdb(symlinks_dir, lmdb_path)
            print(f'Storing targets at {targets_path}')
            with open(targets_path, 'wb') as f:
                pickle.dump(targets, f)
            

    print('Done downloading and setting up the dataset.')




class Hotels50kDataset(Dataset):
    def __init__(self, root, target='chains', transform=None, download=False):
        assert target in ('chains', 'hotels')
        if download:
            download_hotels_50k(root)

        train_path = os.path.join(root, 'lmdb', target, 'train.lmdb')
        train_targets_path = os.path.join(root, 'lmdb', target, 'train_targets.pkl')
        test_path = os.path.join(root, 'lmdb', target, 'test.lmdb')
        test_targets_path = os.path.join(root, 'lmdb', target, 'test_targets.pkl')

        with open(train_targets_path, 'rb') as f:
            train_targets = pickle.load(f)

        with open(test_targets_path, 'rb') as f:
            test_targets = pickle.load(f)

        print('Loading image folders')
        self.original_train_dataset = ImageFolderLMDB(train_path, targets=train_targets, transform=transform)
        self.original_test_dataset = ImageFolderLMDB(test_path, targets=test_targets, transform=transform)
        
        self.train_indices = np.arange(len(self.original_train_dataset))
        self.test_indices = np.arange(len(self.original_train_dataset), len(self.original_train_dataset)+len(self.original_test_dataset))

        print('Concatting dataset')
        self.dataset = torch.utils.data.ConcatDataset([self.original_train_dataset, self.original_test_dataset])

        print('Getting labels')
        # these look useless, but are required by powerful-benchmarker
        self.labels = np.concatenate([self.original_train_dataset.targets, self.original_test_dataset.targets])
        self.transform = transform 
    
    def get_split_indices(self, split_name):
        if split_name == "test":
            return self.test_indices
        return None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
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
    #root = '/data/thesis/Hotels-50K'
    dataset = Hotels50kDataset(root=root, target='hotels', download=True)

    for obs in dataset:
        print(obs['label'])
        print(np.array(obs['data']).shape)
        break

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import numpy as np
import pandas as pd
import csv, multiprocessing, cv2, os
import urllib
import urllib.request
import subprocess
from tqdm import tqdm
from powerful_benchmarker.split_managers import ClassDisjointSplitManager, BaseSplitManager, IndexSplitManager
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
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    for dataset in ['train', 'test']:
        keys = []
        idx = 0

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

                    key = u'{}'.format(idx).encode('ascii')

                    txn.put(key, compress_serialize((image, chain, hotel, im_source)))

                    chains.append(int(chain))
                    hotels.append(int(hotel))

                    if idx % write_frequency == 0:
                        txn.commit()
                        txn = db.begin(write=True)

                    keys.append(key)
                    idx += 1

        txn.commit()

        print('Writing keys and len')
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


def download_and_resize(root, image_list, basedir='images/train'):
    for im in image_list:
        try:
            (chain,hotel,im_source,im_id,im_url) = tuple([str(x) for x in im])
            saveDir = os.path.join(root, basedir, chain, hotel, im_source)
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
        except Exception as e:
            print(e)
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

def download_images_subset(root, df, basedir):
    images = []
    for ix, row in df.iterrows():
        im_id = row.image_id
        im_url = row.image_url
        im_source = row.image_source
        hotel = row.hotel_id
        chain = row.chain_id
        images.append((chain,hotel,im_source,im_id,im_url))

    NUM_THREADS = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(NUM_THREADS)

    imDict = {}
    for cpu in range(NUM_THREADS):
        pool.apply_async(download_and_resize, [root, images[cpu::NUM_THREADS]], {'basedir': basedir})
    print('Started subprocesses')
    pool.close()
    print('Joining threads')
    pool.join()
    print('Done downloading train')


def prepare_hotels50k_subset(root, seed=0):
    np.random.seed(seed)
    hinfo = pd.read_csv(os.path.join(root, 'input', 'dataset',  'hotel_info.csv'))
    
    hotels_to_chain = hinfo[['hotel_id', 'chain_id']]
    hotels_to_chain.index = hotels_to_chain['hotel_id']
    hotels_to_chain = hotels_to_chain['chain_id']
    
    df = pd.read_csv(os.path.join(root, 'input', 'dataset', 'train_set.csv'), header=None)
    df.columns = ['image_id', 'hotel_id', 'image_url', 'image_source', 'upload_timestamp']
    df['chain_id'] = hotels_to_chain[df.hotel_id].values
    
    photos_per_hotel = df.groupby(['hotel_id']).agg({'image_url': 'count'})
    
    photos_per_chain = df.groupby(['chain_id']).agg({'image_url': 'count'})
    
    # Remove chains with <= 100 photos
    bad_chains = set(photos_per_chain[(photos_per_chain.reset_index()['image_url'] < 100).values].index)
    
    bad_chain_hotels = []
    for chain in bad_chains:
        chain_hotels = hinfo[['hotel_id', 'chain_id']][hinfo['chain_id'] == chain]
        bad_chain_hotels += list(chain_hotels['hotel_id'])
    
    # Remove hotels with <= 10 photos
    bad_hotels = list(photos_per_hotel[(photos_per_hotel.reset_index()['image_url'] < 10).values].index)
    
    bad_hotels += bad_chain_hotels
    
    df = df[~df.hotel_id.isin(bad_hotels)]
    
    # Keep all trafficcam images, sample the rest
    trafficcam_df = df[df.image_source == 'traffickcam']
    other_df = df[df.image_source != 'traffickcam']
    
    sample_size = len(trafficcam_df)
    other_df = other_df.sample(sample_size)
    
    subset_train_df = pd.concat([trafficcam_df, other_df])
    
    
    test_df = pd.read_csv(os.path.join(root, 'input', 'dataset', 'test_set.csv'))
    test_df['chain_id'] = hotels_to_chain[test_df.hotel_id].values
    
    subset_test_df = test_df[test_df.hotel_id.isin(subset_train_df.hotel_id)]
    
    return subset_train_df, subset_test_df

def print_subset_stats(subset_train_df, subset_test_df):
    print('Prepared dataset subset')
    print('Train chains:', subset_train_df.chain_id.value_counts().shape[0])
    print('Test chains:', subset_test_df.chain_id.value_counts().shape[0])
    print('Train hotels:', subset_train_df.hotel_id.value_counts().shape[0])
    print('Test hotels:', subset_test_df.hotel_id.value_counts().shape[0])

    print('Train images', subset_train_df.shape[0])
    print('Train traffickcam images', subset_train_df[subset_train_df.image_source == 'traffickcam'].shape[0])
    
    print('Test images', subset_test_df.shape[0])
    print('Test traffickcam images', subset_test_df[subset_test_df.image_source == 'traffickcam'].shape[0])

    return subset_train_df, subset_test_df


def download_hotels_50k(root, subset=True, train_subset_path=None, test_subset_path=None):
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

    train_csv = os.path.join(root, 'input', 'dataset', 'train_set.csv')
    test_csv = os.path.join(root, 'input', 'dataset', 'test_set.csv')
    if subset:
        print('Loading dataset subset')
        train_subset_path = train_subset_path or os.path.join(root, 'input', 'train_subset.csv')
        test_subset_path = test_subset_path or os.path.join(root, 'input', 'test_subset.csv')
        if not os.path.exists(train_subset_path) or not os.path.exists(test_subset_path):
            train_df, test_df = prepare_hotels50k_subset(root)

            train_df.to_csv(train_subset_path, index=None)
            test_df.to_csv(test_subset_path, index=None)
        else:
            print('Subset csvs already exist')
            train_df = pd.read_csv(train_subset_path)
            test_df = pd.read_csv(test_subset_path)


        print('Loading subset csvs')
        train_csv = os.path.join(root, 'input', 'train_subset.csv')
        test_csv = os.path.join(root, 'input', 'test_subset.csv')

        print_subset_stats(train_df, test_df)

        print('Downloading train subset images')
        download_images_subset(root, train_df, basedir='images/train')

        print('Downloading test subset images')
        download_images_subset(root, test_df, basedir='images/test/unoccluded')

    else:
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


    print('Done downloading and setting up the dataset.')




class Hotels50kDataset(Dataset):
    def __init__(self, root, target='chains', transform=None, download=False, subset=True, train_subset_path=None, test_subset_path=None):
        assert target in ('chains', 'hotels')
        self.target = target


        if download:
            train_subset_path = train_subset_path or os.environ.get('HOTELS50K_TRAIN_SUBSET')
            test_subset_path = test_subset_path or os.environ.get('HOTELS50K_TEST_SUBSET')
            download_hotels_50k(root, subset=subset, train_subset_path=train_subset_path, test_subset_path=test_subset_path)

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
        label = int(label)
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
    root = os.environ.get('HOTELS50K_ROOT') or os.path.join(os.getcwd(), 'hotels50k')
    dataset = Hotels50kDataset(root=root, target='hotels', download=True)

    print('Retrieving example')
    for obs in dataset:
        print('Label:', obs['label'])
        print('Data shape:', np.array(obs['data']).shape)
        break

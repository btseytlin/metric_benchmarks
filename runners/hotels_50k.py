from torch.utils.data import Dataset
from torchvision import datasets
import numpy as np

import csv, multiprocessing, cv2, os
import numpy as np
import urllib
import urllib.request
import subprocess

class AppURLopener(urllib.request.FancyURLopener):
    version = "Mozilla/5.0"

opener = AppURLopener()
def url_to_image(url):
    resp = opener.open(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    return image


def download_and_resize(root, image_list):
    for im in image_list:
        try:
            (chain,hotel,im_source,im_id,im_url) = im
            saveDir = os.path.join(root, 'images/train/', chain, hotel, im_source)
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)

            savePath = os.path.join(saveDir, str(im_id)+'.'+ im_url.split('.')[-1])

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
                cv2.imwrite(savePath,img)
                print('Good: ' + savePath)
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
    out_path = os.path.join(root, 'input', 'dataset')    
    if not os.path.exists(out_path):
        subprocess.run(['tar', '-C', out_path, '-zxvf', archive_path],  capture_output=True, check=True)
    else:
        print('Archive already unpacked')

    print('Downloading train images')
    download_train_images(root)

    print('Downloading test images archive')
    url = 'https://cs.slu.edu/~stylianou/images/hotels-50k/test.tar.lz4'
    archive_path = os.path.join(root, 'images', 'test.tar.lz4')
    if not os.path.exists(archive_path):
        subprocess.run(['wget', url, '-O', archive_path],  capture_output=True, check=True)
    else:
        print('Archive already downloaded')

    out_path = os.path.join(root, 'images', 'test')
    if not os.path.exists(out_path):
        subprocess.run(['lz4', '-dc', '--no-sparse', '|',  'tar', '-C', out_path, '-x'],  capture_output=True, check=True)
    else:
        print('Archive already unpacked')


class Hotels50KDatasetChains(Dataset):
    def __init__(self, root, transform=None, download=False):
        if download:
            retcode = download_hotels_50k(root)

        self.dataset = datasets.ImageFolder(root, transform=transform)

        # these look useless, but are required by powerful-benchmarker
        self.labels = np.array([b for (a, b) in self.dataset.imgs])
        self.transform = transform 
 
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

if __name__ == "__main__":
    root = '/data/thesis/Hotels-50K'
    dataset = Hotels50KDatasetChains(root=root, download=True)

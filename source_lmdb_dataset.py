""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """
import sys

import fire
import os
import lmdb
import cv2

import numpy as np

from load import loadLmdb


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def SourceDataset(inputPath1=r'data_lmdb_release\training\MY', inputPath2=r'data_lmdb_release\validation', outputPath=r'data_lmdb_release', checkValid=True):
    outputPathSource = os.path.join(outputPath, 'new')
    os.makedirs(outputPath, exist_ok=True)
    os.makedirs(outputPathSource, exist_ok=True)
    # env = lmdb.open(outputPathSource, map_size=1099511627776)
    env = lmdb.open(outputPathSource, map_size=1e10)
    cache = {}
    cnt = 1

    datalist1 = loadLmdb(inputPath1)
    datalist2 = loadLmdb(inputPath2)
    datalist = datalist1 + datalist2

    nSamples = len(datalist)
    for i in range(nSamples):
        imagePath, label = datalist[i].strip('\n').split('\t')
        imagePath = imagePath.split('/')[-1]

        # # only use alphanumeric data
        # if re.search('[^a-zA-Z0-9]', label):
        #     continue
        if not os.path.exists(inputPath1):
            print('%s does not exist' % inputPath1)
            return
        if not os.path.exists(inputPath2):
            print('%s does not exist' % inputPath1)
            return
        print(imagePath, label)

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imagePath.encode()
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    # fire.Fire(createDataset)
    # SourceDataset()
    SourceDataset(sys.argv[1],sys.argv[2],sys.argv[3])
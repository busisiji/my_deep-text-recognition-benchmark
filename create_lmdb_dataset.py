""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

import fire
import os
import lmdb
import cv2

import numpy as np


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def writeTOLmdb(env,datalist,i,cnt,inputPath,outputPath,checkValid,cache,nSamples):
    imagePath, label = datalist[i].strip('\n').split('\t')
    # imagePath, label = datalist[i].strip('\n').split('.jpg')
    # imagePath = imagePath + '.jpg'
    imagePath = os.path.join(inputPath, imagePath)

    # # only use alphanumeric data
    # if re.search('[^a-zA-Z0-9]', label):
    #     continue

    if not os.path.exists(imagePath):
        print('%s does not exist' % imagePath)
        return
    with open(imagePath, 'rb') as f:
        imageBin = f.read()
    if checkValid:
        try:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                return
        except:
            print('error occured', i)
            with open(outputPath + '/error_image_log.txt', 'a') as log:
                log.write('%s-th image data occured error\n' % str(i))
            return

    imageKey = 'image-%09d'.encode() % cnt
    labelKey = 'label-%09d'.encode() % cnt
    cache[imageKey] = imageBin
    cache[labelKey] = label.encode()

    if cnt % 1000 == 0:
        writeCache(env, cache)
        cache = {}
        print('Written %d / %d' % (cnt, nSamples))

def createDataset(inputPath, gtFile, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    outputPathTrain = os.path.join(outputPath, 'train')
    outputPathValid = os.path.join(outputPath, 'valid')
    os.makedirs(outputPath, exist_ok=True)
    os.makedirs(outputPathTrain, exist_ok=True)
    os.makedirs(outputPathValid, exist_ok=True)
    env = lmdb.open(outputPath,  map_size=int(1e11))
    envTrain = lmdb.open(outputPathTrain,  map_size=int(1e8))
    envValid = lmdb.open(outputPathValid,  map_size=int(1e8))
    # env = lmdb.open(outputPath, map_size=1099511627776)

    cache = {}
    cnt = 1
    with open(gtFile, 'r', encoding='utf-8') as data:
        datalist = data.readlines()

    nSamples = len(datalist)
    for i in range(int(nSamples*0.8)):
        writeTOLmdb(envTrain,datalist,i,cnt,inputPath,outputPath,checkValid,cache,nSamples)
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(envTrain, cache)
    print('Created dataset with %d samples' % nSamples)

    cache = {}
    cnt = 1
    for i in range(int(nSamples * 0.8),nSamples):
        writeTOLmdb(envValid,datalist,i,cnt,inputPath,outputPath,checkValid,cache,nSamples)
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(envValid, cache)
    print('Created dataset with %d samples' % nSamples)

if __name__ == '__main__':
    fire.Fire(createDataset)

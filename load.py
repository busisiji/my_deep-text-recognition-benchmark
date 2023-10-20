import lmdb

def loadLmdb(path = r'result\train'):
    env = lmdb.open(path,  map_size=int(1e11), readonly=True, lock=False, readahead=False, meminit=False)
    txn = env.begin()
    index = 0
    # while 1:
    for index in range(10):
        index += 1  # lmdb starts with 1
        label_key = 'label-%09d'.encode() % index
        label = txn.get(label_key)
        if not label:
            return
        label = label.decode('utf-8')
        print(label_key,label)

if __name__ == '__main__':
    loadLmdb(r'data_lmdb_release\training\MJ\MJ_train')
    loadLmdb(r'data_lmdb_release\validation')
    loadLmdb(r'result\train')
    loadLmdb(r'result\valid')
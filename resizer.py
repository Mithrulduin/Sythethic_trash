import glob
import cv2
from tqdm.contrib.concurrent import process_map

def resize(chunk):
    for i in chunk:
        img = cv2.imread(i, cv2.IMREAD_UNCHANGED)
        resized = cv2.resize(img, (800,600), interpolation=cv2.INTER_AREA)
        cv2.imwrite(i, resized)

def gen_chunks(l, n):
    chunk_size = len(l)//n
    for i in range(n-1):
        yield l[:chunk_size]
        l = l[chunk_size:]
    yield l


chunks = gen_chunks(glob.glob("/datadrive/trash/Jura/*.png"), 6)

r = process_map(resize, chunks, max_workers=6)




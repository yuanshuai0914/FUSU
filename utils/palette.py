import numpy as np
import os
from PIL import Image
from tqdm import tqdm


def color_map():
    cmap = np.zeros((18, 3), dtype=np.uint8)
    cmap[0] = np.array([255, 255, 255])
    cmap[1] = np.array([203,9,9])
    cmap[2] = np.array([8,154,230])
    cmap[3] = np.array([210,42,62])
    cmap[4] = np.array([126,211,33])
    cmap[5] = np.array([135,126,20])
    cmap[6] = np.array([94,47,4])
    cmap[7] = np.array([112,57,9])
    cmap[8] = np.array([184,233,134])
    cmap[9] = np.array([201,8,241])
    cmap[10] = np.array([127,123,127])
    cmap[11] = np.array([252,232,5])
    cmap[12] = np.array([96, 189, 253])
    cmap[13] = np.array([243, 229, 176])
    cmap[14] = np.array([168,153,13])
    cmap[15] = np.array([70,106,28])
    cmap[16] = np.array([161,91,176])
    cmap[17] = np.array([18,227,180])
    
    
    return cmap


if __name__ == '__main__':
    path = '/nfs-data2/ys/ChangeDetection/data/crop_img/im1/'
    filenames = os.listdir(path)

    cmap = color_map()

    for filename in tqdm(filenames):
        mask = Image.open(os.path.join(path, filename)).convert("P")
        mask.putpalette(cmap)
        mask.save(os.path.join(path, filename))

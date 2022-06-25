from utils import *

import os.path

if __name__ == "__main__":
    files = np.char.add('data/tracks_wav/', os.listdir('data/tracks_wav'))
    #files = files[:100]

    x = get_features(files)
    x.to_csv('new_features.csv')

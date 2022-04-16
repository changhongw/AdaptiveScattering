import time
import os
from fnmatch import fnmatch
import subprocess
import warnings
warnings.filterwarnings('ignore')

# check wav files of the CBFdataset
data_root = '/import/c4dm-datasets/ChineseBambooFluteDataset/CBF-multilabel-binaryClassf/'
target = "*.wav"
wav_files = []  
for path, subdirs, files in os.walk(data_root):
    for name in files:
        if fnmatch(name, target):
            wav_files.append(os.path.join(path, name))  
print('Number of audio files:', format(len(wav_files)))

# save file names for convenient feature extraction by matlab
with open('file_names.txt', 'w') as f:
    for item in wav_files:
        f.write("%s\n" % item)


# extract AdaTS+AdaTRS feature in matlab
t0 = time.time()
subprocess.call(["matlab",
                 "-r",
                 "frequency_adaptive_feature_extraction",
                 "-nodisplay",
                 "-nodesktop"])

print('frequency-adaptive feature extraction time:%.2f hours.' % ((time.time() - t0)/3600))


# extract dJTFS-avg feature in matlab
t0 = time.time()
subprocess.call(["matlab",
                 "-r",
                 "direction_adaptive_feature_extraction",
                 "-nodisplay",
                 "-nodesktop"])

print('direction-adaptive feature extraction time:%.2f hours.' % ((time.time() - t0)/3600))
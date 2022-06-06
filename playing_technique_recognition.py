import time
import os, torch
from tqdm import tqdm_notebook as tqdm
import scipy.io
from fnmatch import fnmatch
import librosa
import pandas as pd
import numpy as np
import re, collections, requests
import matplotlib.pyplot as plt
import collections, re
import subprocess
import warnings
warnings.filterwarnings('ignore')

# for classification
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from thundersvm import SVC  # use GPU for SVM
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


###### load data ######
# check wav files of the dataset
data_root = 'data/'
target = "*.wav"
wav_files = []  
for path, subdirs, files in os.walk(data_root):
    for name in files:
        if fnmatch(name, target):
            wav_files.append(os.path.join(path, name))  
print('Number of audio files:', format(len(wav_files)))

for k in range(len(wav_files)):
    wav_files[k] = wav_files[k].replace(data_root,'')

###### Load extracted features ######
# load the extracted frequency-adaptive feature
adapt_time = scipy.io.loadmat('feature_extraction_Matlab/frequency_adaptive_feature.mat')['fileFeatures_time'][0,:]
adapt_timerate = scipy.io.loadmat('feature_extraction_Matlab/frequency_adaptive_feature.mat')['fileFeatures_timerate'][0,:]

print(adapt_time.shape, adapt_timerate.shape)

adapt = []
for k in range(adapt_time.shape[0]):
    adapt.append(np.vstack((adapt_time[k], adapt_timerate[k])))
    
adapt_time_dim = adapt_time[0].shape[0]
del(adapt_time,adapt_timerate)
print(adapt_time_dim, adapt[0].shape[0])


# load the extracted direction-adaptive feature
joint = scipy.io.loadmat('feature_extraction_Matlab/direction_adaptive_feature.mat')['fileFeatures'][0, :]


###### feature concatenation ######
# joint context
context = 2

joint_contexted = [None] * len(joint)
joint_contexted = np.array(joint_contexted)
for k in range(len(joint)):
#     duplicate adapt feature to have the same number of frames
    adapt[k] = np.repeat(adapt[k], 2, axis=1)
    adapt[k] = adapt[k][:,:joint[k].shape[1]]
    joint_contexted[k] = np.vstack((joint[k], joint[k]))
    for m in range(context,joint[k].shape[1]-context):  # mean and std of 5 frames to take account context information for PETs
        joint_contexted[k][0:joint[k].shape[0],m] = np.mean(joint[k][:,m-context:m+context+1], axis=1)
        joint_contexted[k][joint[k].shape[0]:, m] = np.std(joint[k][:,m-context:m+context+1], axis=1)
    joint_contexted[k] = np.vstack((adapt[k], joint_contexted[k]))

feature = joint_contexted
del(adapt, joint, joint_contexted)
print(feature.shape, feature[21].shape)

###### load annotations ######
# prepare annotation from .csvs
tech_name = np.array(['Tremolo', 'Acciacatura', 'Glissando', 'Trill', 'FT', 'Vibrato', 'Portamento'])
anno_files = [None]

for k in range(len(wav_files)):
    if wav_files[k].split('/')[1] == 'Iso':
        anno_files.append(wav_files[k].replace('.wav', '.csv'))
    elif wav_files[k].split('/')[1] == 'Piece': # 'Piece'
        for m in range(len(tech_name)):
            if os.path.exists(data_root + wav_files[k][:-4]+ '_tech_' + tech_name[m] + '.csv'):
                anno_files.append(wav_files[k][:-4]+ '_tech_' + tech_name[m] + '.csv')
                
anno_files = anno_files[1:]

feature_conca = np.zeros((feature[0].shape[0],1))
file_id = 0
player_id = 0

for k in range(len(feature)):
    # connect all features
    feature_conca = np.hstack((feature_conca, feature[k]))
    # gte file ID
    file_id = np.hstack((file_id, np.ones((feature[k].shape[1]), dtype=int) * k))
    # get player ID
    if wav_files[k].split('/')[0][-1] == '0':
        player_id = np.hstack((player_id, np.ones((feature[k].shape[1]), dtype=int) * 10)) 
    else:
        player_id = np.hstack((player_id, np.ones((feature[k].shape[1]), dtype=int) * int(wav_files[k].split('/')[0][-1]))) 
        
player_id = player_id[1:]
file_id = file_id[1:]
feature_conca = np.transpose(feature_conca)
feature_conca = feature_conca[1:]

# scattering params
sr = 44100
T = 2**14  # PMT T=15, PET T=14 => PMT duplicated
oversampling = 2
hop_sample = T/(2**oversampling)
print('frame size: %sms' % (int(hop_sample/44100*1000)))

label_id = {k:[None] for k in range(len(feature))}

# get label ID
for k in range(len(feature)):
    label_id[k] = np.zeros((len(tech_name), feature[k].shape[1]),dtype=int)
    if wav_files[k].split('/')[1] == 'Iso':
        anno_files = wav_files[k].replace('.wav', '.csv')
        file_anno = pd.read_csv(data_root + anno_files)
        file_onoff = np.hstack((float(list(file_anno)[0]), file_anno[list(file_anno)[0]]))
        label_pos = np.where(tech_name == re.search('Iso_(.*).csv', anno_files).group(1))[0] + 1
        for n in range(len(file_onoff)//2):
            start_idx = int(file_onoff[2*n] * sr / hop_sample)  # use PET's hop_sample,alreay considered the feature duplication
            end_idx = int(file_onoff[2*n+1] * sr / hop_sample)
            if label_pos:
                label_id[k][label_pos-1, start_idx:end_idx] = np.ones((end_idx-start_idx), dtype=int) * (label_pos) # label position in tech_name[m] array
    elif wav_files[k].split('/')[1] == 'Piece': # 'Piece'
        for m in range(len(tech_name)):
            if os.path.exists(data_root + wav_files[k][:-4]+ '_tech_' + tech_name[m] + '.csv'):
                anno_files = (wav_files[k][:-4]+ '_tech_' + tech_name[m] + '.csv')
                file_anno = pd.read_csv(data_root + anno_files)
                file_onoff = np.hstack((float(list(file_anno)[0]), file_anno[list(file_anno)[0]]))
                for n in range(len(file_onoff)//2):
                    start_idx = int(file_onoff[2*n] * sr / hop_sample)
                    end_idx = int(file_onoff[2*n+1] * sr / hop_sample)
                    label_id[k][m, start_idx:end_idx] = np.ones((end_idx-start_idx), dtype=int) * (m+1)


# use single-labeled part only
label_all = 0
for k in range(len(label_id)):
    for m in range(label_id[k].shape[1]):  # no. time frame
        if collections.Counter(label_id[k][:,m])[0] < 6:   # only one have label (counter=6)
            label_id[k][:,m] = np.ones((len(tech_name)),dtype=int) * 100
    label_all = np.hstack((label_all, np.sum(label_id[k],axis=0)))
    
label_id = label_all[1:]
del(label_all)

player_id = np.delete(player_id, np.where(label_id==700), 0)
feature_conca = np.delete(feature_conca, np.where(label_id==700), 0)
file_id = np.delete(file_id, np.where(label_id==700), 0)
label_id = np.delete(label_id, np.where(label_id==700), 0)

###### Playing Technique Recognition ######
# classifier-SVM settings
kernel = 'rbf'; gpu_id = 0
# param_grid = {'C': [10], 'gamma': [.0001]}   # param_grid for toy experiment
param_grid = {'C': [256, 128, 64, 32, 16, 8], 'gamma': [2**(-12),2**(-11),2**(-10),2**(-9),2**(-8),2**(-7)]} # para_grid used
scoring = 'f1_macro'; cv = 3

# data split according to players + cross validation
torch.manual_seed(42)
player_split = torch.randperm(len(np.unique(player_id))) + 1
player_split = player_split.numpy()

# trainSplit testSplit player
trainSplit = {k:[] for k in range(5)}; testSplit = {k:[] for k in range(5)}

trainSplit[0] = player_split[0:int(player_split.shape[0]*.8)]    # seg idx for trainSplit
testSplit[0] = player_split[int(player_split.shape[0]*.8):player_split.shape[0]]   # seg idx for testSplit

trainSplit[1] = player_split[2:10]    # seg idx for trainSplit
testSplit[1] = player_split[0:2]   # seg idx for testSplit

trainSplit[2] = np.hstack((player_split[4:10],player_split[0:2]))   # seg idx for trainSplit
testSplit[2] = player_split[2:4]   # seg idx for testSplit

trainSplit[3] = np.hstack((player_split[6:10],player_split[0:4]))  # seg idx for trainSplit
testSplit[3] = player_split[4:6]   # seg idx for testSplit

trainSplit[4] = np.hstack((player_split[8:10],player_split[0:6]))   # seg idx for trainSplit
testSplit[4] = player_split[6:8]   # seg idx for testSplit

# record PRF and confusion obtained at each split
PRF = {split:np.zeros((len(tech_name)+1,3)) for split in range(5)} # including "other" class which is 0
confusion = {split:np.zeros((len(tech_name)+1, len(tech_name)+1)) for split in range(5)} 

# five splits for the whole dataset
# classification for each split
t0 = time.time()
for split in tqdm(range(5)):

    subset = np.ones((len(player_id)), dtype=int) * 100

    for k in range(len(player_id)):
        if player_id[k] in trainSplit[split]:
            subset[k] = 0
        else: # test
            subset[k] = 1

    feature_tr, label_tr = feature_conca[subset == 0], label_id[subset == 0]
    feature_te, label_te = feature_conca[subset == 1], label_id[subset == 1]

    #########################  imputation  ###############################
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    feature_tr = imp.fit_transform(feature_tr)
    feature_te = imp.transform(feature_te)

    #########################  normalisation  ###############################
    stdscaler = StandardScaler()
    feature_tr = stdscaler.fit_transform(feature_tr)
    feature_te = stdscaler.transform(feature_te)
    print(feature_tr.shape, feature_te.shape)

    #########################  classification  ###############################
    clf =  GridSearchCV(SVC(kernel=kernel, gpu_id=gpu_id), param_grid=param_grid, cv=cv, scoring=scoring)
    clf = clf.fit(feature_tr, label_tr)
    label_pred = clf.predict(feature_te)
    print('Result of split %d :' % split)
    print(classification_report(label_te, label_pred))
    print(confusion_matrix(label_te, label_pred))
    
    #########################  record result of each split  ###############################
    # extract P,R,F values from classification_report
    lineSep = 54 ; dist = 10; Pos_firstNum = classification_report(label_te, label_pred).find('\n') + 21 
    for k in range(len(tech_name)+1):
        PRF[split][k,:] = np.array([
        float(classification_report(label_te, label_pred)[Pos_firstNum+lineSep*k:Pos_firstNum+4+lineSep*k]),\
        float(classification_report(label_te, label_pred)[Pos_firstNum+dist*1+lineSep*k:Pos_firstNum+4+dist*1+lineSep*k]),\
        float(classification_report(label_te, label_pred)[Pos_firstNum+dist*2+lineSep*k:Pos_firstNum+4+dist*2+lineSep*k])])
    confusion[split] = confusion_matrix(label_te, label_pred)

np.savez('CBFdataset_PRF_confusion.npz', PRF, confusion)

# average result across splits
PRF_aver = np.mean(np.array([PRF[k] for k in range(5)]), 0)
print('F-measure for each type of playing technique: ')
print((PRF_aver[:,2]))
print('Marco F-measure: %.3f'%np.mean(PRF_aver[:,2]))

confusion_sum = np.sum(np.array([confusion[k] for k in range(5)]), 0)
print('Confusion matrix on the CBFdataset:')
print(confusion_sum)

# confusion matrix
A = confusion_sum
B = np.zeros((confusion_sum.shape[0]+1, confusion_sum.shape[0]+1), dtype=int)
B[:-1, :-1] = A
B[-1, :] = B [0, :]; B[:, -1] = B [:, 0]
B = B[1:, 1:]
confusion = B

tech_name = ['tremolo', 'acciaccatura', 'glissando', 'trill', 'flutter-tongue', 'vibrato', 'portamento']
tech_name.append('other')
tech_name = np.array(tech_name)

norm_confusion = confusion.T / confusion.astype(np.float).sum(axis=1)
norm_confusion = norm_confusion.T

################################# without adapt duplicate & hopsample/2 because of multiplier in joint cal ####################
# use seaborn plotting defaults
import seaborn as sns; sns.set()
from matplotlib import rcParams

plt.figure(figsize=(16,6))
plt.subplot(121)
sns.heatmap(confusion, cmap = "Blues", square=True, annot=True, fmt="d",
            xticklabels=tech_name, yticklabels=tech_name)
plt.xticks([0,1,2,3,4,5,6,7], tech_name, rotation=60, fontsize=11.5); plt.yticks(fontsize=11.5)
plt.ylabel('True label', fontsize=12); plt.xlabel('Predicted label', fontsize=12)
plt.ylim([8,0])
plt.title('(a) Confusion', fontsize=13)
rcParams['axes.titlepad'] = 15

plt.subplot(122)
norm_confusion = np.round(norm_confusion,2)
sns.heatmap(norm_confusion, cmap = "Blues", square=True, annot=True,
            xticklabels=tech_name, yticklabels=tech_name) # cbar=False,
plt.xticks([0,1,2,3,4,5,6,7], tech_name, rotation=60, fontsize=11.5); plt.yticks(fontsize=11.5)
plt.ylabel('True label', fontsize=12); plt.xlabel('Predicted label', fontsize=12)
plt.ylim([8,0])
plt.title('(b) Normalised confusion', fontsize=13)
rcParams['axes.titlepad'] = 15

plt.tight_layout()
plt.savefig('confusion_matrix.png')

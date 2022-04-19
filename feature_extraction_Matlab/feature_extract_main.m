% Extract the frequency-adaptive scattering features
clear all; clc; tic
dataset_directory = '/import/c4dm-datasets/ChineseBambooFluteDataset/CBF-multilabel-binaryClassf/';
addpath(genpath(dataset_directory))
addpath(genpath('../../AdaptiveScattering/'))
run 'addpath_scatnet.m' % ScatNet 

fid=fopen('file_names.txt'); 
tline = fgetl(fid);
file_names = []; k=1;
while ischar(tline)
    file_names{k} = tline; 
    k = k+1;
    tline = fgetl(fid);
end
fclose(fid);

frequency_adaptive_feature_extraction(file_names)
direction_adaptive_feature_extraction(file_names)
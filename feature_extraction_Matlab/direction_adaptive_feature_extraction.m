% Extract the direction-adaptive scattering features
function direction_adaptive_feature_extraction(file_names)

%% parameter
T = 2^14; 
time_filt_opt.Q = [16 2];
timeModuRateMax = 50; % Hz

time_filt_opt.filter_type = {'morlet_1d','morlet_1d'};
time_filt_opt.J = T_to_J([T T],time_filt_opt);
time_scat_opt.M = 2;
time_scat_opt.oversampling = 2;

freq_filt_opt.Q = 2;
freq_filt_opt.filter_type = {'morlet_1d'};
freq_scat_opt.M = 1;

%% extract features for each file
fileFeatures = [];
for k=1:length(file_names)
    [k, length(file_names)]
    [x,fs] = audioread(file_names{k}); x = mean(x,2); 
    [Wop, filters] = joint_tf_wavelet_factory_1d(length(x),time_filt_opt,freq_filt_opt,time_scat_opt);
    S = log_scat(joint_renorm_scat(scat(x, Wop)));

    moduTime_center = filters{1, 2}.psi.meta.center/3.14/2*fs;
    [val, posT2nd] = find(moduTime_center <= timeModuRateMax);
    temp_idx = arrayfun( @(x)( find(posT2nd==x) ), S{1, 3}.meta.j(end,:),...
        'UniformOutput', false);
    moduTime_idx = find(~cellfun(@isempty,temp_idx));
    feature.signal = S{1, 3}.signal(moduTime_idx);
    feature.meta.j = S{1, 3}.meta.j(:,moduTime_idx);
    feature.meta.fr_j = S{1, 3}.meta.fr_j(:,moduTime_idx);
    jointScatFeature = [feature.signal{:}].';
  
    [val_theta1, loc_theta1] = find(feature.meta.fr_j<=0);

    [val_theta2, loc_theta2] = find(feature.meta.fr_j>0);

    dimRedu_theta1Down = jointScatFeature(loc_theta1,:);
    dimRedu_theta2Up = jointScatFeature(loc_theta2,:);

    minDim = min(size(dimRedu_theta1Down,1),size(dimRedu_theta2Up,1));
    frameFeature = zeros(minDim,size(dimRedu_theta1Down,2));
    for kk=1:size(dimRedu_theta1Down,2)
       frameFeature(:,kk) = (dimRedu_theta1Down(end-minDim+1:end,kk) + ...
           dimRedu_theta2Up(end-minDim+1:end,kk))/2;
    end
    
    fileFeatures{k} = frameFeature;
       
    clear Wop filters jointScatFeature feature S x frameFeature minDim dimRedu_theta2Up dimRedu_theta1Down...
         loc_theta1 loc_theta2
end
cal_time = toc
save('direction_adaptive_feature.mat','fileFeatures', 'cal_time');
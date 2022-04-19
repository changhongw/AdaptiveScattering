% Extract the frequency-adaptive scattering features
function frequency_adaptive_feature_extraction(file_names)

%% parametres
T = 2^15;   
Q = [16, 4];
minModuRate = 0; maxModuRate = 100; % in Hz
Nbands = 7; % total number of frequency bands decomposed

tm_filt_opt.filter_type = 'morlet_1d';
tm_filt_opt.Q = Q(1); J_temp = T_to_J(T,tm_filt_opt);
tm_filt_opt.Q = Q(2); J_adapt = T_to_J(T,tm_filt_opt);
J = [J_temp, J_adapt];

tm_scat_opt.M = 1;
tm_scat_opt.oversampling = 2;
freq_filt_opt.filter_type = 'morlet_1d';
freq_filt_opt.Q = 1;
freq_scat_opt.M = 1;

adapt_options.T = T;
adapt_options.oversampling = tm_scat_opt.oversampling;
adapt_options.Nbands = Nbands;

fileFeatures_time = []; fileFeatures_timerate = [];
for k=1:length(file_names)
    [k length(file_names)]
    [x,fs] = audioread(file_names{k});
    x = mean(x,2);
    
    %% 1st-order temp scat
    tm_filt_opt.Q = Q(1); tm_filt_opt.J = J(1);
    [Wop_tm, filters] = wavelet_factory_1d(length(x), tm_filt_opt, tm_scat_opt);
    acoustic_freqcenter = round(filters{1, 1}.psi.meta.center/3.14/2*44100);
    [S, U] = scat(x, Wop_tm);  % S2 is normalised on S1, keep S1
    clear Wop_tm filters
    
    %% adapt scat
    tm_filt_opt.Q = Q(2); tm_filt_opt.J = J(2);
    [Wop_tm, filters] = wavelet_factory_1d_adapt(T, tm_filt_opt, tm_scat_opt);
    [val, pos] = find(U{1, 2}.meta.bandwidth < min(filters{1, 1}.psi.meta.center));
    maxDecmpIdx = length(U{1, 2}.meta.bandwidth) - length(pos); % note that here is len
    adapt_options.maxDecmpIdx = maxDecmpIdx;

    %% specific modulation rate range
    modulation_freqcenter = round(filters{1, 1}.psi.meta.center/3.14/2*44100);
    modulation_freqcenter(modulation_freqcenter>maxModuRate) = [];
    moduIdx_high = length(round(filters{1, 1}.psi.meta.center)) - ... 
                    length(modulation_freqcenter)+1;  % start from high freq but low idx
    modulation_freqcenter(modulation_freqcenter<minModuRate) = []; 
    moduIdx_low = moduIdx_high + length(modulation_freqcenter) -1; 
    adapt_options.moduIdx = [moduIdx_high, moduIdx_low]; % e.g. [high, low]=[63, 87]
    clear filters
    
    %% decomposition trajectory = cloest to 
    firstOrderCoeff = [S{2}.signal{:}].';
    for ii=1:size(firstOrderCoeff,2)  % get the dominant band trajectory
        [val, domIdx(ii)] = max(firstOrderCoeff(:,ii));
        domIdx(ii) = max(domIdx(ii));  % euql => low freq one
    end

    %% expanded bands with freq scat
    moduScaleNumKept = moduIdx_low-moduIdx_high+1;  
    Nfreq = 2^nextpow2(moduScaleNumKept);   % here is different
    freq_filt_opt.J = T_to_J(Nfreq,freq_filt_opt);
    Wop_fr = wavelet_factory_1d_adapt(Nfreq, freq_filt_opt, freq_scat_opt);
    adapt_options.freqScaleNum = freq_filt_opt.J;
    adapt_options.fr_oversampling = 1;
%     
    S_adapt_time = NaN*ones(moduScaleNumKept*Nbands, length(domIdx)); 
    S_adapt_timerate = [];
    for jj=1:Nbands
        adapt_options.domIdx = domIdx+(Nbands-1)/2+1-jj;
        S_adapt_time((jj-1)*moduScaleNumKept+1:jj*moduScaleNumKept,:) = ... 
            adapt_time_scat(U{2},S,Wop_tm, adapt_options);
        S_adapt_timerate = [S_adapt_timerate; ...
            freq_scat_SQ(S_adapt_time((jj-1)*moduScaleNumKept+1:jj*... 
            moduScaleNumKept,:), Wop_fr)]; 
    end
    
    fileFeatures_time{k} = S_adapt_time;
    fileFeatures_timerate{k} =  S_adapt_timerate;
    clear S_adapt_time S_adapt_timerate Wop_tm Wop_fr S U domIdx ii jj ...
        firstOrderCoeff filters modulation_freqcenter acoustic_freqcenter InterTemp

end
cal_time = toc
save('frequency_adaptive_feature.mat','fileFeatures_time', 'fileFeatures_timerate','cal_time');
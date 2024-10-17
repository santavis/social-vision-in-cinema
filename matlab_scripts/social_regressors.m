%%Create high-level social perceptual regressors
%
%   1.   Read and organize raw rating data from Onni reports to better
%        format (for Experiments 2 & 3) and from preprocessed Gorilla data for
%        Experiment 3
%
%   2.   Create average regressors for eye-tracking analysis and
%        upsample to 1ms
%
% Severi Santavirta 18th of August 2023

%% 1. Read and organize raw rating data from Onni reports to better format and calculate rating averages for each time point, Exepriment 1

if(~exist('path/socialdata/localizer/onnidata_processed_fmri/onnidata_processed_12.csv','file'))
    f = find_files('path/socialdata/localizer/onnidata_raw','*.csv');
    feature_order = table2array(readtable('path/socialdata/localizer/onni_feature_order.xlsx','ReadVariableNames',false));

    % Timings in Onni (copied from Tomi's old script)
    video_clip_name_onni = {'1a3a';'1a3b';'1a3c';'1a5a';'1a5b';'1a11a';'1a11b';'1a11c';'1a13a';'1a13b';'1a13c';'1a20a';'1a20b';'1a20c';'1b2a';'1b2b';'1b2c';'1b3a';'1b3b';'1b7a';'1b7b';'1b8a';'1b8b';'1b9a';'1b9b';'1b10a';'1b10b';'1b18a';'2a2a';'2a2b';'2a3a';'2a3b';'2a3c';'2a7a';'2a7b';'2a19a';'2a19b';'2b1a';'2b3a';'2b4a';'2b4b';'2b11a';'2b11b';'2b12a';'2b12b';'2b13a';'2b19a';'2b19b';'3a2a';'3a10a';'3a10b';'3a10c';'3a17a';'3a17b';'3a23a';'3a23b';'3a24a';'3a24b';'3a25a';'3a25b';'5a5a';'5a5b';'5a6a';'5a6b';'5a6c';'5a6d';'5a8a';'5a8b';'5a8c';'5a11a';'5a11b';'5a11c';'5a13a';'5a13b';'5a13c';'5a17a';'5a17b';'5a17c';'5a19a';'5a19b';'5a19c';'5a20a';'5a20b';'5a20c';'6a1a';'6a1b';'6a1c';'6a2a';'6a2b';'6a2c';'6a4a';'6a4b';'6a4c';'6a4d';'6a5a';'6a5b';'6a5c';'6a5d';'6a6a';'6a6b';'6a6c';'6a10a';'6a10b';'6a10c';'6a10d';'6a12a';'6a12b';'6a12c';'6a12d';'6a12e';'6a13a';'6a13b';'6a14a';'6a14b';'6a15a';'6a15b';'6a15c';'6a17a';'6a17b';'6a17c';'6a17d';'6a18a';'6a18b';'7a6a';'7a6b';'7a6c';'7a7a';'7a7b';'7a8a';'7a8b';'7a8c';'7a8d';'7a9a';'7a9b';'7a12a';'7a12b';'7a14a';'7a14b';'7a14c';'7a15a';'7a15b';'7a15c';'7a16a';'7a16b';'7a16c';'7a17a';'7a17b';'7a17c';'7a18a';'7a18b';'7b1a';'7b1b';'7b2a';'7b2b';'7b6a';'7b6b';'7b6c';'7b7a';'7b7b';'7b8a';'7b8b';'7b9a';'7b9b';'7b9c';'7b10a';'7b10b';'7b12a';'7b12b';'7b13a';'7b13b';'7b17a';'7b17b';'8a1a';'8a1b';'8a1c';'8a1d';'8a3a';'8a3b';'8a4a';'8a4b';'8a4c';'8a7a';'8a7b';'8a9a';'8a9b';'8a10a';'8a10b';'8a12a';'8a12b';'8a12c';'8a15a';'8a15b';'8a15c';'8a16a';'8a16b';'8a16c';'9a1a';'9a1b';'9a1c';'9a2a';'9a2b';'9a2c';'9a3a';'9a3b';'9a3c';'9a3d';'9a3e';'9a4a';'9a4b';'9a4c';'9a4d';'9a5a';'9a5b';'9a5c';'9a6a';'9a6b';'9a6c';'9a6d';'9a6e';'9a6f';'9a6g';'9a8a';'9a8b';'9a8c';'9a9a';'9a9b';'9a9c';'9a9d';'9a10a';'9a10b';'9a10c';'9a10d';'9a10e';'9a10f';'10a1a';'10a1b';'10a2a';'10a2b';'10a3a';'10a3b';'10a4a';'10a4b';'10a5a';'10a5b';'10a6a';'10a6b';'10a6c';'10a6d';'10a7a';'10a7b';'10a7c';'10a7d';'10a8a';'10a8b';'10a8c';'10a8d'};
    video_clip_name_onni = cellfun(@(x) x(1:end-1),video_clip_name_onni,'UniformOutput',false);
    video_dur_onni = [4;4;3.20000000000000;4;5.80000000000000;4;4;4.40000000000000;4;4;3.10000000000000;4;4;3.80000000000000;4;4;3.70000000000000;4;6;4;6.90000000000000;4;6.80000000000000;4;7;4;6.10000000000000;6.60000000000000;4;5.90000000000000;4;4;6.80000000000000;4;6.30000000000000;4;5.40000000000000;5.30000000000000;7.20000000000000;4;4.40000000000000;4;6.40000000000000;4;5.20000000000000;6.70000000000000;4;5.40000000000000;7.10000000000000;4;4;6.40000000000000;4;6.30000000000000;4;6.10000000000000;4;6.30000000000000;4;7.30000000000000;4;7.20000000000000;4;4;4;4.70000000000000;4;4;3.70000000000000;4;4;7.20000000000000;4;4;5.80000000000000;4;4;4;4;4;4.20000000000000;4;4;3.60000000000000;4;4;6;4;4;6.70000000000000;4;4;4;4.30000000000000;4;4;4;4.10000000000000;4;4;4.20000000000000;4;4;4;3.70000000000000;4;4;4;4;6.50000000000000;4;6.10000000000000;4;7.10000000000000;4;4;4;4;4;4;4.30000000000000;4;6.30000000000000;4;4;4;4;6.40000000000000;4;4;4;6.40000000000000;4;6.60000000000000;4;7.20000000000000;4;4;6.90000000000000;4;4;3.20000000000000;4;4;6.30000000000000;4;4;4.10000000000000;4;6.80000000000000;4;6;4;6.70000000000000;4;4;3.70000000000000;4;7;4;7.10000000000000;4;4;3.90000000000000;4;7.10000000000000;4;7;4;6.80000000000000;4;6.40000000000000;4;3.90000000000000;4.10000000000000;6;4;6.20000000000000;4;4;7;4;7.10000000000000;4;5.20000000000000;4;6.80000000000000;4;4;4.80000000000000;4;4;5.70000000000000;4;4;3.80000000000000;4;4;5.20000000000000;4;4;3.90000000000000;4;4;4;4;5.20000000000000;4;4;4;3.70000000000000;4;4;6.50000000000000;4;4;4;4;4;4;4.20000000000000;4;4;4.70000000000000;4;4;4;5.50000000000000;4;4;4;4;4;5;4;4.30000000000000;4;6.30000000000000;4;6;4;7;4;6.90000000000000;4;4;4;4.50000000000000;4;4;4;4.90000000000000;4;4;4;6.90000000000000];

    % Create data table
    for I = 1:size(f,1)

        % Load rating data
        tbl = readtable(f{I});
        raw = table2array(tbl(:,3:end))';

        % Select the rated features
        feats = feature_order(~strcmp(feature_order(:,I),'nan'),I);
        if(size(raw,1)/size(feats,1)==256) % Number of data points
            feats_column = repmat(feats,size(raw,1)/size(feats,1),1);
            timepoint_idx = (repelem(1:size(raw,1)/size(feats,1),size(feats,1)))';
        else
            error('Investigate');
        end

        % Add the video names
        video_column = repelem(video_clip_name_onni,size(feats,1));

        % Add the rating time point information
        ratingtime_column = repelem(video_dur_onni,size(feats,1));
        
        % IMPORTANT! Check manually that all the subjects data columns are
        % valid before calculating the averages (CHECKED)
        if(I == 10 || I == 12) % 1-2 nan columns
            raw = raw(:,1:5);
        end
        
        % Add new colums to raw data table
        data = array2table(raw);

        data = horzcat(data,array2table(mean(raw,2,'omitnan')),array2table(feats_column),array2table(timepoint_idx),array2table(video_column),array2table(ratingtime_column));
        if(size(data,2)==12)
            data.Properties.VariableNames = {'r1','r2','r3','r4','r5','r6','r7','average_rating','feature','timepoint_index','video','rating_time'};
        elseif(size(data,2)==11)
            data.Properties.VariableNames = {'r1','r2','r3','r4','r5','r6','average_rating','feature','timepoint_index','video','rating_time'};
        elseif(size(data,2)==10)
            data.Properties.VariableNames = {'r1','r2','r3','r4','r5','average_rating','feature','timepoint_index','video','rating_time'};
        else
            error('Investigate');
        end
        writetable(data,sprintf('path/socialdata/localizer/onnidata_processed_fmri/onnidata_processed_%02d.csv',I));
    end
end

%% 2. Extract the data points that are presented in the megatrack localizer experiment (Exepriment 1) (only a subset of the rated videos)

if(~exist('path/socialdata/localizer/onnidata_processed_eyetracking/onnidata_processed_12.csv','file'))

    % List stimulus videos
    f = find_files('path/stimulus/localizer/localizer_eyetracking','*.mp4');
    [~,fname,~] = fileparts(f);
    tmp = cellfun(@strsplit,fname,cellstr(repmat('_',size(fname,1),1)),'UniformOutput',false);
    tmp = vertcat(tmp{:});
    vidname = tmp(:,2);
    vidrder = (1:size(vidname,1))';

    % Read the onni data and select rows of the eye tracking videos
    f = find_files('path/socialdata/localizer/onnidata_processed_fmri','*.csv');
    for I = 1:size(f,1)

        % Load rating data
        tbl = readtable(f{I});

        % Find the rows that contain the eye tracking videos
        idx= find(ismember(tbl.video, vidname));
        tbl_eyetracking = tbl(idx,:);

        writetable(tbl_eyetracking,sprintf('path/socialdata/localizer/onnidata_processed_eyetracking/onnidata_processed_%02d.csv',I));
    end
end

%% 3. Create regressors for eye-tracking analysis by extracking feature wise average data and sorting to the eye tracking presentation order, Experiment 1

f = find_files('path/socialdata/localizer/onnidata_processed_eyetracking','*.csv');

% Read eye tracking video order for sorting
fv = find_files('path/stimulus/localizer/localizer_eyetracking','*.mp4');
[~,fname,~] = fileparts(fv);
tmp = cellfun(@strsplit,fname,cellstr(repmat('_',size(fname,1),1)),'UniformOutput',false);
tmp = vertcat(tmp{:});
vidname = tmp(:,2);

% Read eyedata to get the corrected video durations for each clip.
load('path/eyedata/subdata/localizer/subjects/L001.mat');
counts = hist(subdata.trial_indices,unique(subdata.trial_indices));

regressors = zeros(189,126); % Hard-coded
time = [];
trial = [];
features = {};
n=0;
for I = 1:size(f,1)

    % Load rating data
    tbl = readtable(f{I});
    
    % Create regressor for each social feature
    feats = unique(tbl.feature);
    for J = 1:size(feats,1)
        feat_data = tbl(cellfun(@strcmp,tbl.feature,cellstr(repmat(feats{J},size(tbl,1),1))),:);
        
        % Sort to eye tracking presentation order and correct the timings
        % by chosing the last rating time point from the corrected lenght of the
        % eyetracking data
        feat_data_sorted = table;
        for K = 1:size(vidname)
            viddata = feat_data(cellfun(@strcmp,feat_data.video,cellstr(repmat(vidname{K},size(feat_data,1),1))),:);
            if(J == 1 && I == 1)
                time_vid = viddata.rating_time*1000;
                if(sum(time_vid)~=counts(K)) % The lenght read from the eyetracking data do not match with the original hard-coded rating timepoints in Onni
                    tmp = sum(time_vid(1:(end-1))); % Calculate the starting time of the last rating interval
                    time_vid(end) = counts(K)-tmp; % Change the last time point to the corrected end point of the video based on eyetracking info
                end
                time = vertcat(time,time_vid);
                trial = vertcat(trial,repmat(K,size(viddata,1),1));
            end
            
            feat_data_sorted = vertcat(feat_data_sorted,viddata);
        end
        n = n+1;
        regressors(:,n) = feat_data_sorted.average_rating;
        
    end
    features = vertcat(features,feats);
end

% Save regressor table
regressors = array2table(regressors);
regressors.Properties.VariableNames = features;

% Make a "nonsocial" control regressor by first combining human
% voice, number of people and animals and then taking timepoints
% where all of these are zero
idx = [1,4,24];
comb = sum(table2array(regressors(:,idx)),2);
nonsocial = zeros(size(regressors,1),1);
nonsocial(comb==0) = 1;
nonsocial = array2table(nonsocial);
regressors = horzcat(regressors,nonsocial);

regressors.timepoint_index  = table2array(feat_data_sorted(:,8));
regressors.video = table2array(feat_data_sorted(:,9));
regressors.trial = trial; 
regressors.time  = cumsum(time);
regressors.time_video = time;
writetable(regressors,'path/socialdata/localizer/localizer_eyetracking_social_regressors.csv');

%% 4. Upsample to 1ms regressors, Experiment 1

regressors = readtable('path/socialdata/localizer/localizer_eyetracking_social_regressors.csv');

% In the analyses, we only use 8 social features
regressors = regressors(:,[31,36,72,82,99,101,102,110,130:132]);

for row = 1:size(regressors,1)
    fprintf('%d/%d\n',row,size(regressors,1));
    r = regressors(row,:);
    
    t = r.time_video;
    
    row_1ms = repelem(r,t,1);
    row_1ms.time = (1:t)';
    if(row==1)
        regressors_1ms = row_1ms;
    else
        regressors_1ms = vertcat(regressors_1ms,row_1ms);
    end
end

% Sort predictors to alphabetical order
predictors = regressors_1ms(:,1:8);
timings = regressors_1ms(:,9:end);
sortedNames = sort(predictors.Properties.VariableNames);
predictors = predictors(:,sortedNames);
regressors_1ms = horzcat(predictors,timings);
save('path/socialdata/localizer/localizer_eyetracking_social_regressors_1ms.mat','regressors_1ms');

clear; clc;

%% 1. Read and organize raw rating data from Onni reports to better format and calculate rating averages for each time point, Experiment 2

if(~exist('path/socialdata/kasky/onnidata_processed/onnidata_processed_a_12.csv','file'))
    f_a = find_files('path/socialdata/kasky/onnidata_raw','*_a_');
    f_b = find_files('path/socialdata/kasky/onnidata_raw','*_b_');

    feature_order = table2array(readtable('path/socialdata/kasky/onni_feature_order.xlsx','ReadVariableNames',false));

    % The ratings have been collected in 8 sec intervals. First time window
    % for A (N=527) is 0-8 and for B is 0-4 (N=528).
    
    % Read the subject data to get the real stimulus length 
    load('path/eyedata/subdata/kasky/subjects/K01.mat');
    tend = size(subdata.pupil,1);
    t_a = (8000:8000:tend)'; t_a = vertcat(t_a,tend);
    t_b = (4000:8000:tend)'; t_b = vertcat(t_b,tend);

    % Create data table
    for I = 1:size(f_a,1)

        % Load rating data
        tbl_a = readtable(f_a{I});
        tbl_b = readtable(f_b{I});
        raw_a = table2array(tbl_a(:,3:end))';
        raw_b = table2array(tbl_b(:,3:end))';

        % Select the rated features
        feats = feature_order(~strcmp(feature_order(:,I),'nan'),I);
        if(size(raw_a,1)/size(feats,1)==527) % Number of data points
            feats_column_a = repmat(feats,size(raw_a,1)/size(feats,1),1);
            timepoint_idx_a = (repelem(1:size(raw_a,1)/size(feats,1),size(feats,1)))';
        else
            error('Investigate');
        end
        if(size(raw_b,1)/size(feats,1)==528) % Number of data points
            feats_column_b = repmat(feats,size(raw_b,1)/size(feats,1),1);
            timepoint_idx_b = (repelem(1:size(raw_b,1)/size(feats,1),size(feats,1)))';
        else
            error('Investigate');
        end
        
    
        % Add the rating time point information
        ratingtime_column_a = repelem(t_a,size(feats,1));
        ratingtime_column_b = repelem(t_b,size(feats,1));
        
        % IMPORTANT! Check manually that all the subjects data columns are
        % valid before calculating the averages (CHECKED)
        
        if(I==1)
            raw_a = raw_a(:,2:end);
        end

        % Add new colums to raw data table
        data_a = array2table(raw_a);
        data_a = horzcat(data_a,array2table(mean(raw_a,2,'omitnan')),array2table(feats_column_a),array2table(timepoint_idx_a),array2table(ratingtime_column_a));
        data_b = array2table(raw_b);
        data_b = horzcat(data_b,array2table(mean(raw_b,2,'omitnan')),array2table(feats_column_b),array2table(timepoint_idx_b),array2table(ratingtime_column_b));
        
        data_a.Properties.VariableNames = {'r1','r2','r3','average_rating','feature','timepoint_index','rating_time'};
        data_b.Properties.VariableNames = {'r1','r2','r3','average_rating','feature','timepoint_index','rating_time'};

        writetable(data_a,sprintf('path/socialdata/kasky/onnidata_processed/onnidata_processed_a_%02d.csv',I));
        writetable(data_b,sprintf('path/socialdata/kasky/onnidata_processed/onnidata_processed_b_%02d.csv',I));

    end
end

%% 2. Create regressors for eye-tracking analysis by extracking feature wise average data and combining interleaved A & B, Experiment 2

f_a = find_files('path/socialdata/kasky/onnidata_processed','*_a_');
f_b = find_files('path/socialdata/kasky/onnidata_processed','*_b_');

regressors = zeros(1054,126); % Hard-coded
features = {};
n=0;
for I = 1:size(f_a,1)

    % Load rating data
    tbl_a = readtable(f_a{I});
    tbl_b = readtable(f_b{I});
    
    % Create regressor for each social feature
    feats = unique(tbl_a.feature);
    for J = 1:size(feats,1)
        
        % Extract feature data
        feat_data_a = tbl_a(cellfun(@strcmp,tbl_a.feature,cellstr(repmat(feats{J},size(tbl_a,1),1))),:);
        feat_data_b = tbl_b(cellfun(@strcmp,tbl_b.feature,cellstr(repmat(feats{J},size(tbl_b,1),1))),:);
        
        % Interpolate data to 1second for easy combining of A & B
        t_int = (1000:1000:floor(feat_data_a.rating_time(end)))';
        t_int(end) = feat_data_a.rating_time(end);
        a_int = interp1(feat_data_a.rating_time,table2array(feat_data_a(:,1:3)),t_int,'next');
        b_int = interp1(feat_data_b.rating_time,table2array(feat_data_b(:,1:3)),t_int,'next');
        
        % First time-window is still nan
        a_int(1:7,:) = repmat(table2array(feat_data_a(1,1:3)),7,1);
        b_int(1:3,:) = repmat(table2array(feat_data_a(1,1:3)),3,1);
        
        % Combine and take the average
        comb_int = horzcat(a_int,b_int);
        average_int = mean(comb_int,2);
        
        % Downsample back to the rating scale
        t_idx = (4:4:size(t_int,1))';
        t_idx = vertcat(t_idx,size(t_int,1));
        t = t_int(t_idx);
        n = n+1;
        regressors(:,n) = average_int(t_idx);
        
        
    end
    features = vertcat(features,feats);
end

% Save regressor table
regressors = array2table(regressors);
regressors.Properties.VariableNames = features;

% Make a "nonsocial" control regressor by first combining human
% voice, number of people and animals and then taking timepoints
% where all of these are zero
idx = [1,4,24];
comb = sum(table2array(regressors(:,idx)),2);
nonsocial = zeros(size(regressors,1),1);
nonsocial(comb==0) = 1;
nonsocial = array2table(nonsocial);
regressors = horzcat(regressors,nonsocial);

% Add timing information
t = array2table(t);
t.Properties.VariableNames = {'time'};
regressors = horzcat(regressors,t);

% Add eyetracking trial information
load('path/eyedata/subdata/kasky/subjects/K01.mat');
counts = cumsum(hist(subdata.trial_indices,unique(subdata.trial_indices)));
trials = zeros(size(regressors,1),1);
t0 = 1;
for I = 1:size(counts,2)
    idx = max(find(regressors.time<=counts(I))); % Find the last time time window that ends within the trial (the first time window of a new trial starts before the last have ended but end after it)
    trials(t0:idx) = I;
    t0 = idx+1;
end
regressors.trial = trials;

writetable(regressors,'path/socialdata/kasky/kasky_eyetracking_social_regressors.csv');

%% 3. Upsample to 1ms regressors, Experiment 2

regressors = readtable('path/socialdata/kasky/kasky_eyetracking_social_regressors.csv');
load('path/eyedata/subdata/kasky/subjects/K01.mat');
trials = subdata.trial_indices;

% In the analyses, we only use 8 social features
regressors = regressors(:,[31,36,72,82,99,101,102,110,128]);

t0 = 0;
for row = 1:size(regressors,1)
    fprintf('%d/%d\n',row,size(regressors,1));
    r = regressors(row,:);
    
    t = r.time;
    r.time_rating = t;
   
    row_1ms = repelem(r,t-t0,1);
    row_1ms.time = (t0+1:t)';
    
    if(row==1)
        regressors_1ms = row_1ms;
    else
        regressors_1ms = vertcat(regressors_1ms,row_1ms);
    end
    
    t0 = t;
end

% Sort predictors to alphabetical order
predictors = regressors_1ms(:,1:8);
timings = regressors_1ms(:,9:end);
sortedNames = sort(predictors.Properties.VariableNames);
predictors = predictors(:,sortedNames);
regressors_1ms = horzcat(predictors,timings);
regressors_1ms.trial = trials;
save('path/socialdata/kasky/kasky_eyetracking_social_regressors_1ms.mat','regressors_1ms');

clear; clc;

%% 1. Combine interleaved A & B raiting data from preprocessed Gorilla data, Experiment 3

input = 'path/socialdata/conjuring/gorilladata/data/data_reliable';
dims = {'feeling_pleasant','feeling_unpleasant','aroused','aggressive','feeling_pain','moving_their_bodies','playful','talking'}; % Rated dimensions
timing_file = 'path/socialdata/conjuring/gorilladata/conjuring_clip_timings_gorilla.txt';

% In eye-tracking, after one clip (conjuring_19, block 7,
% trial 1) the clip has been cut ~6.5sec before the clip actually ended.
% The next clip still starts at the right time compared to stimulus files.
% For other clips the duration read from the mp4 clips is up to 1.1sec
% longer than reported in the eye-tracker reports. This is because the avi (original presentation file)
% to mp4 conversion (for lowlevel feature extraction) is not accurate and the mp4 files are not cut accurately at the end (checked by comparing visually the mp4 files and the stimulus files in eyelink software). 
% These timing missmatches are corrected so that the timings for the last time wiondows for the social data are read from the eye-tracking clip durations.

% Read the video duration from fixation report and then cut the low-level
% data from the end to match collected eye-tracking data
fixrep = readtable('path/eyedata/raw/conjuring/fixrep_conjuring_preprocessed.txt');
tmp = fixrep.dur;
vid_dur_eyetracking = zeros(30,1); % Hard-coded
for I=1:size(vid_dur_eyetracking,1)
    vid_dur_eyetracking(I) = unique(tmp(strcmp(fixrep.SUBJECT,'C01') & fixrep.TRIAL_INDEX==I)); % Hard-coded
end

% Read the clip timings
tbl = table2array(readtable(timing_file,"ReadVariableNames",false));

block_a = tbl(contains(tbl(:,15),'c2_a'),1);
block_b = tbl(contains(tbl(:,15),'c2_b'),1);

block_a = cellfun(@(x) strsplit(x,' '),block_a,'UniformOutput',false);
block_b = cellfun(@(x) strsplit(x,' '),block_b,'UniformOutput',false);

block_a = vertcat(block_a{:});
block_b = vertcat(block_b{:});

t0_a = str2double(block_a(:,3));
t0_b = str2double(block_b(:,3));
t1_a = str2double(block_a(:,5));
t1_b = str2double(block_b(:,5));

% Collect rating data, A
data_a = cell(size(dims,2),1);
block_a = [];
for I = 1:size(dims,2)
    data_dim = nan(size(t0_a,1),50); % We do not know how many subjects we eventually have and each block may have different amount of subjects, so we create large NaN array first
    idx0 = 1;
    idx1 = 0;
    for J = 1:15 % Blocks
        data_block = table2array(readtable(sprintf('%s/conjuring_%s_a_%i.csv',input,dims{I},J),"ReadVariableNames",false));
        idx1 = idx1 + size(data_block,1);
        data_dim(idx0:idx1,1:size(data_block,2)) = data_block;
        idx0 = idx0 + size(data_block,1);
        
        % Collect block information only once
        if(I==1)
            block_a = vertcat(block_a,repmat(J,size(data_block,1),1));
        end
    end
    data_dim(:,sum(isnan(data_dim),1)==size(t0_a,1)) = []; % Delete redundant columns 
    data_a(I) = {data_dim};
end

% Collect rating data, B
data_b = cell(size(dims,2),1);
block_b = [];
for I = 1:size(dims,2)
    data_dim = nan(size(t0_b,1),50); % We do not know how many subjects we eventually have and each block may have different amount of subjects, so we create large NaN array first
    idx0 = 1;
    idx1 = 0;
    for J = 1:15 % Blocks
        data_block = table2array(readtable(sprintf('%s/conjuring_%s_b_%i.csv',input,dims{I},J),"ReadVariableNames",false));
        idx1 = idx1 + size(data_block,1);
        data_dim(idx0:idx1,1:size(data_block,2)) = data_block;
        idx0 = idx0 + size(data_block,1);
        
        % Collect block information only once
        if(I==1)
            block_b = vertcat(block_b,repmat(J,size(data_block,1),1));
        end
    end
    data_dim(:,sum(isnan(data_dim),1)==size(t0_b,1)) = []; % Delete redundant columns 
    data_b(I) = {data_dim};
end

% Combine interleaved data (A & B) and take the average rating for each 4 sec
% interval.
idx_a = find(t0_a==0); % New trial starting points in eye-tracking, we do not want to combine data between trials since there have been calibration and pause between.
idx_b = find(t0_b==0); % New trial starting points in eye-tracking, we do not want to combine data between trials since there have been calibration and pause between.

ratings = [];
block_a_final = [];
block_b_final = [];
time = [];
t0 = 0;
for I = 1:(size(idx_a,1)) % Trials 
    
    % Indices
    if(I<size(idx_a,1))
        idx_a_trial = idx_a(I):idx_a(I+1)-1;
        idx_b_trial = idx_b(I):idx_b(I+1)-1;
    else % Last trial
        idx_a_trial = idx_a(I):size(t1_a,1);
        idx_b_trial = idx_b(I):size(t1_b,1);
    end

    % Trial timings (ms)
    t_a_trial = horzcat(t0_a(idx_a_trial)*1000,t1_a(idx_a_trial)*1000);
    t_b_trial = horzcat(t0_b(idx_b_trial)*1000,t1_b(idx_b_trial)*1000);
    t1_a_trial = t1_a(idx_a_trial)*1000;
    t1_b_trial = t1_b(idx_b_trial)*1000;
    
    % Correct the trial end timings to match the timings read from the
    % fixation report
    
    % Select all tw:s that started before the end of the trial
    t_a_trial = t_a_trial(t_a_trial(:,1)<vid_dur_eyetracking(I),:);
    t_b_trial = t_b_trial(t_b_trial(:,1)<vid_dur_eyetracking(I),:);
    
    % Take the end time from the fixation report
    t_a_trial(end,2) = vid_dur_eyetracking(I);
    t_b_trial(end,2) = vid_dur_eyetracking(I);
    
    % Collect the block information for the trial
    block_a_trial = block_a(idx_a_trial);
    block_b_trial = block_b(idx_b_trial);
    
    ratings_trial = [];
    for J = 1:size(dims,2)

        % Extract trial dimensions ratings
        ratings_a_dim = data_a{J};
        ratings_b_dim = data_b{J};
        ratings_a_dim_trial = ratings_a_dim(idx_a_trial,:);
        ratings_b_dim_trial = ratings_b_dim(idx_b_trial,:);
        
        % For one trial(conjuring_19) the eye-tracking ended ~6,5 sec
        % earlier than the clip, so it is possible that some rows should de
        % deleted from the end of the rating data
        ratings_a_dim_trial = ratings_a_dim_trial(1:size(t_a_trial,1),:);
        ratings_b_dim_trial = ratings_b_dim_trial(1:size(t_b_trial,1),:);
        if(J==1)
            block_a_trial = block_a_trial(1:size(t_a_trial,1),:);
            block_b_trial = block_b_trial(1:size(t_b_trial,1),:);
        end
    
        % Interpolate to 1 ms interval for easier combining of A & B
        t_trial_int = (1:1:vid_dur_eyetracking(I))';
        ratings_a_dim_trial_int = interp1(t_a_trial(:,1),ratings_a_dim_trial,t_trial_int,'previous');
        ratings_b_dim_trial_int = interp1(t_b_trial(:,1),ratings_b_dim_trial,t_trial_int,'previous');
        
        
        % The last time window is still NaN
        ratings_a_dim_trial_int(t_a_trial(end,1)+1:end,:) = repmat(ratings_a_dim_trial(end,:),t_a_trial(end,2)-t_a_trial(end,1),1);
        ratings_b_dim_trial_int(t_b_trial(end,1)+1:end,:) = repmat(ratings_b_dim_trial(end,:),t_b_trial(end,2)-t_b_trial(end,1),1);

        % Combine A & B by taking the average over all subjects 
        ratings_trial_dim_int = mean(horzcat(ratings_a_dim_trial_int,ratings_b_dim_trial_int),2,'omitnan');
           
        % Downsample to the real rating interval (take average over the time_window to tackle a possible coding error in millisecond scale)
        t_trial = sort(vertcat(t_a_trial(1:end-1,2),t_b_trial(:,2)));
        t = 1;
        ratings_trial_dim = zeros(size(t_trial,1),1);

        for K = 1:size(t_trial,1)
            ratings_trial_dim(K) = mean(ratings_trial_dim_int(t:t_trial(K),1));
            if(J==1)
                ratings_trial_dim(K) = mean(ratings_trial_dim_int(t:t_trial(K),1));
            end
            t = t_trial(K)+1;
        end
        ratings_trial = horzcat(ratings_trial,ratings_trial_dim);
        
        % Create block information column for cheking how many subjects
        % still needed to collect
        if(J==1)
            % Interpolate to ms scale
            block_b_trial_int = interp1(t_b_trial(:,1),block_b_trial,t_trial_int,'previous');
            
            % The last time window is still NaN
            block_b_trial_int(t_a_trial(end,1)+1:end,:) = repmat(block_b_trial_int(end,:),t_a_trial(end,2)-t_a_trial(end,1),1);
            
            % Downsample to the real rating interval (take average over the time_window to tackle a possible coding error in millisecond scale)
            block_b_trial_final = interp1(t_trial_int,block_b_trial_int,t_trial,'nearest');
            
            % The last time window is still NaN
            idx = find(isnan(block_b_trial_final));
            block_b_trial_final(idx) = block_b_trial_final(idx(1)-1);
            
        end
        
 
    end

    % Get information of how many subjects we have for each time point and bpth blocks.
    subs = size(horzcat(ratings_a_dim_trial_int,ratings_b_dim_trial_int),2)-sum(isnan(horzcat(ratings_a_dim_trial_int,ratings_b_dim_trial_int)),2); % How many subjects gave ratings for each time point
    subs = subs(t_trial);
    
    subs_a = size(ratings_a_dim_trial_int,2)-sum(isnan(ratings_a_dim_trial_int),2); % How many subjects gave ratings for each time point
    subs_a = subs_a(t_trial);
    
    subs_b = size(ratings_b_dim_trial_int,2)-sum(isnan(ratings_b_dim_trial_int),2); % How many subjects gave ratings for each time point
    subs_b = subs_b(t_trial);
    

    % Stack trial data into one data table
    t_trial = t_trial + t0;
    ratings_trial = horzcat(ratings_trial,t_trial,subs,subs_a,subs_b,block_b_trial_final,repmat(I,size(ratings_trial,1),1));
    ratings = vertcat(ratings,ratings_trial);
    t0 = t_trial(end);

end

ratings = array2table(ratings);
ratings.Properties.VariableNames = horzcat(dims,'time','subject','subject_a','subject_b','block','trial');

% Save rating data
writetable(ratings,'path/socialdata/conjuring/conjuring_eyetracking_social_regressors.csv');

%% 2. Upsample to 1ms regressors, Experiment 3

regressors = readtable('path/socialdata/conjuring/conjuring_eyetracking_social_regressors.csv');
regressors = regressors(:,1:9);
load('path/eyedata/subdata/conjuring/subjects/C01.mat');
trials = subdata.trial_indices;

% Consistent naming with localizer and kasky
cols = regressors.Properties.VariableNames;
cols{1} = 'pleasant_feelings';
cols{2} = 'unpleasant_feelings';
cols{5} = 'pain';
cols{6} = 'body_movement';
regressors.Properties.VariableNames = cols;
t0 = 0;
for row = 1:size(regressors,1)
    fprintf('%d/%d\n',row,size(regressors,1));
    r = regressors(row,:);
    
    t = r.time;
    r.time_rating = t;
   
    row_1ms = repelem(r,t-t0,1);
    row_1ms.time = (t0+1:t)';
    
    if(row==1)
        regressors_1ms = row_1ms;
    else
        regressors_1ms = vertcat(regressors_1ms,row_1ms);
    end
    
    t0 = t;
end

% Sort predictors to alphabetical order
predictors = regressors_1ms(:,1:8);
timings = regressors_1ms(:,9:end);
sortedNames = sort(predictors.Properties.VariableNames);
predictors = predictors(:,sortedNames);
regressors_1ms = horzcat(predictors,timings);
regressors_1ms.trial = trials;
save('path/socialdata/conjuring/conjuring_eyetracking_social_regressors_1ms.mat','regressors_1ms');

    



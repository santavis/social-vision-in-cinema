%% Read extracted 39 features for plotting covariance analysis (that is done in R, Figure SI5)
%
% Severi Santavirta 14.11.2023

%% INPUT

dset = {'localizer';'kasky';'conjuring'};
excluded = {'C08';'C27';'K05';'K15';'K19';'K20';'K24';'L096'}; % Excluded based on QC
input_eyedata = 'path/eyedata/subdata';
input_predictors_lowlevel = 'path/lowlevel';
input_predictors_highlevel = 'path/socialdata';
input_predictors_cuts = 'path/video_segmentation/cuts';
output = 'path/predictor_correlations'; % where to store the results?

% For kasky and conjuring the last trial contains the end tesxts (no
% lifelike context), exclude those
include_trials = [{1:68},{1:25},{1:29}];

tw = [200,500,1000,2000,4000]; % Sampling time window in milliseconds

%% Load and process predictors
% For localizer the trials=videos are shown continuously and there were only
% 3 calibrations breaks (after trials 17, 34, 51) so the above mentioned procedures are not done for
% each "trial" in Localizer but for the real presentation breaks. For
% Conjuring and Kasky the presentations breaks are between every trial

for w = 1:size(tw,2)
    % Loop over datasets
    x = [];
    for d = 1:size(dset,1)

        % Find the data
        f = find_files(sprintf('%s/%s/',input_eyedata,dset{d}),'*.mat');

        % Exclude bad subjects
        [path,subs,ext] = fileparts(f);
        subs = setdiff(subs,excluded);
        subjects{d} = subs;

        % Loop over subjects
        for s = 1:size(subs,1)
            fprintf('%s: Reading predictor data: %d/%d\n',dset{d},s,size(subs,1));
            eyedata = load(sprintf('%s/%s%s',path{s},subs{s},ext{s}));

            % Define trials
            if(d==1 && s==1) % Localizer has different trials
                trial = zeros(size(eyedata.subdata.trial_indices,1),1);
                trial(1:find(eyedata.subdata.trial_indices==17,1,'last')) = 1;
                trial((find(eyedata.subdata.trial_indices==17,1,'last')+1):find(eyedata.subdata.trial_indices==34,1,'last')) = 2;
                trial((find(eyedata.subdata.trial_indices==34,1,'last')+1):find(eyedata.subdata.trial_indices==51,1,'last')) = 3;
                trial((find(eyedata.subdata.trial_indices==51,1,'last')+1):end) = 4;
            elseif(s==1)
                trial = eyedata.subdata.trial_indices;
            end

            % Drop excluded trials
            trials_included = ismember(trial,include_trials{d});
            trial = trial(trials_included);

            % Load subjectwise gaze class regressors and transform to dummy
            % variables. "Unknown", "Outside video area" and "animals" are
            % excludeded. Not enough animals in the stimulus and others are
            % uninteresting.
            if(s==1)
                gaze = zeros(size(trial,1),size(subs,1),6);
            end
            gaze_class = eyedata.subdata.gaze_class;
            gaze_class = gaze_class(trials_included,:);
            gaze_dummy = zeros(size(gaze_class,1),size(unique(gaze_class),1));
            for c = 1:size(unique(gaze_class),1)
                gaze_dummy(:,c) = gaze_class==c;
            end
            gaze(:,s,:) = gaze_dummy(:,[1:4,6:7]);
        end

        % Create mean gaze position time-series to get the idea of the gaze
        % predictor correlations with the low- and high-level features
        gaze_avg = squeeze(mean(gaze,2));

        % Process lowlevel predictors
        predictors_lowlevel = load(sprintf('%s/%s/lowlevel_data_1ms.mat',input_predictors_lowlevel,dset{d}));
        predictors_lowlevel = table2array(predictors_lowlevel.lowlevel.data);
        predictors_lowlevel = predictors_lowlevel(trials_included,:);
        predictors_lowlevel_processed = processPredictors(predictors_lowlevel,trial,1,0,tw(w));

        % Process gaze class predictors
        predictors_gaze_processed = processGaze(gaze_avg,trial,0,tw(w));

        % Process cut predictor
        predictor_cuts = load(sprintf('%s/%s/gigatrack_%s_scene_cut_regressor.mat',input_predictors_cuts,dset{d},dset{d}));
        predictor_cuts = predictor_cuts.regressor;
        predictor_cuts = predictor_cuts(trials_included,:);
        predictor_cuts = processCuts(predictor_cuts.cut,trial,0,tw(w));

        % Process higlevel predictors
        predictors_highlevel = load(sprintf('%s/%s/%s_eyetracking_social_regressors_1ms.mat',input_predictors_highlevel,dset{d},dset{d}));
        predictors_highlevel = table2array(predictors_highlevel.regressors_1ms(:,1:8));
        predictors_highlevel = predictors_highlevel(trials_included,:);
        predictors_highlevel_processed = processPredictors(predictors_highlevel,trial,1,0,tw(w));

        x = vertcat(x,horzcat(predictors_lowlevel_processed,predictors_gaze_processed,predictor_cuts,predictors_highlevel_processed));

    end

    % Get the predictor names
    predictors_lowlevel = load(sprintf('%s/%s/lowlevel_data_1ms.mat',input_predictors_lowlevel,dset{1}));
    predictors_highlevel = load(sprintf('%s/%s/%s_eyetracking_social_regressors_1ms.mat',input_predictors_highlevel,dset{1},dset{1}));
    predictors_mid = load(sprintf('%s/%s/subjects/L001.mat',input_eyedata,dset{1}));
    predictors_mid = predictors_mid.subdata.gaze_class_catalog([1:4,6:7])';
    predictor_names = horzcat(predictors_lowlevel.lowlevel.data.Properties.VariableNames,predictors_mid,'cuts_dummy',predictors_highlevel.regressors_1ms.Properties.VariableNames(1:8));

    x = array2table(x);
    x.Properties.VariableNames = predictor_names;
    writetable(x,sprintf('%s/predictors_tw%d.csv',output,tw(w)));
end

%% Functions

function [predictors_sub_shift_ds,trial_shift_ds] = processPredictors(predictors,trial,standardize,shift_dur,downsample_dur)
% Function takes the 1ms "predictors" time series and the "trial" indices for
% each millisecond as input and standardizes the data if "standardize"=true. Next the trialwise data is shifted backward the amount 
% specified in "shift_dur" (in milliseconds). Finally, the data is downasampled by
% averaging in time windows specified in "downsample_dur" (in milliseconds)
%
% Severi Santavirta 1.11.2023

    for tr = 1:size(unique(trial),1)

        predictors_trial = predictors(trial==tr,:);

        % Shift predictors backward (cut from the end)
        predictors_trial_shift = predictors_trial(1:(end-shift_dur),:);

        % Downsample
        t = (0:downsample_dur:size(predictors_trial_shift,1))';
        
        % Combine last and second last tw of the trial if the last tw would be under half
        % of the desired tw
        if((size(predictors_trial_shift,1)-t(end))<(downsample_dur/2))
            t = t(1:end-1);
        end
        
        predictors_trial_shift_ds = zeros(size(t,1),size(predictors,2));
        for ti = 1:(size(t,1))
            t0 = t(ti)+1;
            if(ti==size(t,1))
                t1 = size(predictors_trial_shift,1);
            else
                t1 = t(ti+1);
            end
            predictors_trial_shift_ds(ti,:) = mean(predictors_trial_shift(t0:t1,:));
        end

        % Record trial indices
        if(tr==1)
            trial_shift_ds = repmat(tr,size(predictors_trial_shift_ds,1),1);
            predictors_sub_shift_ds = predictors_trial_shift_ds;
        else
            trial_shift_ds = vertcat(trial_shift_ds,repmat(tr,size(predictors_trial_shift_ds,1),1));
            predictors_sub_shift_ds = vertcat(predictors_sub_shift_ds,predictors_trial_shift_ds);
        end
    end
    
    % Standardize
    if(standardize)
        predictors_sub_shift_ds = zscore(predictors_sub_shift_ds);
    end
end
function [predictors_sub_shift_ds,trial_shift_ds] = processCuts(predictor,trial,shift_dur,downsample_dur)
% Function takes the 1ms "cuts" time series and the "trial" indices for
% each millisecond as input. The trialwise data is shifted backward the amount 
% specified in "shift_dur" (in milliseconds). Finally, the data is downsampled by
% in time windows specified in "downsample_dur" (in milliseconds) by
% assigning 1 (cut in the time window) or 0 (no cut in the time window).
%
% Severi Santavirta 1.11.2023

    for tr = 1:size(unique(trial),1)

        predictors_trial = predictor(trial==tr,:);

        % Shift predictors backward (cut from the end)
        predictors_trial_shift = predictors_trial(1:(end-shift_dur),:);

        % Downsample
        t = (0:downsample_dur:size(predictors_trial_shift,1))';
        
        % Combine last and second last tw of the trial if the last tw would be under half
        % of the desired tw
        if((size(predictors_trial_shift,1)-t(end))<(downsample_dur/2))
            t = t(1:end-1);
        end
        
        predictors_trial_shift_ds = zeros(size(t,1),1);
        for ti = 1:(size(t,1))
            t0 = t(ti)+1;
            if(ti==size(t,1))
                t1 = size(predictors_trial_shift,1);
            else
                t1 = t(ti+1);
            end
            % Cut in this tw
            if(any(predictors_trial_shift(t0:t1)))
                
                % Simple regressor 
                predictors_trial_shift_ds(ti,1) = 1; 
                
            end

        end

        % Record trial indices
        if(tr==1)
            trial_shift_ds = repmat(tr,size(predictors_trial_shift_ds,1),1);
            predictors_sub_shift_ds = predictors_trial_shift_ds;
        else
            trial_shift_ds = vertcat(trial_shift_ds,repmat(tr,size(predictors_trial_shift_ds,1),1));
            predictors_sub_shift_ds = vertcat(predictors_sub_shift_ds,predictors_trial_shift_ds);
        end
    end
end
function [predictors_sub_shift_ds,trial_shift_ds] = processGaze(predictors,trial,shift_dur,downsample_dur)
% Function takes the 1ms "gaze" time series and the "trial" indices for
% each millisecond as input. The trialwise data is shifted backward the amount 
% specified in "shift_dur" (in milliseconds). Finally, the data is downsampled
% in time windows specified in "downsample_dur" (in milliseconds) stating 1 if at any time point the subjects was wathing the given class (0 = no time watching the class, 1 = watched the class during the time window).

% Severi Santavirta 1.11.2023

    for tr = 1:size(unique(trial),1)

        predictors_trial = predictors(trial==tr,:,:);

        % Shift predictors backward (cut from the end)
        predictors_trial_shift = predictors_trial(1:(end-shift_dur),:,:);

        % Downsample
        t = (0:downsample_dur:size(predictors_trial_shift,1))';
        
        % Combine last and second last tw of the trial if the last tw would be under half
        % of the desired tw
        if((size(predictors_trial_shift,1)-t(end))<(downsample_dur/2))
            t = t(1:end-1);
        end
        
        predictors_trial_shift_ds = zeros(size(t,1),size(predictors_trial_shift,2),size(predictors_trial_shift,3));
        for ti = 1:(size(t,1))
            t0 = t(ti)+1;
            if(ti==size(t,1))
                t1 = size(predictors_trial_shift,1);
            else
                t1 = t(ti+1);
            end
            % Loop over predictors
            for r = 1:size(predictors,3)
                predictors_trial_shift_ds(ti,:,r) = any(predictors_trial_shift(t0:t1,:,r));
            end
        end

        % Record trial indices
        if(tr==1)
            trial_shift_ds = repmat(tr,size(predictors_trial_shift_ds,1),1);
            predictors_sub_shift_ds = predictors_trial_shift_ds;
        else
            trial_shift_ds = vertcat(trial_shift_ds,repmat(tr,size(predictors_trial_shift_ds,1),1));
            predictors_sub_shift_ds = vertcat(predictors_sub_shift_ds,predictors_trial_shift_ds);
        end
    end
end
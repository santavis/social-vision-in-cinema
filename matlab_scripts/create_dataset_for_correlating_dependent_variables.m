%% Create datasets for correlation analysis (done in R) between dependent variables in different time windows.

% Severi Santavirta 23.4.2024

%% INPUT

dset = {'localizer';'kasky';'conjuring'}; % localizer (Exp. 1), kasky (Exp. 2), conjuring, (Exp. 3)
excluded = {'C08';'C27';'K05';'K15';'K19';'K20';'K24';'L096'}; % Excluded based on QC
input_eyedata = 'path/eyedata/subdata';
input_isc = 'path/isc/isc_changing';
output = 'path/dependent_variable_correlations/data';

% For kasky and conjuring the last trial contains the end tesxts (no
% lifelike context), exclude those
include_trials = [{1:68},{1:25},{1:29}];

% Calculate correlation in multiple time windows
tws = [200,500,1000,2000,4000];

% Pupil size processing parameters
normalize = 50; % The duration of the normalization period in millisecond at the beginning of each trial (all data points are normalized to the mean of this interval)
shift = 0; % How many milliseconds the pupil size should be shifted?

%% Load and process eyedata
% Calculate population average pupil size, ISC and fixation rate for given
% time windows, then calculate how many subjects blinked in a given time
% frame. Calculate correlations within trials separately and then average over trials.

% For localizer the trials=videos are shown continuously and there were only
% 3 calibrations breaks (after trials 17, 34, 51) so the above mentioned procedures are not done for
% each "trial" in Localizer but for the real presentation breaks. For
% Conjuring and Kasky the presentations breaks are between every trial

% Loop over time windows
for tw = 1:size(tws,2)

    for d = 1:size(dset,1)

        % Find the eyedata
        f = find_files(sprintf('%s/%s/',input_eyedata,dset{d}),'*.mat');

        % Exclude bad subjects
        [path,subs,ext] = fileparts(f);
        subs = setdiff(subs,excluded);
        subjects{d} = subs;
        
        % Load ISC time series
        isc_dset = readtable(sprintf('%s/isc_%d_millisecond_tw_%s.csv',input_isc,tws(tw),dset{d}));
        trial = isc_dset.trial;

        % Drop excluded trials
        trials_included = ismember(trial,include_trials{d});
        trial = trial(trials_included);
        subs = isc_dset.Properties.VariableNames(3:end)';
        isc = table2array(isc_dset(trials_included,3:end));

        % Loop over subjects
        for i = 1:size(subs,1)
            fprintf('%s: Reading eyedata: %d/%d\n',dset{d},i,size(subs,1));
            
             % Load pupil size, blinks and fixations
            eyedata = load(sprintf('%s/%s%s',path{i},subs{i},ext{i}));
            pupil_sub = eyedata.subdata.pupil;
            blink_sub = eyedata.subdata.blinks.ts;
            fix_end_times = eyedata.subdata.fixations.timestamps(:,2);
            blink_end_times = eyedata.subdata.blinks.timestamps(:,2);

            % Define trials
            if(d==1 && i==1) % Localizer has different trials
                trial = zeros(size(eyedata.subdata.trial_indices,1),1);
                trial(1:find(eyedata.subdata.trial_indices==17,1,'last')) = 1;
                trial((find(eyedata.subdata.trial_indices==17,1,'last')+1):find(eyedata.subdata.trial_indices==34,1,'last')) = 2;
                trial((find(eyedata.subdata.trial_indices==34,1,'last')+1):find(eyedata.subdata.trial_indices==51,1,'last')) = 3;
                trial((find(eyedata.subdata.trial_indices==51,1,'last')+1):end) = 4;
            elseif(i==1)
                trial = eyedata.subdata.trial_indices;
            end

            % Drop excluded trials
            trials_included = ismember(trial,include_trials{d});
            trial = trial(trials_included);
            pupil_sub = pupil_sub(trials_included,:);
            blink_sub = blink_sub(trials_included,:);

            % Process and collect pupil data
            % Pupil size is normalized separately for each trial by the average
            % of the "normalize" period.
            if(i==1)
                [pupil_dset,trial_ds] = processPupil(pupil_sub,trial,normalize,shift,tws(tw));
                fixationRate_dset = processFixations(fix_end_times,trial,tws(tw));
                blinkRate_dset = processBlinks(blink_end_times,trial,tws(tw));
            else
                pupil_dset(:,i) = processPupil(pupil_sub,trial,normalize,shift,tws(tw));
                fixationRate_dset(:,i) = processFixations(fix_end_times,trial,tws(tw));
                blinkRate_dset(:,i) = processBlinks(blink_end_times,trial,tws(tw));
            end
    
        end
        
        fname = sprintf('%s/pupil_for_timeseries_%s_tw%d.csv',output,dset{d},tws(tw));
        writetable(array2table(pupil_dset),fname);
        fname = sprintf('%s/isc_for_timeseries_%s_tw%d.csv',output,dset{d},tws(tw));
        writetable(isc_dset,fname);

        % Calculate how many subjects blinked (at least once) in each time
        % window
        blinkRate_dset_dummy = blinkRate_dset>0;
        blinks = sum(blinkRate_dset_dummy,2)/size(blinkRate_dset_dummy,2);
        
        % Create data matrix and save it (calculate correlations in R)
        data = array2table(trial_ds);
        data.pupil = mean(pupil_dset,2,"omitnan");
        data.isc = mean(isc,2,"omitnan");
        data.fixation_rate = mean(fixationRate_dset,2,"omitnan");
        data.blinks = blinks;

        fname = sprintf('%s/correlation_dataset_%s_tw%d.csv',output,dset{d},tws(tw));
        writetable(data,fname);
        
    end
end

function [pupil_sub_shift_ds,trial_shift_ds] = processPupil(pupil,trial,normalize_dur,shift_dur,downsample_dur)
% Function takes the 1ms "pupil" time series and the "trial" indices for
% each millisecond as input and normalizes the data separately for each trial. The data is
% normalized with the mean pupil size of the first milliseconds specified
% in "normalize_dur". Next the trialwise data is shifted forward the amount 
% specified in "shift_dur" (in milliseconds). Finally, the data is downsampled by
% averaging in time windows specified in "downsample_dur" (in milliseconds)
%
% Severi Santavirta 1.11.2023

    for tr = 1:size(unique(trial),1)

        pupil_trial = pupil(trial==tr);

        % Normalization: Divide with the mean of the first milliseconds
        pupil_trial = pupil_trial./(mean(pupil_trial(1:normalize_dur)));

        % Shift pupil forward (cut from the beginning)
        pupil_trial_shift = pupil_trial((shift_dur+1):end);

        % Downsample
        t = (0:downsample_dur:size(pupil_trial_shift,1))';
        
        % Combine last and second last tw of the trial if the last tw would be under half
        % of the desired tw
        if((size(pupil_trial_shift,1)-t(end))<(downsample_dur/2))
            t = t(1:end-1);
        end
        pupil_trial_shift_ds = zeros(size(t,1),1);
        for ti = 1:(size(t,1))
            t0 = t(ti)+1;
            if(ti==size(t,1))
                t1 = size(pupil_trial_shift,1);
            else
                t1 = t(ti+1);
            end
            pupil_trial_shift_ds(ti) = mean(pupil_trial_shift(t0:t1));
        end

        % Record trial indices
        if(tr==1)
            trial_shift_ds = repmat(tr,size(pupil_trial_shift_ds,1),1);
            pupil_sub_shift_ds = pupil_trial_shift_ds;
        else
            trial_shift_ds = vertcat(trial_shift_ds,repmat(tr,size(pupil_trial_shift_ds,1),1));
            pupil_sub_shift_ds = vertcat(pupil_sub_shift_ds,pupil_trial_shift_ds);
        end
    end  
end
function fixationRate = processFixations(fix_end_times,trial,tw)
% Function calculates the fixation rates for each time window.
%
% Severi Santavirta 9.11.2023

    for tr = 1:size(unique(trial),1)
        
        % Trial timing
        t0 = find(trial==tr,1);
        t1 = find(trial==tr,1,'last');
        dur_tr = sum(trial==tr);

        % Fixation end timings related to the trial start time
        fix_end_trial = fix_end_times(fix_end_times>t0 & fix_end_times<t1)-t0;

        % Time window timings for downsampling
        t = (0:tw:dur_tr)';
        
        % Combine last and second last tw of the trial if the last tw would be under half
        % of the desired tw
        if((dur_tr-t(end))<(tw/2))
            t = t(1:end-1);
        end

        fixationRate_trial = zeros(size(t,1),1);
        for ti = 1:(size(t,1))
            t0_tw = t(ti);
            if(ti==size(t,1))
                t1_tw = dur_tr;
            else
                t1_tw = t(ti+1);
            end

            % Calculate how many different fixations occured within this
            % time window
            fixationRate_trial(ti,1) = sum(fix_end_trial>t0_tw & fix_end_trial<t1_tw)+1;
        end

        if(tr==1)
            fixationRate = fixationRate_trial;
        else
            fixationRate = vertcat(fixationRate,fixationRate_trial);
        end
    end  
end
function blinkRate = processBlinks(blink_end_times,trial,tw)
% Function calculates the blink rates for each time window (the number of ending blinks wihtin the tw).
%
% Severi Santavirta 5.12.2023

    for tr = 1:size(unique(trial),1)
        
        % Trial timing
        t0 = find(trial==tr,1);
        t1 = find(trial==tr,1,'last');
        dur_tr = sum(trial==tr);

        % Blink end timings related to the trial start time
        blink_end_trial = blink_end_times(blink_end_times>t0 & blink_end_times<t1)-t0;

        % Time window timings for downsampling
        t = (0:tw:dur_tr)';
        
        % Combine last and second last tw of the trial if the last tw would be under half
        % of the desired tw
        if((dur_tr-t(end))<(tw/2))
            t = t(1:end-1);
        end

        blinkRate_trial = zeros(size(t,1),1);
        for ti = 1:(size(t,1))
            t0_tw = t(ti);
            if(ti==size(t,1))
                t1_tw = dur_tr;
            else
                t1_tw = t(ti+1);
            end

            % Calculate how many different blinks occured within this
            % time window
            blinkRate_trial(ti,1) = sum(blink_end_trial>t0_tw & blink_end_trial<t1_tw);
        end

        if(tr==1)
            blinkRate = blinkRate_trial;
        else
            blinkRate = vertcat(blinkRate,blinkRate_trial);
        end
    end  
end
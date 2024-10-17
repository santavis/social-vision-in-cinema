%% Reference histograms for typical eye-tracking parameters when people view movies
%
%       1. Fixation, saccade and blink durations
%       2. Fixation, saccade and blink rates (per second)
%       3. Total time for fixation, saccades and blinks
%
% Extract the data and plot in R (Figure 1).
%
% Severi Santavirta 18.10.2023

%% Read eyedata from all subjects

excluded = {'C08';'C27';'K05';'K15';'K19';'K20';'K24';'L096'}; % Excluded based on QC
dset = {'localizer','kasky','conjuring'}; % Experiments 1-3

% Loop over datasets

% Occurrence rate
fix_sec = [];
sac_sec = [];
blink_sec = [];
dset_sec = {};
subj_sec = {};

% Total tme distribution
fix_dur = [];
sac_dur = [];
blink_dur = [];
fix_sub = {};
fix_dset = {};
sac_sub = {};
sac_dset = {};
blink_sub = {};
blink_dset = {};

% Duration distribution
fix_total = [];
sac_total = [];
blink_total = [];

n=0;
for d = 1:size(dset,2)
    
    % Subjects
    input = sprintf('path/eyedata/subdata/%s/subjects',dset{d}); % Where are the eye-tracking data?
    f = find_files(input,'*.mat');
    [~,subjects,~] = fileparts(f);
    subjects = setdiff(subjects,excluded); % Exclude bad subjects

    % Loop over subjects
    for s = 1:size(subjects,1)
        fprintf('Reading fixations, %s: %i/%i\n',dset{d},s,size(subjects,1));
        n=n+1;

        % Load eyedata
        load(sprintf('%s/%s.mat',input,subjects{s}));

        % To get the number of fixations/blinks/saccades per 1 second, we divide the
        % total count by the duration of the experiment
        fix_sec(n,1) = subdata.fixations.count/(size(subdata.trial_indices,1)/1000);
        sac_sec(n,1) = subdata.saccades.count/(size(subdata.trial_indices,1)/1000);
        blink_sec(n,1) = subdata.blinks.count/(size(subdata.trial_indices,1)/1000);
        dset_idx(n,1) = dset(d);
        subj_sec = vertcat(subj_sec,subjects(s));

        % Get the total time of fixations/blinks/saccades per subject
        fix_total(n,1) = subdata.fixations.total_time;
        sac_total(n,1) = subdata.saccades.total_time;
        blink_total(n,1) = subdata.blinks.total_time;

        
        
        % Extract saccade durations and exclude those saccades that
        % contain blink since they may not bee "real" saccades. 
        blink_timestamps = subdata.blinks.timestamps;
        sac_timestamps = subdata.saccades.timestamps;
        sac_dur_sub = subdata.saccades.durations;
        blink_dur_sub = subdata.blinks.durations;
        blink_saccades = zeros(size(sac_timestamps,1),1);
        for sac = 1:size(sac_timestamps,1)
            blink_within_sac = find(blink_timestamps(:,1)>=sac_timestamps(sac,1) & blink_timestamps(:,2)<=sac_timestamps(sac,2));
            if(~isempty(blink_within_sac))
                a = 1;
                blink_saccades(sac) = 1;
            end
        end
        sac_dur_sub(logical(blink_saccades)) = [];
        sac_dur = vertcat(sac_dur,sac_dur_sub);
        
        % Extract fixation durations 
        fix_dur_sub = subdata.fixations.durations;
        fix_dur = vertcat(fix_dur,fix_dur_sub);
        
        % Extract blink durations
        blink_dur = vertcat(blink_dur,blink_dur_sub);

        % For plotting durations we need the information about subjects
        % and dataset
        fix_sub = vertcat(fix_sub,repmat(subjects(s),size(fix_dur_sub,1),1));
        sac_sub = vertcat(sac_sub,repmat(subjects(s),size(sac_dur_sub,1),1));
        blink_sub = vertcat(blink_sub,repmat(subjects(s),size(blink_dur_sub,1),1));
        fix_dset = vertcat(fix_dset,repmat(dset(d),size(fix_dur_sub,1),1));
        sac_dset = vertcat(sac_dset,repmat(dset(d),size(sac_dur_sub,1),1));
        blink_dset = vertcat(blink_dset,repmat(dset(d),size(blink_dur_sub,1),1));

    end
end

% Create tables
data = horzcat(array2table(fix_sec),array2table(sac_sec),array2table(blink_sec),array2table(fix_total),array2table(sac_total),array2table(blink_total),array2table(subj_sec),array2table(dset_idx));
data.Properties.VariableNames = {'fixation_per_second','saccades_per_second','blinks_per_second','fixations_total_time','saccades_total_time','blinks_total_time','subject','dataset'};
writetable(data,'path/typical_eye_measures/second_data.csv');

data_fix = horzcat(array2table(fix_dur),array2table(fix_sub),array2table(fix_dset));
data_fix.Properties.VariableNames = {'fixation_durations','subject','dataset'};
writetable(data_fix,'path/typical_eye_measures/fixation_durations.csv');
data_sac = horzcat(array2table(sac_dur),array2table(sac_sub),array2table(sac_dset));
data_sac.Properties.VariableNames = {'saccade_durations','subject','dataset'};
writetable(data_sac,'path/typical_eye_measures/saccade_durations.csv');
data_blink = horzcat(array2table(blink_dur),array2table(blink_sub),array2table(blink_dset));
data_blink.Properties.VariableNames = {'blink_durations','subject','dataset'};
writetable(data_blink,'path/typical_eye_measures/blink_durations.csv');



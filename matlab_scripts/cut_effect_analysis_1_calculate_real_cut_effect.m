%% Cut Effect Analysis: Analyse pupil size, synchronization and blink rate changes after each cut separately for each experiment
% Extract eyedata for some duration after each scene cut and plot changes in eISC, blinks synchroniation and
% pupil size to define the effect of the scene cut

% Information for heatmap kernel calculations
% Localizer (Exp. 1): sigma=1, viewDist = 90, viewWidth = 31, viewWidthPixel = 720, viewHeightPixel = 576, VideoArea: x = [153,873], y = [97,672]
% Kasky (Exp. 2): sigma=1, viewDist = 70 (assumption), viewWidth = 61, viewWidthPixel = 1024, viewHeightPixel = 580, VideoArea: x = [0,1024], y = [96,676]
% Conjuring (Exp. 3): sigma=1, viewDist = 70 (assumption), viewWidth = ,60 viewWidthPixel = 1000, viewHeightPixel = 564, VideoArea: x = [13,1012], y = [105,668]

% Severi Santavirta 27.10.2023

%% INPUT, Experiment 1

dataset = 'localizer'; % localizer (Exp. 1), kasky (Exp. 2), conjuring (Exp. 3)
tw = 3000; % How long duration after each cut is taken into the analysis
tww = 200; % In how short intervals the tw is analysed
twws_before = 3; % How many time windows to calculate before each cut for reference
npool = 3; % Number of workers
include_trials = 1:68;

excluded = {'C08';'C27';'K05';'K15';'K19';'K20';'K24';'L096'}; % Excluded based on QC
video_area_x = 153; % The first x coordinate that is inside video area
video_area_y = 97; % The first y coordinate that is inside video area

sigma = 1; % heatmap radius as degrees (Nummenmaa 2014 or Lahnakoski 2014)
viewDist = 90; %cm, viewing distance
viewWidth = 31; %cm, Width of the presentations area in cm, not the whole screen width
viewWidthPixel = 720; % Width of the presentation area as pixels, not the whole screen width
viewHeightPixel = 576; % Height of the presentation area as pixels, not the whole screen width

input = sprintf('path/eyedata/subdata/%s/subjects',dataset); % Where are the eye-tracking data?
cuts = readtable(sprintf('path/video_segmentation/cuts/%s/gigatrack_%s_scene_cut_times.csv',dataset,dataset));% Where is the cut information?
output = sprintf('path/scene_cut_effect/scene_cut_effect_%s.mat',dataset); % Where to store the results?

%%  Read fixations from preprocessed eye tracker data, Experiment 1

f = find_files(input,'*.mat');
[~,subjects,~] = fileparts(f);
subjects = setdiff(subjects,excluded);

for I = 1:size(subjects,1)
    fprintf('Reading fixations: %i/%i\n',I,size(subjects,1));

    load(sprintf('%s/%s.mat',input,subjects{I}));
    x_sub = subdata.fix_x;
    y_sub = subdata.fix_y;
    pupil_sub = subdata.pupil;
    blink_sub = subdata.blinks.ts;
    
    x_sub = x_sub-video_area_x+1; % correct coordinates for video area
    y_sub = y_sub-video_area_y+1; % correct coordinates for video area
    
    if(I==1)
        x = zeros(size(x_sub,1),size(subjects,1));
        y = zeros(size(x_sub,1),size(subjects,1));
        pupil = zeros(size(x_sub,1),size(subjects,1));
        blink = zeros(size(x_sub,1),size(subjects,1));
    end
    x(:,I) = x_sub;
    y(:,I) = y_sub;
    pupil(:,I) = pupil_sub;
    blink(:,I) = blink_sub;
end

%% Create Gaussian kernel (sigma 1 degrees, Nummenmaa 2014 or Lahnakoski 2014))

kern = eISC_gaussKernel(sigma,[],viewDist,viewWidth,viewWidthPixel,1);

%% Calculate eyedata for each cut, Experiment 1

% Identify time points where new trial starts. We cannot analyze these as
% cuts, since we cannot normalize the data with the states before the trial
% started. Theses cuts are removed from the list of cuts.
trial_start_times = zeros(4,1); % Hard coded for LOCALIZER since the clips whre shown in 4 trials
trial_start_times(1) = find(subdata.trial_indices==1,1);
trial_start_times(2) = find(subdata.trial_indices==18,1);
trial_start_times(3) = find(subdata.trial_indices==35,1);
trial_start_times(4) = find(subdata.trial_indices==52,1);
[~,ia,~] = intersect(cuts.time,trial_start_times);
cuts(ia,:) = [];

% Exclude cuts in trials that are not included
cuts = cuts(ismember(cuts.trial,include_trials),:);

% Trial indices need to be corrected for LOCALIZER only
trial_true = zeros(size(cuts,1),1);
trial_true(ismember(cuts.trial,1:17)) = 1;
trial_true(ismember(cuts.trial,18:34)) = 2;
trial_true(ismember(cuts.trial,35:51)) = 3;
trial_true(ismember(cuts.trial,52:68)) = 4;
cuts.trial = trial_true;

% Remove trials that do not have enough time within trial before (tww) or
% after (tw) the cut'
rmcut = zeros(size(cuts,1),1);
for c = 1:size(cuts,1)
    
    % Cut start time
    t_start = cuts.time(c);
    
    % Time between cut and trial start
    t_before = t_start - trial_start_times(cuts.trial(c));
    
    % Time between cut and trial end
    if(cuts.trial(c) < max(cuts.trial))
        t_after = trial_start_times(cuts.trial(c)+1) - t_start;
    else
        t_after = size(y,1) - t_start;
    end
    
    if(t_before<tww*twws_before && t_after<tw)
        rmcut(c) = 1;
    end
end
cuts(logical(rmcut),:) = []; 

% Open parallel pool
p = gcp('nocreate'); % If no pool, do not create new one.
if(isempty(p))
    p = parpool(npool);
end

%Load if already calculated some rounds
if(exist(output,'file'))
    load(output);
    r = results.eisc;
    blinks_avg = results.blink_synchronization;
    pupil_change = results.pupil_change;
    tmp = results.pupil_change(:,1);
    n = find(tmp==0,1);
else
    % Extract eyedata for each cut
    r = zeros(size(cuts,1),floor(tw/tww)+twws_before);
    pupil_change = zeros(size(cuts,1),tw+tww*twws_before);
    blinks_avg = zeros(size(cuts,1),floor(tw/tww)+twws_before);
    n = 1;
end
for c = n:size(cuts,1)
    tic;
    fprintf('Reading eyedata, cut: %i/%i\n',c,size(cuts,1));
    
    % Cut start time
    t_start = cuts.time(c);

    % Slice (with reference time period before the cut)
    y_cut = y((t_start-tww*twws_before):(t_start+tw-1),:);
    x_cut = x((t_start-tww*twws_before):(t_start+tw-1),:);
    pupil_cut = pupil((t_start-tww*twws_before):(t_start+tw-1),:);
    blink_cut = blink((t_start-tww*twws_before):(t_start+tw-1),:);
    
    % Small time window start times
    t0 = 1:tww:size(y_cut,1);
    t1 = tww:tww:size(y_cut,1);

    % Loop through the small time windows ofter the cut
    parfor i = 1:size(t0,2)

        t0_i = t0(i);
        t1_i = t1(i);

        % eISC for the small time windows
        % The data is stored in millisecond interval instead of fixations, hence this
        duration = zeros(tww,1);
        duration(:) = 1;

        % Initialize the heatmap variable
        fixation_heatmap = zeros(viewHeightPixel,viewWidthPixel,size(subjects,1));
        for s = 1:size(subjects,1) % subjects

            % Convert gaze coordinates to eISC format
            points = zeros(tww,2);
            points(:,1) = y_cut(t0(i):t1(i),s);
            points(:,2) = x_cut(t0(i):t1(i),s);

            % Calculate heatmap
            fixation_heatmap(:,:,s) = eISC_fixationHeatmap(points,kern,viewWidthPixel,viewHeightPixel,duration);

        end

        % Calculate ISC
        r(c,i) = eISC_spatialSimilarity(fixation_heatmap);

        % Calculate the avarage number of blinks (over subjects) for each
        % time points. If there were significant blink synchronizations, it
        % would be visible in the resulting time-series
        blink_tww = blink_cut(t0_i:t1_i,:);

        % Calculate how many subjects blinked within this tww and save the
        % proprotion of the total number of subjects
        blinks_avg(c,i) = sum(any(blink_tww))./size(subjects,1);
    end
    
    % Calculate the pupil size change for this cut by first normalizing the pupil size for each subject and then taking the average
    pupil_cut_norm = pupil_cut./mean(pupil_cut(1:tww*(twws_before),:));
    pupil_change(c,:) = mean(pupil_cut_norm,2,'omitnan');

    % Save results each round to protect for data loss
    results = struct;
    results.eisc = r;
    results.blink_synchronization = blinks_avg;
    results.pupil_change = pupil_change;
    save(output,'results');
    toc;  
end

% Normalize ISC to the first time windows (windows before the cut)
results.eisc_norm = results.eisc./mean(results.eisc(:,1:(twws_before)),2);

% Normalize blinks to the first time windows (windows before the cut)
results.blink_synchronization_norm = (results.blink_synchronization+1)./mean((results.blink_synchronization(:,1:(twws_before))+1),2);

%%  Save also as a csv file, Experiment 1
[path,name,ext] = fileparts(output);
writetable(array2table(results.eisc),sprintf('%s/scene_cut_effect_eisc_%s.csv',path,dataset),'WriteVariableNames',false);
writetable(array2table(results.eisc_norm),sprintf('%s/scene_cut_effect_eisc_normalized_%s.csv',path,dataset),'WriteVariableNames',false);
writetable(array2table(results.pupil_change),sprintf('%s/scene_cut_effect_pupil_%s.csv',path,dataset),'WriteVariableNames',false);
writetable(array2table(results.blink_synchronization),sprintf('%s/scene_cut_effect_blink_%s.csv',path,dataset),'WriteVariableNames',false);
writetable(array2table(results.blink_synchronization_norm),sprintf('%s/scene_cut_effect_blink_normalized_%s.csv',path,dataset),'WriteVariableNames',false);

clear; clc;

%% INPUT, Experiment 2

dataset = 'kasky'; % localizer (Exp. 1), kasky (Exp. 2), conjuring (Exp. 3)
tw = 3000; % How long duration after each cut is taken into the analysis
tww = 200; % In how short intervals the tw is analysed
twws_before = 3; % How many time windows to calculate before each cut for reference
npool = 10; % Number of workers
include_trials = 1:25; 

excluded = {'C08';'C27';'K05';'K15';'K19';'K20';'K24';'L096'}; % Excluded based on QC
video_area_x = 0; % The first x coordinate that is inside video area
video_area_y = 97; % The first y coordinate that is inside video area

sigma = 1; % heatmap radius as degrees (Nummenmaa 2014 or Lahnakoski 2014)
viewDist = 70; %cm, viewing distance
viewWidth = 61; %cm, Width of the presentations area in cm, not the whole screen width
viewWidthPixel = 1024; % Width of the presentation area as pixels, not the whole screen width
viewHeightPixel = 580; % Height of the presentation area as pixels, not the whole screen width

input = sprintf('path/eyedata/subdata/%s/subjects',dataset); % Where are the eye-tracking data?
cuts = readtable(sprintf('path/video_segmentation/cuts/%s/gigatrack_%s_scene_cut_times.csv',dataset,dataset));% Where is the cut information?
output = sprintf('path/scene_cut_effect/scene_cut_effect_%s.mat',dataset); % Where to store the results?

%%  Read fixations from preprocessed eye tracker data, Experiment 2

f = find_files(input,'*.mat');
[~,subjects,~] = fileparts(f);
subjects = setdiff(subjects,excluded);

for I = 1:size(subjects,1)
    fprintf('Reading fixations: %i/%i\n',I,size(subjects,1));

    load(sprintf('%s/%s.mat',input,subjects{I}));
    x_sub = subdata.fix_x;
    y_sub = subdata.fix_y;
    pupil_sub = subdata.pupil;
    blink_sub = subdata.blinks.ts;
    
    x_sub = x_sub-video_area_x+1; % correct coordinates for video area
    y_sub = y_sub-video_area_y+1; % correct coordinates for video area
    
    if(I==1)
        x = zeros(size(x_sub,1),size(subjects,1));
        y = zeros(size(x_sub,1),size(subjects,1));
        pupil = zeros(size(x_sub,1),size(subjects,1));
        blink = zeros(size(x_sub,1),size(subjects,1));
    end
    x(:,I) = x_sub;
    y(:,I) = y_sub;
    pupil(:,I) = pupil_sub;
    blink(:,I) = blink_sub;
end

%% Create Gaussian kernel (sigma 1 degrees, Nummenmaa 2014 or Lahnakoski 2014))

kern = eISC_gaussKernel(sigma,[],viewDist,viewWidth,viewWidthPixel,1);

%% Calculate eyedata for each cut, Experiment 2

% Identify time points where new trial starts. We cannot analyze these as
% cuts, since we cannot normalize the data with the states before the trial
% started. Theses cuts are removed from the list of cuts.
trial_start_times = zeros(size(unique(cuts.trial),1),1);
for tr = 1:size(trial_start_times,1)
    trial_start_times(tr) = find(subdata.trial_indices==tr,1);
end
[~,ia,~] = intersect(cuts.time,trial_start_times);
cuts(ia,:) = [];

% Exclude cuts in trials that are not included
cuts = cuts(ismember(cuts.trial,include_trials),:);

% Remove trials that do not have enough time within trial before (tww) or
% after (tw) the cut'
rmcut = zeros(size(cuts,1),1);
for c = 1:size(cuts,1)
    
    % Cut start time
    t_start = cuts.time(c);
    
    % Time between cut and trial start
    t_before = t_start - trial_start_times(cuts.trial(c));
    
    % Time between cut and trial end
    if(cuts.trial(c) < max(cuts.trial))
        t_after = trial_start_times(cuts.trial(c)+1) - t_start;
    else
        t_after = size(y,1) - t_start;
    end
    
    if(t_before<tww*twws_before && t_after<tw)
        rmcut(c) = 1;
    end
end
cuts(logical(rmcut),:) = []; 

% Open parallel pool
p = gcp('nocreate'); % If no pool, do not create new one.
if(isempty(p))
    p = parpool(npool);
end

%Load if already calculated some rounds
if(exist(output,'file'))
    load(output);
    r = results.eisc;
    blinks_avg = results.blink_synchronization;
    pupil_change = results.pupil_change;
    tmp = results.pupil_change(:,1);
    n = find(tmp==0,1);
else
    % Extract eyedata for each cut
    r = zeros(size(cuts,1),floor(tw/tww)+twws_before);
    pupil_change = zeros(size(cuts,1),tw+tww*twws_before);
    blinks_avg = zeros(size(cuts,1),floor(tw/tww)+twws_before);
    n = 1;
end
for c = n:size(cuts,1)
    tic;
    fprintf('Reading eyedata, cut: %i/%i\n',c,size(cuts,1));
    
    % Cut start time
    t_start = cuts.time(c);

    % Slice (with reference time period before the cut)
    y_cut = y((t_start-tww*twws_before):(t_start+tw-1),:);
    x_cut = x((t_start-tww*twws_before):(t_start+tw-1),:);
    pupil_cut = pupil((t_start-tww*twws_before):(t_start+tw-1),:);
    blink_cut = blink((t_start-tww*twws_before):(t_start+tw-1),:);
    
    % Small time window start times
    t0 = 1:tww:size(y_cut,1);
    t1 = tww:tww:size(y_cut,1);

    % Loop through the small time windows ofter the cut
    parfor i = 1:size(t0,2)

        t0_i = t0(i);
        t1_i = t1(i);

        % eISC for the small time windows
        % The data is stored in millisecond interval instead of fixations, hence this
        duration = zeros(tww,1);
        duration(:) = 1;

        % Initialize the heatmap variable
        fixation_heatmap = zeros(viewHeightPixel,viewWidthPixel,size(subjects,1));
        for s = 1:size(subjects,1) % subjects

            % Convert gaze coordinates to eISC format
            points = zeros(tww,2);
            points(:,1) = y_cut(t0(i):t1(i),s);
            points(:,2) = x_cut(t0(i):t1(i),s);

            % Calculate heatmap
            fixation_heatmap(:,:,s) = eISC_fixationHeatmap(points,kern,viewWidthPixel,viewHeightPixel,duration);

        end

        % Calculate ISC
        r(c,i) = eISC_spatialSimilarity(fixation_heatmap);

        % Calculate the avarage number of blinks (over subjects) for each
        % time points. If there were significant blink synchronizations, it
        % would be visible in the resultting time-series
        blink_tww = blink_cut(t0_i:t1_i,:);

        % Calculate how many subjects blinked within this tww and save the
        % proprotion of the total number of subjects
        blinks_avg(c,i) = sum(any(blink_tww))./size(subjects,1);

    end

    % Calculate the pupil size change for this cut by first normalizing the pupil size for each subject and then taking the average
    pupil_cut_norm = pupil_cut./mean(pupil_cut(1:tww*(twws_before),:));
    pupil_change(c,:) = mean(pupil_cut_norm,2,'omitnan');

    % Save results each round to protect for data loss
    results = struct;
    results.eisc = r;
    results.blink_synchronization = blinks_avg;
    results.pupil_change = pupil_change;
    save(output,'results');
    toc;  
end

% Normalize ISC to the first time windows (windows before the cut)
results.eisc_norm = results.eisc./mean(results.eisc(:,1:(twws_before)),2);

% Normalize blinks to the first time windows (windows before the cut)
results.blink_synchronization_norm = (results.blink_synchronization+1)./mean((results.blink_synchronization(:,1:(twws_before))+1),2);

%%  Save also as a csv file, Experiment 2
[path,name,ext] = fileparts(output);
writetable(array2table(results.eisc),sprintf('%s/scene_cut_effect_eisc_%s.csv',path,dataset),'WriteVariableNames',false);
writetable(array2table(results.eisc_norm),sprintf('%s/scene_cut_effect_eisc_normalized_%s.csv',path,dataset),'WriteVariableNames',false);
writetable(array2table(results.pupil_change),sprintf('%s/scene_cut_effect_pupil_%s.csv',path,dataset),'WriteVariableNames',false);
writetable(array2table(results.blink_synchronization),sprintf('%s/scene_cut_effect_blink_%s.csv',path,dataset),'WriteVariableNames',false);
writetable(array2table(results.blink_synchronization_norm),sprintf('%s/scene_cut_effect_blink_normalized_%s.csv',path,dataset),'WriteVariableNames',false);

clear; clc;

%% INPUT, Experiment 3

dataset = 'conjuring'; % localizer (Exp. 1), kasky (Exp. 2), conjuring (Exp. 3)
tw = 3000; % How long duration after each cut is taken into the analysis
tww = 200; % In how short intervals the tw is analysed
twws_before = 3; % How many time windows to calculate before each cut for reference
npool = 3; % Number of workers
include_trials = 1:29; 

excluded = {'C08';'C27';'K05';'K15';'K19';'K20';'K24';'L096'}; % Excluded based on QC
video_area_x = 13; % The first x coordinate that is inside video area
video_area_y = 105; % The first y coordinate that is inside video area

sigma = 1; % heatmap radius as degrees (Nummenmaa 2014 or Lahnakoski 2014)
viewDist = 70; %cm, viewing distance
viewWidth = 60; %cm, Width of the presentations area in cm, not the whole screen width
viewWidthPixel = 1000; % Width of the presentation area as pixels, not the whole screen width
viewHeightPixel = 564; % Height of the presentation area as pixels, not the whole screen width

input = sprintf('path/eyedata/subdata/%s/subjects',dataset); % Where are the eye-tracking data?
cuts = readtable(sprintf('path/video_segmentation/cuts/%s/gigatrack_%s_scene_cut_times.csv',dataset,dataset));% Where is the cut information?
output = sprintf('path/scene_cut_effect/scene_cut_effect_%s.mat',dataset); % Where to store the results?

%%  Read fixations from preprocessed eye tracker data, Experiment 3

f = find_files(input,'*.mat');
[~,subjects,~] = fileparts(f);
subjects = setdiff(subjects,excluded);

for I = 1:size(subjects,1)
    fprintf('Reading fixations: %i/%i\n',I,size(subjects,1));

    load(sprintf('%s/%s.mat',input,subjects{I}));
    x_sub = subdata.fix_x;
    y_sub = subdata.fix_y;
    pupil_sub = subdata.pupil;
    blink_sub = subdata.blinks.ts;
    
    x_sub = x_sub-video_area_x+1; % correct coordinates for video area
    y_sub = y_sub-video_area_y+1; % correct coordinates for video area
    
    
    if(I==1)
        x = zeros(size(x_sub,1),size(subjects,1));
        y = zeros(size(x_sub,1),size(subjects,1));
        pupil = zeros(size(x_sub,1),size(subjects,1));
        blink = zeros(size(x_sub,1),size(subjects,1));
    end
    x(:,I) = x_sub;
    y(:,I) = y_sub;
    pupil(:,I) = pupil_sub;
    blink(:,I) = blink_sub;
end

%% Create Gaussian kernel (sigma 1 degrees, Nummenmaa 2014 or Lahnakoski 2014))

kern = eISC_gaussKernel(sigma,[],viewDist,viewWidth,viewWidthPixel,1);

%% Calculate eyedata for each cut

% Identify time points where new trial starts. We cannot analyze these as
% cuts, since we cannot normalize the data with the states before the trial
% started. Theses cuts are removed from the list of cuts.
trial_start_times = zeros(size(unique(cuts.trial),1),1);
for tr = 1:size(trial_start_times,1)
    trial_start_times(tr) = find(subdata.trial_indices==tr,1);
end
[~,ia,~] = intersect(cuts.time,trial_start_times);
cuts(ia,:) = [];

% Exclude cuts in trials that are not included
cuts = cuts(ismember(cuts.trial,include_trials),:);

% Remove trials that do not have enough time within trial before (tww) or
% after (tw) the cut'
rmcut = zeros(size(cuts,1),1);
for c = 1:size(cuts,1)
    
    % Cut start time
    t_start = cuts.time(c);
    
    % Time between cut and trial start
    t_before = t_start - trial_start_times(cuts.trial(c));
    
    % Time between cut and trial end
    if(cuts.trial(c) < max(cuts.trial))
        t_after = trial_start_times(cuts.trial(c)+1) - t_start;
    else
        t_after = size(y,1) - t_start;
    end
    
    if(t_before<tww*twws_before && t_after<tw)
        rmcut(c) = 1;
    end
end
cuts(logical(rmcut),:) = []; 

% Open parallel pool
p = gcp('nocreate'); % If no pool, do not create new one.
if(isempty(p))
    p = parpool(npool);
end

%Load if already calculated some rounds
if(exist(output,'file'))
    load(output);
    r = results.eisc;
    blinks_avg = results.blink_synchronization;
    pupil_change = results.pupil_change;
    tmp = results.pupil_change(:,1);
    n = find(tmp==0,1);
else
    % Extract eyedata for each cut
    r = zeros(size(cuts,1),floor(tw/tww)+twws_before);
    pupil_change = zeros(size(cuts,1),tw+tww*twws_before);
    blinks_avg = zeros(size(cuts,1),floor(tw/tww)+twws_before);
    n = 1;
end
for c = n:size(cuts,1)
    tic;
    fprintf('Reading eyedata, cut: %i/%i\n',c,size(cuts,1));
    
    % Cut start time
    t_start = cuts.time(c);

    % Slice (with reference time period before the cut)
    y_cut = y((t_start-tww*twws_before):(t_start+tw-1),:);
    x_cut = x((t_start-tww*twws_before):(t_start+tw-1),:);
    pupil_cut = pupil((t_start-tww*twws_before):(t_start+tw-1),:);
    blink_cut = blink((t_start-tww*twws_before):(t_start+tw-1),:);
    
    % Small time window start times
    t0 = 1:tww:size(y_cut,1);
    t1 = tww:tww:size(y_cut,1);

    % Loop through the small time windows ofter the cut
    parfor i = 1:size(t0,2)

        t0_i = t0(i);
        t1_i = t1(i);

        % eISC for the small time windows
        % The data is stored in millisecond interval instead of fixations, hence this
        duration = zeros(tww,1);
        duration(:) = 1;

        % Initialize the heatmap variable
        fixation_heatmap = zeros(viewHeightPixel,viewWidthPixel,size(subjects,1));
        for s = 1:size(subjects,1) % subjects

            % Convert gaze coordinates to eISC format
            points = zeros(tww,2);
            points(:,1) = y_cut(t0(i):t1(i),s);
            points(:,2) = x_cut(t0(i):t1(i),s);

            % Calculate heatmap
            fixation_heatmap(:,:,s) = eISC_fixationHeatmap(points,kern,viewWidthPixel,viewHeightPixel,duration);

        end

        % Calculate ISC
        r(c,i) = eISC_spatialSimilarity(fixation_heatmap);

        % Calculate the avarage number of blinks (over subjects) for each
        % time points. If there were significant blink synchronizations, it
        % would be visible in the resultting time-series
        blink_tww = blink_cut(t0_i:t1_i,:);

        % Calculate how many subjects blinked within this tww and save the
        % proprotion of the total number of subjects
        blinks_avg(c,i) = sum(any(blink_tww))./size(subjects,1);

    end

    % Calculate the pupil size change for this cut by first normalizing the pupil size for each subject and then taking the average
    pupil_cut_norm = pupil_cut./mean(pupil_cut(1:tww*(twws_before),:));
    pupil_change(c,:) = mean(pupil_cut_norm,2,'omitnan');

    % Save results each round to protect for data loss
    results = struct;
    results.eisc = r;
    results.blink_synchronization = blinks_avg;
    results.pupil_change = pupil_change;
    save(output,'results');
    toc;  
end

% Normalize ISC to the first time windows (windows before the cut)
results.eisc_norm = results.eisc./mean(results.eisc(:,1:(twws_before)),2);

% Normalize blinks to the first time windows (windows before the cut)
results.blink_synchronization_norm = (results.blink_synchronization+1)./mean((results.blink_synchronization(:,1:(twws_before))+1),2);

%%  Save also as a csv file, Experiment 3
[path,name,ext] = fileparts(output);
writetable(array2table(results.eisc),sprintf('%s/scene_cut_effect_eisc_%s.csv',path,dataset),'WriteVariableNames',false);
writetable(array2table(results.eisc_norm),sprintf('%s/scene_cut_effect_eisc_normalized_%s.csv',path,dataset),'WriteVariableNames',false);
writetable(array2table(results.pupil_change),sprintf('%s/scene_cut_effect_pupil_%s.csv',path,dataset),'WriteVariableNames',false);
writetable(array2table(results.blink_synchronization),sprintf('%s/scene_cut_effect_blink_%s.csv',path,dataset),'WriteVariableNames',false);
writetable(array2table(results.blink_synchronization_norm),sprintf('%s/scene_cut_effect_blink_normalized_%s.csv',path,dataset),'WriteVariableNames',false);

%% Functions
function out = eISC_fixationHeatmap(points,kern,w,h,duration)
% out = eISC_fixationHeatmap(points,kern,w,h)
% ------------------------------------------------------------------------
% Creates heatmaps of gaze locations.
%
% Kernel dimensions should be odd so that the maximum of the kernel is
% in the middle pixel. This function will return an error if size in
% either direction is even. Kernels should also be square. If you do not
% want your kernel to be isotropically distributed you may still use a
% square matrix.
%
% Inputs:
% points:   Fixation locations in a fixations*coordinates matrix,
%           i.e. each row corresponds to x and y coordinates of fixation.
% kern:     External kernel or standard deviation of kernel in PIXELS.
% w:        Width of the image in pixels.
% h:        Height of the image in pixels.
%in
% Output:
% out:      Heatmap of fixations as the normalized sum of fixation kernels
%           at each fixation location.
%
% Version 0.01
% 10.4.2012 Juha Lahnakoski
% juha.lahnakoski@aalto.fi

%Defaults for the image size from a single experiment
if nargin<4 || isempty(h)
    h=1200;
end;
if nargin<3 || isempty(w)
    w=1920;
end;

if nargin>=2 && ~isempty(kern) && min(size(kern))==1
    %If kern-parameter is a plain number we use it as the radius and create
    %a new kernel
    kern=eISC_gaussKernel(kern);
    kernRadius=(size(kern,1)-1)/2;
elseif nargin>=2 && ~isempty(kern) && min(size(kern))>1
    %If kern-parameter is a matrix let's calculate the radius as the 
    kernRadius=(max(size(kern))-1)/2;
    
    %We do not want to use kernels where the size in any direction is even
    if min(mod(size(kern),2))==0%max(iseven(size(kern)))
        error('eISC_fixationHeatmap supports only kernels where the size along each dimension is odd.');
    end;
    
    %Also, we only want square kernels for now
    if min(size(kern))~=max(size(kern))
        error('eISC_fixationHeatmap supports only square kernels.');
    end;
        
else
    %If kern-parameter is not defined let's make the default kernel
    kern=eISC_gaussKernel;
    kernRadius=(size(kern,1)-1)/2;
end;

if size(kern,1)~=size(kern,2)
    %Give an error if the kernel is not square
    error('eISC currently allows only square kernels.');
end;

%Normalize the kernel just in case it wasn't already
kern=kern-min(kern(:));
kern=kern/max(kern(:));

%Create empty output
out=zeros(h,w);

%Find the non-NaN entries
idx=find(~isnan(points(:,1)).*~isnan(points(:,2)));
x=points(idx,2);
y=points(idx,1);
if nargin<5 || isempty(duration)
    duration=ones(length(x),1);
end

%Loop through fixations taking into account that some fixations will go
%over the image edges
for jj=1:length(x)
    %This calculates where the kernel ends
    xLo=round(x(jj))-kernRadius;
    xHi=round(x(jj))+kernRadius;
    yLo=round(y(jj))-kernRadius;
    yHi=round(y(jj))+kernRadius;
    %This calculates where we should cut the kernel if it goes outside the
    %image
    xKernLo=max(1,-xLo+2);
    xKernHi=min(2*kernRadius+1,2*kernRadius+1-(xHi-w));
    yKernLo=max(1,-yLo+2);
    yKernHi=min(2*kernRadius+1,2*kernRadius+1-(yHi-h));
    xLo=max(1,xLo);
    xHi=min(w,xHi);
    yLo=max(1,yLo);
    yHi=min(h,yHi);
    %And here the kernel is added to the image.
    out(yLo:yHi,xLo:xHi)=out(yLo:yHi,xLo:xHi)+duration(jj)*kern(yKernLo:yKernHi,xKernLo:xKernHi);
end;
%Normalization of the image
if sum(out(:))~=0
out=out/sum(out(:));
end;

end
function [r,cMat] = eISC_spatialSimilarity(fixMaps)
% [r,cMat] = eISC_spatialSimilarity(fixMaps)
% ------------------------------------------------------------------------
% Calculates the intersubject spatial correlations of fixation heatmaps.
% Alternative similarity or distance measures may be added later.
%
% Inputs:
% fixMaps:  width*height*subjects or height*width*subjects array of
%           fixation heatmaps, i.e. fixMaps(:,:,i) is the fixation maps of
%           the i'th subject.
%
% Output:
% r:        Mean pairwise spatial correlation coefficient across subjects
% cMat:     Upper triangle entries of the pairwise correlation matrix
%
% Version 0.01
% 10.4.2012 Juha Lahnakoski
% juha.lahnakoski@aalto.fi

%Calculate the correlations and select the upperm triangle entries without
%the diagonal (i.e. triu(...,1))
cMat=corr(reshape(fixMaps,[],size(fixMaps,3)));
cMat=cMat(find(triu(ones(size(cMat)),1)));

%Calculate the mean using the Fisher Z-transform first (atanh) and then
%transforming back.
r=tanh(nanmean(atanh(cMat)));

end
function out = eISC_gaussKernel(sigma,kernRadius,viewDistance,viewWidth,viewResolution,viewScaling)
% out = eISC_gaussKernel(sigma,kernRadius)
%-------------------------------------------------------------------------------
%	Creates a 2D gaussian kernel for plotting gaze heat maps.
% All inputs are optional, but it is advicable to change them to match the experimental
% parameters (view distance, width and resolution) for optimal results.
% Default parameters are based on a single experiment.
%
% Inputs:
% sigma:			Standard deviation of the fixation kernel. By default
%                   this will be used as deviation in number of PIXELS.
%                   If view all parameters are defined this is given in
%                   visual angle (in degrees) and calculated accordingly.
% kernRadius:		Radius of the kernel matrix (i.e. output will be a
%                   square matrix with 2*kerRadius+1 columns and rows)
% viewDistance: 	Distance from the subjects' eyes to the screen in
%                   arbitrary units.
% viewWidth:		Width of the image. Should in the same units as
%                   viewDistance.
% viewResolution:	Horizontal resolution of the image
% viewScaling:		Height of a pixel on the screen divided by the width
%                   of a pixel. This is NOT the same as vertical/horizontal
%                   resolution, but rather a measure of the shape of
%                   individual image pixels in case some sort of
%                   anisotropic scaling has been applied. The results
%                   should not change massively if this is omitted and
%                   isotropic voxels are assumed if the	images have not
%                   been stretched excessively.
%
% Ouput:
%	out:			The created 2D gaussian kernel matrix
%
% Version 0.01
% 10.4.2012 Juha Lahnakoski
% juha.lahnakoski@aalto.fi

if nargin==0 || isempty(sigma)
	%These are the default settings corresponding to sigma ~1 degree
	%at 34 cm viewing distance and pixel size ~1024pix/28cm
	sigma=34*tan(pi/180)*1024/28;
end;
%Here is the full definition of the sigma, if the parameters are given
if nargin>=5 && ~isempty(viewDistance) && ~isempty(viewWidth)...
             && ~isempty(viewResolution)
         
	sigma=viewDistance*tan(sigma*pi/180)*viewResolution/viewWidth;
    
end;
%Default radius of the kernel is 3 standard deviations
if nargin<2 || isempty(kernRadius)
	kernRadius=ceil(3*sigma);
end;
%Set the view scaling
if nargin<6 ||isempty(viewScaling)
	viewScaling=1;
end;

[X,Y]=ndgrid(viewScaling*-kernRadius:viewScaling:kernRadius*viewScaling,...
                         -kernRadius:kernRadius);
out = exp(-(X.^2+Y.^2)/(2*sigma^2)) / sqrt(2*pi*sigma^2);

%Normalize the kernel
out=out/sum(out(:));
end

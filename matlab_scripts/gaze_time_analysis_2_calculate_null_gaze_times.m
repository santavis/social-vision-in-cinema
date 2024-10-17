%% Gaze Time Analysis: Create shuffled data for each subject for permutations testing and then calculate null distributions for gaze times
%
% Process:
%       1. For each subject resample gaze coordinates and pupil size using
%       circular resampling so that the variation in the eye-tracking
%       measures does not change
%
%       2. Time series are cut between adjacent fixations so the number of fixations and
%       the fixations duration distribution do not change
%
%       3. 500 rounds should be enough
%
%       4. Null distibution estimation is repeated for each Experiment (1-3)
%
% Severi Santavirta 22.10.2023

%% Create circularly bootsrapped null data

nperm = 500;
dset = {'localizer','kasky','conjuring'}; % Experiments
counter = 0;

for d = 1:size(dset,2)
    
    % Get the eye-tracking data
    eyedata_files = find_files(sprintf('path/eyedata/subdata/%s/subjects',dset{d}),'*.mat');
    
    %%  Read fixations from preprocessed eye tracker data
    
    [~,subjects,~] = fileparts(eyedata_files);
    
    
    for I = 1:size(subjects,1)
        fprintf('%s: Reading fixations: %i/%i\n',dset{d},I,size(subjects,1));
    
        load(eyedata_files{I});
        if(I==1)
            fix_end = cell(size(subjects,1),1);
            data = zeros(size(round(subdata.fix_x),1),size(subjects,1),6,'single');
        end
        
        fix_end{I,1} = round(subdata.fixations.timestamps(:,2));
        data(:,I,1) = round(subdata.fix_x);
        data(:,I,2) = round(subdata.fix_y);
        data(:,I,3) = round(subdata.pupil);
        data(:,I,4) = round(subdata.saccades.ts);
        data(:,I,5) = round(subdata.blinks.ts);
        data(:,I,6) = round(subdata.fix_in_video_area);
    
    end
    
    %% Shuffle the data using circular bootstrap for each permutation
    
    % Define cut points for each subject
    cutpoints = zeros(nperm,size(subjects,1));
    for s = 1:size(subjects,1)
        counter = counter+1;
        rng(counter); % Different random cut points for each subject, and each permutation round
        cutpoints(:,s) = randi([1,size(fix_end{s,1},1)-1],nperm,1);
    end
    
    % Shuffle circularly
    for p = 1:nperm
        fprintf('%s: Shuffling data: %d/%d\n',dset{d},p,nperm);
        for s = 1:size(subjects,1)
            sfixend = fix_end{s};
            sdata = squeeze(data(:,s,:));
            scutpoint = cutpoints(p,s); % Circle fixations starting from this index to the front
    
            % First, need to define the time where the ts are cut
            t = sfixend(scutpoint);
    
            % Next, we can shuffle the data circularly
            data(:,s,:) = vertcat(data(t+1:end,s,:),data(1:t,s,:));
    
            % Finally, the fixation end timing needs to be updated for next
            % permutation
            block1 = sfixend(scutpoint+1:end); % Fixations that were shifted to beginning
            block2 = sfixend(1:scutpoint); % Fixations that were shifted to end
            block1 = block1-block2(end); % Correct the end times for the fixations that are shifted to the beginning
            block2 = block2+block1(end); % Correct the end times for the fixations that are shifted to the end
            fix_end{s} = vertcat(block1,block2); % Save new tming for the next round
        end

        % Save shuffled dataset after each round
        var = {'x','y','pupil','saccades','blinks','fix_in_video_area'};
        save(sprintf('path/eyedata/random_data/%s/%04d_%s_random_dataset.mat',dset{d},p,dset{d}),'data','subjects','fix_end','var','-v7.3');

    end
end

clear; clc;

%% Calculate the null gaze times for Experiment 1

% INPUT
dset = 'localizer'; % Experiment 1
frameArea_x = [153,873]; % X coordinates of the stimulus frame
frameArea_y = [97,672]; % Y coordinates of the stimulus frame
stimulus_frame = [576,720]; % Stimulus video size (stimulus frame size, not the area of the display, nor the true borderless video area)
npool = 4; % Number of parallel workers

%% Preps, Experiment 1
tic;

% Get the segmentation data
segmentation_catalog_path = sprintf('path/video_segmentation/%s/face/catalog.mat',dset);
segmentation_catalog = load(segmentation_catalog_path);
segmentation_catalog = cellstr(segmentation_catalog.catalog);
segmentation_data_folder = sprintf('path/video_segmentation/%s/face/data',dset);

% Update the catalog for "outside_videoarea" and "unknown categories
segmentation_catalog = vertcat(segmentation_catalog,'outside_videoarea','unknown');

% Get the broad class catalog
class_catalog = load(sprintf('path/video_segmentation/%s/face/catalog_class.mat',dset));
class_catalog = class_catalog.catalog_class;
cl = {'eyes';'mouth';'face';'person';'animal';'object';'background';'outside_video_area';'unknown'};

% Folder for the original stimulus files
stimulus_folder = sprintf('path/stimulus/%s/%s_eyetracking_mp4',dset,dset);

% Get the video names
vf = find_files(stimulus_folder,'*.mp4'); % !!Remember to check that the folder names are correctly numbered (with leading zeros)
[~,vids,~] = fileparts(vf);

% Get real trial information
eyedata = load(sprintf('path/eyedata/subdata/%s/subjects/L001.mat',dset));
trial = eyedata.subdata.trial_indices;

% Identify shuffled data
f = find_files(sprintf('path/eyedata/random_data/%s/',dset),'*.mat');

%% Run permutations, Experiment 1

% Open parallel pool
p = gcp('nocreate'); % If no pool, create new one.
if(isempty(p))
    p = parpool(npool);
end

% Identify already calculated permutations
fp = find_files(sprintf('path/gaze_object_detection/nulldata/%s/',dset),'*.mat');
np = size(fp,1);

% Loop over shuffled datasets
for r = (np+1):size(f,1)
    
    tic;
    % Get the shuffled data for this permutation round
    permdata = load(f{r});
    subjects = permdata.subjects;
    x = squeeze(permdata.data(:,:,1));
    y = squeeze(permdata.data(:,:,2));
    vidArea = squeeze(permdata.data(:,:,6));

    % Convert original pixel coordinates to the stimulus frame pixel
    % coordinates
    x = x-frameArea_x(1)+1; % correct coordinates for video area
    y = y-frameArea_y(1)+1; % correct coordinates for video area

    % Even when the coordinates are correct for the stimulus frame all of them may not fall within the stimulus frame. Hence, these
    % coordinates has to be outside video area as stated in the vidArea_sub
    % variable. To speed up the computations we first correct these outlier
    % coordinates within the stimulus frame and after the segmentation
    % information extraction we change the erroneous object category back
    % to to "outside_video_area"
    x(x>stimulus_frame(2)) = stimulus_frame(2);
    x(x<1) = 1;
    y(y>stimulus_frame(1)) = stimulus_frame(1);
    y(y<1) = 1;

    for s = 1:size(subjects,1)
        if(s==1)
            gaze = zeros(size(x,1),size(subjects,1),'single');
        end
        gaze(:,s) = sub2ind(stimulus_frame,y(:,s),x(:,s));
    end
    
    % Loop through videos and extract gaze_objects
    gaze_object = cell(size(gaze,2),1);
    parfor J = 1:size(vids,1)
        fprintf('Permutation %d: Extracting gaze categories: vid: %s\n',r,vids{J});
        
        % Gaze coordinates and video area for the video
        gaze_trial = gaze(trial==J,:);
        vidArea_trial = vidArea(trial==J,:);
    
        % The stimulus video clips are <100ms longer than the eye-tracking
        % trial length due to avi to mp4 conversion innaccuracy. This is
        % corrected by excluding the last few frames to match the timings.
    
        % Read the stimulus video for checking the frame rate
        vidIn = VideoReader(sprintf('%s/%s.mp4',stimulus_folder,vids{J}));
    
        % Get the right frame indices
        idx = (1:1:floor((vidIn.FrameRate*size(gaze_trial,1)/1000)))';
    
        % Get the frame time points
        t = (40:40:floor((vidIn.FrameRate*size(gaze_trial,1)/1000))*40)';
    
        % For few clips it is possible that the gaze data ends between
        % frames
        if(t(end)<size(gaze_trial,1))
            idx = [idx;idx(end)+1];
            t = [t;size(gaze_trial,1)];
        end
    
        % Loop over the frames
        t0 = 1; % Time frame start point
        gaze_object_trial = [];
        for K = 1:size(idx,1)
            t1=t(K); % Time frame end point
    
            % Gaze positions and video area within the frame
            gaze_frame = gaze_trial(t0:t1,:);
            vidArea_frame = vidArea_trial(t0:t1,:);
    
            % Load the segmentation information of the frame
            digit = numel(num2str(length(idx)));
            formatStr = sprintf('%s/%s/frame_%%0%dd.mat',segmentation_data_folder,vids{J},digit);
            frame = url_parLoad(sprintf(formatStr,K));
    
            % Get prediction for all gaze postions
            gaze_object_frame = frame.mask(gaze_frame);
        
            % Some pixel were outside video are, correct the category index
            gaze_object_frame(~vidArea_frame) = 138;

            % Change zeros to 139 (unknown category for the gaze position)
            gaze_object_frame(gaze_object_frame==0) = 139;
    
            % Collect the object information from the frame
            gaze_object_trial = vertcat(gaze_object_trial,gaze_object_frame);
            t0 = t1+1; % Starting point for the next frame
        end
    
        % Collect the object information from the video
        gaze_object{J,1} = gaze_object_trial;
    end
    
    % Create one time series from the gaze_object
    gaze_ts = [];
    for I = 1:size(gaze_object,1)
        gaze_ts = vertcat(gaze_ts,gaze_object{I,1});
    end
    
    % Count gaze instances for all categories
    counts_category = zeros(size(segmentation_catalog,1),size(gaze_ts,2));
    for o = 1:size(segmentation_catalog,1)
        counts_category(o,:) = sum(gaze_ts==o);
    end

    % Tranform instances to proportional time
    time_category = counts_category./size(gaze_ts,1);

    % Tranform categories to broad classes 
    gaze_class = gaze_ts;
    for c = 1:size(cl,1)
        idxClass = find(class_catalog.class_idx==(c));
        for cc = 1:size(idxClass,1)
            gaze_class(gaze_ts==idxClass(cc)) = c;
        end
    end

    % Count gaze instances for all broad classes
    counts_class = zeros(size(cl,1),size(gaze_class,2));
    for c = 1:size(cl,1)
        counts_class(c,:) = sum(gaze_class==c);
    end

    % Tranform instances to proportional time
    time_class = counts_class./size(gaze_class,1);

    % Save results after each round, if something happens
    save(sprintf('path/gaze_object_detection/nulldata/%s/%04d_%s_random_times.mat',dset,r,dset),'time_class','time_category');
    toc;
end

clear; clc;

%% Calculate the null gaze times for Experiment 2

% Input
dset = 'kasky'; % Exepriment 2
frameArea_x = [0,1024]; % X coordinates of the stimulus frame
frameArea_y = [25,744]; % Y coordinates of the stimulus frame
stimulus_frame = [720,1280]; % Stimulus video size (display size: 768,1024)
npool = 4; % Number of parallel workers

%% Preps, Experiment 2
tic;

% Get the segmentation data
segmentation_catalog_path = sprintf('path/video_segmentation/%s/face/catalog.mat',dset);
segmentation_catalog = load(segmentation_catalog_path);
segmentation_catalog = cellstr(segmentation_catalog.catalog);
segmentation_data_folder = sprintf('path/video_segmentation/%s/face/data',dset);

% Update the catalog for "outside_videoarea" and "unknown categories
segmentation_catalog = vertcat(segmentation_catalog,'outside_videoarea','unknown');

% Get the broad class catalog
class_catalog = load(sprintf('path/video_segmentation/%s/face/catalog_class.mat',dset));
class_catalog = class_catalog.catalog_class;
cl = {'eyes';'mouth';'face';'person';'animal';'object';'background';'outside_video_area';'unknown'};

% Folder for the original stimulus files
stimulus_folder = sprintf('path/stimulus/%s/%s_eyetracking_mp4',dset,dset);

% Get the video names
vf = find_files(stimulus_folder,'*.mp4'); % !!Remember to check that the folder names are correctly numbered (with leading zeros)
[~,vids,~] = fileparts(vf);

% Get real trial information
eyedata = load(sprintf('path/eyedata/subdata/%s/subjects/K01.mat',dset));
trial = eyedata.subdata.trial_indices;

% Identify shuffled data
f = find_files(sprintf('path/eyedata/random_data/%s/',dset),'*.mat');

%% Run permutations, Experiment 2

% Open parallel pool
p = gcp('nocreate'); % If no pool, create new one.
if(isempty(p))
    p = parpool(npool);
end

% Identify already calculated permutations
fp = find_files(sprintf('path/gaze_object_detection/nulldata/%s/',dset),'*.mat');
np = size(fp,1);

% Loop over shuffled datasets
for r = (np+1):size(f,1)
    
    tic;
    % Get the shuffled data for this permutation round
    permdata = load(f{r});
    subjects = permdata.subjects;
    x = squeeze(permdata.data(:,:,1));
    y = squeeze(permdata.data(:,:,2));
    vidArea = squeeze(permdata.data(:,:,6));

    % Convert original pixel coordinates to the stimulus frame pixel
    % coordinates
    % KASKY: X coordinates are already in correct space since the stimulus
    % frame is bigger than the display in x axis
    y = y-frameArea_y(1)+1; % correct coordinates for video area

    % Even when the coordinates are correct for the stimulus frame all of them may not fall within the stimulus frame. Hence, these
    % coordinates has to be outside video area as stated in the vidArea_sub
    % variable. To speed up the computations we first correct these outlier
    % coordinates within the stimulus frame and after the segmentation
    % information extraction we change the erroneous object category back
    % to to "outside_video_area"
    x(x>1024) = 1024;
    x(x<1) = 1;
    y(y>stimulus_frame(1)) = stimulus_frame(1);
    y(y<1) = 1;

    for s = 1:size(subjects,1)
        if(s==1)
            gaze = zeros(size(x,1),size(subjects,1),'single');
        end
        gaze(:,s) = sub2ind(stimulus_frame,y(:,s),x(:,s));
    end
    
    % Loop through videos and extract gaze_objects
    gaze_object = cell(size(gaze,2),1);
    parfor J = 1:size(vids,1)
        fprintf('Permutation %d: Extracting gaze categories: vid: %s\n',r,vids{J});
        
        % Gaze coordinates and video area for the video
        gaze_trial = gaze(trial==J,:);
        vidArea_trial = vidArea(trial==J,:);
    
        % KASKY: The stimulus video clips are 0,04 - 1,3sec longer than the eye-tracking
        % trial length due to avi to mp4 conversion innaccuracy. This is
        % corrected by excluding the last few frames to match the timings.

        % Read the stimulus video for checking the frame rate
        vidIn = VideoReader(sprintf('%s/%s.mp4',stimulus_folder,vids{J}));
    
        % Get the right frame indices
        idx = (1:1:floor((vidIn.FrameRate*size(gaze_trial,1)/1000)))';
    
        % Get the frame time points
        t = (40:40:floor((vidIn.FrameRate*size(gaze_trial,1)/1000))*40)';
    
        % For few clips it is possible that the gaze data ends between
        % frames
        if(t(end)<size(gaze_trial,1))
            idx = [idx;idx(end)+1];
            t = [t;size(gaze_trial,1)];
        end
    
        % Loop over the frames
        t0 = 1; % Time frame start point
        gaze_object_trial = [];
        for K = 1:size(idx,1)
            t1=t(K); % Time frame end point
    
            % Gaze positions and video area within the frame
            gaze_frame = gaze_trial(t0:t1,:);
            vidArea_frame = vidArea_trial(t0:t1,:);
    
            % Load the segmentation information of the frame
            digit = numel(num2str(length(idx)));
            formatStr = sprintf('%s/%s/frame_%%0%dd.mat',segmentation_data_folder,vids{J},digit);
            
            % KASKY: For the last frame of the last clip, we do not have segmentation
            % data, to correct this missmatch we duplicate the last segmented
            % frame (the second last frame)
            if(J==26 && K == 6408)
                frame = url_parLoad(sprintf(formatStr,idx(K-1)));
            else
                frame = url_parLoad(sprintf(formatStr,idx(K)));
            end

            % KASKY, the frame does not fit exaclty to the presentation
            % screen in the x axis (frame size 1280, while display size 1024).
            % The tails are left out from the display
            frame.mask = frame.mask(:,(129:(129+1023)));
    
            % Get prediction for all gaze postions
            gaze_object_frame = frame.mask(gaze_frame);
        
            % Some pixel were outside video are, correct the category index
            gaze_object_frame(~vidArea_frame) = 138;

            % Change zeros to 139 (unknown category for the gaze position)
            gaze_object_frame(gaze_object_frame==0) = 139;
    
            % Collect the object information from the frame
            gaze_object_trial = vertcat(gaze_object_trial,gaze_object_frame);
            t0 = t1+1; % Starting point for the next frame
        end
    
        % Collect the object information from the video
        gaze_object{J,1} = gaze_object_trial;
    end
    
    % Create one time series from the gaze_object
    gaze_ts = [];
    for I = 1:size(gaze_object,1)
        gaze_ts = vertcat(gaze_ts,gaze_object{I,1});
    end
    
    % Count gaze instances for all categories
    counts_category = zeros(size(segmentation_catalog,1),size(gaze_ts,2));
    for o = 1:size(segmentation_catalog,1)
        counts_category(o,:) = sum(gaze_ts==o);
    end

    % Tranform instances to proportional time
    time_category = counts_category./size(gaze_ts,1);

    % Tranform categories to broad classes 
    gaze_class = gaze_ts;
    for c = 1:size(cl,1)
        idxClass = find(class_catalog.class_idx==(c));
        for cc = 1:size(idxClass,1)
            gaze_class(gaze_ts==idxClass(cc)) = c;
        end
    end

    % Count gaze instances for all broad classes
    counts_class = zeros(size(cl,1),size(gaze_class,2));
    for c = 1:size(cl,1)
        counts_class(c,:) = sum(gaze_class==c);
    end

    % Tranform instances to proportional time
    time_class = counts_class./size(gaze_class,1);

    % Save results after each round, if something happens
    save(sprintf('path/gaze_object_detection/nulldata/%s/%04d_%s_random_times.mat',dset,r,dset),'time_class','time_category');
    toc;
end

clear; clc;

%% Calculate the null gaze times for Experiment 3

% Input
dset = 'conjuring'; % Experiment 3
frameArea_x = [13,1012]; % X coordinates of the stimulus frame
frameArea_y = [10,759]; % Y coordinates of the stimulus frame
stimulus_frame = [750,1000]; % Stimulus video size (stimulus frame size, not the area of the display, nor the true borderless video area)
npool = 2; % Number of parallel workers

%% Preps, Experiment 3
tic;

% Get the segmentation data
segmentation_catalog_path = sprintf('path/video_segmentation/%s/face/catalog.mat',dset);
segmentation_catalog = load(segmentation_catalog_path);
segmentation_catalog = cellstr(segmentation_catalog.catalog);
segmentation_data_folder = sprintf('path/video_segmentation/%s/face/data',dset);

% Update the catalog for "outside_videoarea" and "unknown categories
segmentation_catalog = vertcat(segmentation_catalog,'outside_videoarea','unknown');

% Get the broad class catalog
class_catalog = load(sprintf('path/video_segmentation/%s/face/catalog_class.mat',dset));
class_catalog = class_catalog.catalog_class;
cl = {'eyes';'mouth';'face';'person';'animal';'object';'background';'outside_video_area';'unknown'};

% Folder for the original stimulus files
stimulus_folder = sprintf('path/stimulus/%s/%s_eyetracking_mp4',dset,dset);

% Get the video names
vf = find_files(stimulus_folder,'*.mp4'); % !!Remember to check that the folder names are correctly numbered (with leading zeros)
[~,vids,~] = fileparts(vf);

% Get real trial information
eyedata = load(sprintf('path/eyedata/subdata/%s/subjects/C01.mat',dset));
trial = eyedata.subdata.trial_indices;

% Identify shuffled data
f = find_files(sprintf('path/eyedata/random_data/%s/',dset),'*.mat');

%% Run permutations, Experiment 3

% Open parallel pool
p = gcp('nocreate'); % If no pool, do not create new one.
if(isempty(p))
    p = parpool(npool);
end

% Identify already calculated permutations
fp = find_files(sprintf('path/gaze_object_detection/nulldata/%s/',dset),'*.mat');
np = size(fp,1);

% Loop over shuffled datasets
for r = (np+1):size(f,1)
    
    tic;
    % Get the shuffled data for this permutation round
    permdata = load(f{r});
    subjects = permdata.subjects;
    x = squeeze(permdata.data(:,:,1));
    y = squeeze(permdata.data(:,:,2));
    vidArea = squeeze(permdata.data(:,:,6));

    % Convert original pixel coordinates to the stimulus frame pixel
    % coordinates
    x = x-frameArea_x(1)+1; % correct coordinates for video area
    y = y-frameArea_y(1)+1; % correct coordinates for video area

    % Even when the coordinates are correct for the stimulus frame all of them may not fall within the stimulus frame. Hence, these
    % coordinates has to be outside video area as stated in the vidArea_sub
    % variable. To speed up the computations we first correct these outlier
    % coordinates within the stimulus frame and after the segmentation
    % information extraction we change the erroneous object category back
    % to "outside_video_area"
    x(x>stimulus_frame(2)) = stimulus_frame(2);
    x(x<1) = 1;
    y(y>stimulus_frame(1)) = stimulus_frame(1);
    y(y<1) = 1;

    for s = 1:size(subjects,1)
        if(s==1)
            gaze = zeros(size(x,1),size(subjects,1),'single');
        end
        gaze(:,s) = sub2ind(stimulus_frame,y(:,s),x(:,s));
    end
    
    % Loop through videos and extract gaze_objects
    gaze_object = cell(size(gaze,2),1);
    parfor J = 1:size(vids,1)
        fprintf('Permutation %d: Extracting gaze categories: vid: %s\n',r,vids{J});
        
        % Gaze coordinates and video area for the video
        gaze_trial = gaze(trial==J,:);
        vidArea_trial = vidArea(trial==J,:);
    
        % CONJURING: The stimulus video clips are 0,04 - 1,1sec longer than the eye-tracking
        % trial length due to avi to mp4 conversion innaccuracy. One video clip
        % is ~6.5sec longer than the eytracking data (for eaytracking the clip were cut shorter). These are
        % corrected by excluding the last few frames to match the timings.
    
        % Read the stimulus video for checking the frame rate
        vidIn = VideoReader(sprintf('%s/%s.mp4',stimulus_folder,vids{J}));
    
        % Get the right frame indices
        idx = (1:1:floor((vidIn.FrameRate*size(gaze_trial,1)/1000)))';
    
        % Get the frame time points
        t = (40:40:floor((vidIn.FrameRate*size(gaze_trial,1)/1000))*40)';
    
        % For few clips it is possible that the gaze data ends between
        % frames
        if(t(end)<size(gaze_trial,1))
            idx = [idx;idx(end)+1];
            t = [t;size(gaze_trial,1)];
        end
    
        % Loop over the frames
        t0 = 1; % Time frame start point
        gaze_object_trial = [];
        for K = 1:size(idx,1)
            t1=t(K); % Time frame end point
    
            % Gaze positions and video area within the frame
            gaze_frame = gaze_trial(t0:t1,:);
            vidArea_frame = vidArea_trial(t0:t1,:);
    
            % Load the segmentation information of the frame
            digit = numel(num2str(length(idx)));
            formatStr = sprintf('%s/%s/frame_%%0%dd.mat',segmentation_data_folder,vids{J},digit);
            frame = url_parLoad(sprintf(formatStr,K));
    
            % Get prediction for all gaze postions
            gaze_object_frame = frame.mask(gaze_frame);
        
            % Some pixel were outside video are, correct the category index
            gaze_object_frame(~vidArea_frame) = 138;

            % Change zeros to 139 (unknown category for the gaze position)
            gaze_object_frame(gaze_object_frame==0) = 139;
    
            % Collect the object information from the frame
            gaze_object_trial = vertcat(gaze_object_trial,gaze_object_frame);
            t0 = t1+1; % Starting point for the next frame
        end
    
        % Collect the object information from the video
        gaze_object{J,1} = gaze_object_trial;
    end
    
    % Create one time series from the gaze_object
    gaze_ts = [];
    for I = 1:size(gaze_object,1)
        gaze_ts = vertcat(gaze_ts,gaze_object{I,1});
    end
    
    % Count gaze instances for all categories
    counts_category = zeros(size(segmentation_catalog,1),size(gaze_ts,2));
    for o = 1:size(segmentation_catalog,1)
        counts_category(o,:) = sum(gaze_ts==o);
    end

    % Tranform instances to proportional time
    time_category = counts_category./size(gaze_ts,1);

    % Tranform categories to broad classes 
    gaze_class = gaze_ts;
    for c = 1:size(cl,1)
        idxClass = find(class_catalog.class_idx==(c));
        for cc = 1:size(idxClass,1)
            gaze_class(gaze_ts==idxClass(cc)) = c;
        end
    end

    % Count gaze instances for all broad classes
    counts_class = zeros(size(cl,1),size(gaze_class,2));
    for c = 1:size(cl,1)
        counts_class(c,:) = sum(gaze_class==c);
    end

    % Tranform instances to proportional time
    time_class = counts_class./size(gaze_class,1);

    % Save results after each round, if something happens
    save(sprintf('path/gaze_object_detection/nulldata/%s/%04d_%s_random_times.mat',dset,r,dset),'time_class','time_category');
    toc;
end

%% Functions

function var = url_parLoad(path)
    var = load(path);
end



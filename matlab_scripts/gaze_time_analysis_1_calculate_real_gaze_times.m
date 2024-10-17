%% Gaze Time Analysis: Extract what the subjects are watching at any given time point.
%
% Procedure:
%   1. Create a 1 ms time series for each object whther a subject is gazing the category or not (1 = watching, 0 = not watching)
%   2. Calculate the total watching time for each broad class for each subject
%
% Severi Santavirta 24.10.2023

%% INPUT, Experiment 1

dset = 'localizer'; % localizer (Exp. 1), kasky (Exp. 2), conjuring (Exp. 3)
frameArea_x = [153,873]; % X coordinates of the stimulus frame
frameArea_y = [97,672]; % Y coordinates of the stimulus frame
stimulus_frame = [576,720]; % Stimulus video size (stimulus frame size, not the area of the display, nor the true borderless video area)
npool = 3; % Number of parallel workers

%% Preps, Experiment 1
tic;

% Get the eye-tracking data
eyedata_files = find_files(sprintf('path/eyedata/subdata/%s/subjects',dset),'*.mat');

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


%%  Read fixations from preprocessed eye tracker data, Experiment 1
tic;
[~,subjects,~] = fileparts(eyedata_files);
for I = 1:size(subjects,1)
    fprintf('Reading fixations: %i/%i\n',I,size(subjects,1));

    load(eyedata_files{I});
    x_sub = round(subdata.fix_x);
    y_sub = round(subdata.fix_y);
    
    % Collect inside video area information that is used later 
    vidArea_sub = subdata.fix_in_video_area;
    
    % Convert original pixel coordinates to the stimulus frame pixel
    % coordinates
    x_sub = x_sub-frameArea_x(1)+1; % correct coordinates for video area
    y_sub = y_sub-frameArea_y(1)+1; % correct coordinates for video area

    if(I==1)
        vidArea = zeros(size(x_sub,1),size(subjects,1),'single');
        gaze = zeros(size(x_sub,1),size(subjects,1),'single');

        trial = subdata.trial_indices;
    end
    
    vidArea(:,I) = vidArea_sub;
    
    % Even when the coordinates are correct for the stimulus frame all of them may not fall within the stimulus frame. Hence, these
    % coordinates has to be outside video area as stated in the vidArea_sub
    % variable. To speed up the computations we first correct these outlier
    % coordinates within the stimulus frame and after the segmentation
    % information extraction we change the erroneous object category back
    % to to "outside_video_area"
    if(max(x_sub)>stimulus_frame(2))
        fprintf('Max x coordinate is out of bounds! Value: %d\n',max(x_sub));
        x_sub(x_sub>stimulus_frame(2)) = stimulus_frame(2);
    end
    if(min(x_sub)<1)
        fprintf('Min x coordinate is out of bounds! Value: %d\n',min(x_sub));
        x_sub(x_sub<1) = 1;
    end
    if(max(y_sub)>stimulus_frame(1))
        fprintf('Max y coordinate is out of bounds! Value: %d\n',max(y_sub));
        y_sub(y_sub>stimulus_frame(1)) = stimulus_frame(1);
    end
    if(min(y_sub)<1)
        fprintf('Min y coordinate is out of bounds! Value: %d\n',min(y_sub));
        y_sub(y_sub<1) = 1;
    end

    % Convert gaze coordinates to to indices
    gaze(:,I) = sub2ind(stimulus_frame,y_sub,x_sub);
end

%% Extreact gaze objects (for each millisecond), Experiment 1
p = gcp('nocreate'); % If no pool, create new one.
if(isempty(p))
    p = parpool(npool);
end
% Loop through videos
gaze_object = cell(size(gaze,2),1);
parfor J = 1:size(vids,1)
    fprintf('Extracting gaze categories: vid: %s\n',vids{J});
    
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

%% Calculate total watching times for each category and also total times for each broad class, Experiment 1

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
time_class_mean = horzcat(array2table(mean(time_class,2)),array2table(cl));

% Save the subjectwise total time results
tbl_class = horzcat(array2table(cl),array2table(time_class));
tbl_class.Properties.VariableNames = [{'class'};subjects];
writetable(tbl_class,sprintf('path/gaze_object_detection/total_gaze_time/time_class_%s.csv',dset));
tbl_category = horzcat(array2table(segmentation_catalog),array2table(time_category));
tbl_category.Properties.VariableNames = [{'category'};subjects];
writetable(tbl_class,sprintf('path/gaze_object_detection/total_gaze_time/time_category_%s.csv',dset));

toc;

%% Save results for each subject, Experiment 1
for I = 1:size(subjects,1)
    fprintf('Saving results: sub %d/%d\n',I,size(subjects,1));

    load(eyedata_files{I});
    
    result_category = horzcat(segmentation_catalog,array2table(time_category(:,I)));
    result_category.Properties.VariableNames = {'object','time_percentage'};

    result_class = horzcat(array2table(cl),array2table(time_class(:,I)));
    result_class.Properties.VariableNames = {'class','time_percentage'};

    % Store the results
    subdata.gaze_object = gaze_ts(:,I);
    subdata.gaze_object_catalog = segmentation_catalog;
    subdata.gaze_object_time = result_category;
    subdata.gaze_class = gaze_class(:,I);
    subdata.gaze_class_catalog = cl;
    subdata.gaze_class_time = result_class;

    % Save the file
    save(eyedata_files{I},'subdata');
end
toc;

clear; clc;

%% INPUT, Experiment 2

dset = 'kasky'; % localizer (Exp. 1), kasky (Exp. 2), conjuring (Exp. 3)
frameArea_x = [0,1024]; % X coordinates of the stimulus frame
frameArea_y = [25,744]; % Y coordinates of the stimulus frame
stimulus_frame = [720,1280]; % Stimulus video size (display size: 768,1024)
npool = 2; % Number of parallel workers

%% Preps, Experiment 2
tic;
% Get the eye-tracking data
eyedata_files = find_files(sprintf('path/eyedata/subdata/%s/subjects',dset),'*.mat');

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

%%  Read fixations from preprocessed eye tracker data, Experiment 2
tic;
[~,subjects,~] = fileparts(eyedata_files);
for I = 1:size(subjects,1)
    fprintf('Reading fixations: %i/%i\n',I,size(subjects,1));

    load(eyedata_files{I});
    x_sub = round(subdata.fix_x);
    y_sub = round(subdata.fix_y);
    
    % Collect inside video area information that is used later 
    vidArea_sub = subdata.fix_in_video_area;
    
    % Convert original pixel coordinates to the stimulus frame pixel
    % coordinates
    % KASKY: X coordinates are already in correct space since the stimulus
    % frame is bigger than the display in x axis
    y_sub = y_sub-frameArea_y(1)+1; % correct coordinates for video area

    
    if(I==1)
        vidArea = zeros(size(x_sub,1),size(subjects,1),'single');
        gaze = zeros(size(x_sub,1),size(subjects,1),'single');

        trial = subdata.trial_indices;
    end
    
    vidArea(:,I) = vidArea_sub;
    
    % Even when the coordinates are correct for the stimulus frame, all of them may not fall within the stimulus frame. Hence, these
    % coordinates has to be outside video area as stated in the vidArea_sub
    % variable. To speed up the computations we first correct these outlier
    % coordinates within the stimulus frame and after the segmentation
    % information extraction we change the erroneous object category back
    % to "outside_video_area"
    if(max(x_sub)>1024)
        fprintf('Max x coordinate is out of bounds! Value: %d\n',max(x_sub));
        x_sub(x_sub>1024) = 1024;
    end
    if(min(x_sub)<1)
        fprintf('Min x coordinate is out of bounds! Value: %d\n',min(x_sub));
        x_sub(x_sub<1) = 1;
    end
    if(max(y_sub)>stimulus_frame(1))
        fprintf('Max y coordinate is out of bounds! Value: %d\n',max(y_sub));
        y_sub(y_sub>stimulus_frame(1)) = stimulus_frame(1);
    end
    if(min(y_sub)<1)
        fprintf('Min y coordinate is out of bounds! Value: %d\n',min(y_sub));
        y_sub(y_sub<1) = 1;
    end

    % Convert gaze coordinates to to indices
    gaze(:,I) = sub2ind(stimulus_frame,y_sub,x_sub);
end

%% Extract gaze objects (for each millisecond), Experiment 2
p = gcp('nocreate'); % If no pool, create new one.
if(isempty(p))
    p = parpool(npool);
end
% Loop through videos
gaze_object = cell(size(gaze,2),1);
parfor J = 1:size(vids,1)
    fprintf('Extracting gaze categories: vid: %s\n',vids{J});
    
    % Gaze coordinates and video area for the video
    gaze_trial = gaze(trial==J,:);
    vidArea_trial = vidArea(trial==J,:);

    % KASKY: The stimulus video clips are 0,04 - 1,3sec longer than the eye-tracking
    % trial length due to avi to mp4 conversion innaccuracy. This is
    % corrected by excluding the last few frames to match the timings.

    % Read the stimulus video for checking the frame rate
    vidIn = VideoReader(sprintf('%s/%s.mp4',stimulus_folder,vids{J}));

    % Get the segmentation frame indices
    idx = (1:1:floor((vidIn.FrameRate*size(gaze_trial,1)/1000)))';

    % Get the video time points for frames
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
        
        % KASKY, the frame does not fit exactly to the presentation
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

%% Calculate total watching times for each category and also total times for each broad class, Experiment 2

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
time_class_mean = horzcat(array2table(mean(time_class,2)),array2table(cl));

% Save the subjectwise total time results
tbl_class = horzcat(array2table(cl),array2table(time_class));
tbl_class.Properties.VariableNames = [{'class'};subjects];
writetable(tbl_class,sprintf('path/gaze_object_detection/total_gaze_time/time_class_%s.csv',dset));
tbl_category = horzcat(array2table(segmentation_catalog),array2table(time_category));
tbl_category.Properties.VariableNames = [{'category'};subjects];
writetable(tbl_class,sprintf('path/gaze_object_detection/total_gaze_time/time_category_%s.csv',dset));

toc;

%% Save results for each subject, Experiment 2
for I = 1:size(subjects,1)
    fprintf('Saving results: sub %d/%d\n',I,size(subjects,1));

    load(eyedata_files{I});
    
    result_category = horzcat(segmentation_catalog,array2table(time_category(:,I)));
    result_category.Properties.VariableNames = {'object','time_percentage'};

    result_class = horzcat(array2table(cl),array2table(time_class(:,I)));
    result_class.Properties.VariableNames = {'class','time_percentage'};

    % Store the results
    subdata.gaze_object = gaze_ts(:,I);
    subdata.gaze_object_catalog = segmentation_catalog;
    subdata.gaze_object_time = result_category;
    subdata.gaze_class = gaze_class(:,I);
    subdata.gaze_class_catalog = cl;
    subdata.gaze_class_time = result_class;

    % Save the file
    save(eyedata_files{I},'subdata');
end
toc;

clear; clc;

%% INPUT, Experiment 3

dset = 'conjuring'; % localizer (Exp. 1), kasky (Exp. 2), conjuring (Exp. 3)
frameArea_x = [13,1012]; % X coordinates of the stimulus frame
frameArea_y = [10,759]; % Y coordinates of the stimulus frame
stimulus_frame = [750,1000]; % Stimulus video size (stimulus frame size, not the area of the display, nor the true borderless video area)
npool = 3; % Number of workers

%% Preps, Experiment 3
tic;
% Get the eye-tracking data
eyedata_files = find_files(sprintf('path/eyedata/subdata/%s/subjects',dset),'*.mat');

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

%%  Read fixations from preprocessed eye tracker data, Experiment 3
tic;
[~,subjects,~] = fileparts(eyedata_files);
for I = 1:size(subjects,1)
    fprintf('Reading fixations: %i/%i\n',I,size(subjects,1));

    load(eyedata_files{I});
    x_sub = round(subdata.fix_x);
    y_sub = round(subdata.fix_y);
    
    % Collect inside video area information that is used later 
    vidArea_sub = subdata.fix_in_video_area;
    
    % Convert original pixel coordinates to the stimulus frame pixel
    % coordinates
    x_sub = x_sub-frameArea_x(1)+1; % correct coordinates for video area
    y_sub = y_sub-frameArea_y(1)+1; % correct coordinates for video area

    if(I==1)
        vidArea = zeros(size(x_sub,1),size(subjects,1),'single');
        gaze = zeros(size(x_sub,1),size(subjects,1),'single');

        trial = subdata.trial_indices;
    end
    
    vidArea(:,I) = vidArea_sub;
    
    % Even when the coordinates are correct for the stimulus frame all of them may not fall within the stimulus frame. Hence, these
    % coordinates has to be outside video area as stated in the vidArea_sub
    % variable. To speed up the computations we first correct these outlier
    % coordinates within the stimulus frame and after the segmentation
    % information extraction we change the erroneous object category back
    % to "outside_video_area"
    if(max(x_sub)>stimulus_frame(2))
        fprintf('Max x coordinate is out of bounds! Value: %d\n',max(x_sub));
        x_sub(x_sub>stimulus_frame(2)) = stimulus_frame(2);
    end
    if(min(x_sub)<1)
        fprintf('Min x coordinate is out of bounds! Value: %d\n',min(x_sub));
        x_sub(x_sub<1) = 1;
    end
    if(max(y_sub)>stimulus_frame(1))
        fprintf('Max y coordinate is out of bounds! Value: %d\n',max(y_sub));
        y_sub(y_sub>stimulus_frame(1)) = stimulus_frame(1);
    end
    if(min(y_sub)<1)
        fprintf('Min y coordinate is out of bounds! Value: %d\n',min(y_sub));
        y_sub(y_sub<1) = 1;
    end

    % Convert gaze coordinates to to indices
    gaze(:,I) = sub2ind(stimulus_frame,y_sub,x_sub);
end

%% Extreact gaze objects (for each millisecond), Experiment 3
p = gcp('nocreate'); % If no pool, create new one.
if(isempty(p))
    p = parpool(npool);
end
% Loop through videos
gaze_object = cell(size(gaze,2),1);
parfor J = 1:size(vids,1)
    fprintf('Extracting gaze categories: vid: %s\n',vids{J});
    
    % Gaze coordinates and video area for the video
    gaze_trial = gaze(trial==J,:);
    vidArea_trial = vidArea(trial==J,:);

    % CONJURING: The stimulus video clips are 0,04 - 1,1sec longer than the eye-tracking
    % trial length due to avi to mp4 conversion innaccuracy. One video clip
    % is ~6.5sec longer than the eytracking data (for eyetracking the clip was cut shorter). These are
    % corrected by excluding the last few frames to match the timings.

    % Read the stimulus video for checking the frame rate
    vidIn = VideoReader(sprintf('%s/%s.mp4',stimulus_folder,vids{J}));

    % Get the segmentation frame indices
    idx = (1:1:floor((vidIn.FrameRate*size(gaze_trial,1)/1000)))';

    % Get the video time points for frames
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
        frame = url_parLoad(sprintf(formatStr,idx(K)));
        
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

%% Calculate total watching times for each category and also total times for each broad class, Experiment 3

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
time_class_mean = horzcat(array2table(mean(time_class,2)),array2table(cl));

% Save the subjectwise total time results
tbl_class = horzcat(array2table(cl),array2table(time_class));
tbl_class.Properties.VariableNames = [{'class'};subjects];
writetable(tbl_class,sprintf('path/gaze_object_detection/total_gaze_time/time_class_%s.csv',dset));
tbl_category = horzcat(array2table(segmentation_catalog),array2table(time_category));
tbl_category.Properties.VariableNames = [{'category'};subjects];
writetable(tbl_class,sprintf('path/gaze_object_detection/total_gaze_time/time_category_%s.csv',dset));
toc;

%% Save results for each subject, Experiment 3
for I = 1:size(subjects,1)
    fprintf('Saving results: sub %d/%d\n',I,size(subjects,1));

    load(eyedata_files{I});
    
    result_category = horzcat(segmentation_catalog,array2table(time_category(:,I)));
    result_category.Properties.VariableNames = {'object','time_percentage'};

    result_class = horzcat(array2table(cl),array2table(time_class(:,I)));
    result_class.Properties.VariableNames = {'class','time_percentage'};

    % Store the results
    subdata.gaze_object = gaze_ts(:,I);
    subdata.gaze_object_catalog = segmentation_catalog;
    subdata.gaze_object_time = result_category;
    subdata.gaze_class = gaze_class(:,I);
    subdata.gaze_class_catalog = cl;
    subdata.gaze_class_time = result_class;

    % Save the file
    save(eyedata_files{I},'subdata');
end
toc;

%% Functions

function var = url_parLoad(path)
    var = load(path);
end

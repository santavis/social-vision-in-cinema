%% Gigatrack: Panoptic segmentation and face detection quality control
%
% There is no ground truth information about the segmentation, so
% quantitative quality control is difficult. However, we can overlay the
% population level ISC heatmap over the segmented images and 
% check whether a human annotator would agree about the object category under
% population level average gaze position.
%
% The frame-by-frame check would be extremely laborious and human
% annotations are only intended for quality checks. Therefore, we take take
% Experiment 1 dataset and take frames once in each second and overlay a
% heatmap over that.
%
% The heatmap is generated from 100 ms window around the frame to match any
% plausible inconsistencies in the timings between gaze position and
% frames.
%
% Severi Santavirta 11.10.2023

%% Inputs
dataset = 'localizer'; % Experiment 1 
input = sprintf('path/eyedata/subdata/%s/subjects',dataset); % Where are the eye-tracking data?
input_stimulus = 'path/video_segmentation/localizer/face/viz'; % Where are the segmented stimulus videos?
output = 'path/video_segmentation/segmentation_qc'; % Where to store the ISC results

sigma = 1; % heatmap radius as degrees (Nummenmaa 2014 or Lahnakoski 2014)
viewDist = 90; %cm, viewing distance
viewWidth = 31; %cm, Width of the presentations area in cm, not the whole screen width
viewWidthPixel = 720; % Width of the presentation area as pixels, not the whole screen width
viewHeightPixel = 576; % Height of the presentation area as pixels, not the whole screen width
excluded = {'L096'}; % Excluded based on QC
video_area_x = 153; % The first x coordinate that is inside video area
video_area_y = 97; % The first y coordinate that is inside video area

%% Get the gaze coordinates

if(~exist(sprintf('%s/temp.mat',output),'file'))

    f = find_files(input,'*.mat');
    [~,subjects,~] = fileparts(f);
    subjects = setdiff(subjects,excluded);

    for I = 1:size(subjects,1)
        fprintf('Reading fixations: %i/%i\n',I,size(subjects,1));

        load(sprintf('%s/%s.mat',input,subjects{I}));
        x_sub = subdata.fix_x;
        y_sub = subdata.fix_y;


        x_sub = x_sub-video_area_x+1; % correct coordinates for video area
        y_sub = y_sub-video_area_y+1; % correct coordinates for video area

        x_sub(logical(subdata.saccades.ts)) = NaN; % during saccade
        y_sub(logical(subdata.saccades.ts)) = NaN; % during saccade
        x_sub(~logical(subdata.fix_in_video_area)) = NaN; % outside video area.
        y_sub(~logical(subdata.fix_in_video_area)) = NaN; % outside video area.

        if(I==1)
            x = zeros(size(x_sub,1),size(subjects,1));
            y = zeros(size(x_sub,1),size(subjects,1));
            trials = subdata.trial_indices;
        end
        x(:,I) = x_sub;
        y(:,I) = y_sub;
    end

    % Save temporarily
    save(sprintf('%s/temp.mat',output),'x','y','trials','subjects');
    clear;
end

%% Create Gaussian kernel (sigma 1 degrees, Nummenmaa 2014 or Lahnakoski 2014))

kern = eISC_gaussKernel(sigma,[],viewDist,viewWidth,viewWidthPixel,1);

%% Loop over videos, extrack frames and calculate ISC around that frame

f = find_files(input_stimulus,'*mp4');
load(sprintf('%s/temp.mat',output));
n=0;
for I = 1:size(f,1)
    
    % Read the video
    vidIn = VideoReader(f{I});
    
    % Get the trialwise gaze coordinates
    x_trial = x(trials==I,:);
    y_trial = y(trials==I,:);
    
    % Identify frame time points (every one second). 
    t = (1000:1000:(vidIn.Duration*1000-50))'; % 50ms is here for making sure that we have enough data after the frame to calculate the ISC
    t((t+50)>size(y_trial,1)) = []; % For some videos eye-tracking stimulus may be a bit shorter than the video duration
    fr = ((t/1000)*vidIn.FrameRate)';
    
    % Loop thorugh the time points: Extract the frame and calculate ISC
    
    for J = 1:size(t,1)
        fprintf('Calculating heatmaps, Vid: %i, Frame %i\n',I,J);
 
        % Get the frame
        frame = read(vidIn,fr(J));
        
        % Get the ISC data for the frame (+-50ms)
        idx = (t(J)-50):(t(J)+50);
        x_frame = x_trial(idx,:);
        y_frame = y_trial(idx,:);
        
        % Data are calculated at each millisecond
        duration = zeros(size(idx,2),1);
        duration(:) = 1; % The data is stored in millisedond interval instead of fixations, hence this
        
        % Calculate heatmaps for each subject and then average for
        % population level
        fixation_heatmap = zeros(viewHeightPixel,viewWidthPixel,size(subjects,1));
        for K = 1:size(subjects,1)
            points = zeros(size(idx,2),2);
            points(:,1) = y_frame(:,K);
            points(:,2) = x_frame(:,K);

            fixation_heatmap(:,:,K) = eISC_fixationHeatmap(points,kern,viewWidthPixel,viewHeightPixel,duration);
            
        end
        
        % Average over subjects
        mean_heatmap = mean(fixation_heatmap,3);
        
        % Show only values over the 95% prctile
        min_val = prctile(mean_heatmap(:),95);
        thr_heatmap = mean_heatmap;
        thr_heatmap(mean_heatmap<min_val) = nan;

        % The highest individual value for scaling the colormap (99,9% prctile)
        max_val = prctile(mean_heatmap(:),99.9999);

        % Normalize the images to values between thr and max_val
        norm_heatmap = (thr_heatmap - min_val) / (max_val - min_val);
        norm_heatmap_scaled = norm_heatmap;
        norm_heatmap_scaled(norm_heatmap_scaled>1) = 1; % Correct extreme values (1% highest prctile) to 1 for scaling purposes

        % Define alpha channel
        alpha_channel = norm_heatmap-0.2;  % The lower the intensity the higher the transparency
        alpha_channel(isnan(alpha_channel)) = 0;
        image = ind2rgb(uint8(norm_heatmap_scaled * 255), jet(256));
        
        % Overlay the segmented frame with the scaled heatmap and alpha channel 
        imshow(frame);
        hold on;
        h = imshow(image);
        hold off;
        set(h, 'AlphaData', alpha_channel);
        n=n+1;
        saveas(gcf,sprintf('%s/frames/frame_%03d_segmentation_qc.png',output,n));
        
    end
    
end

%% Calculate the segmentation accuracy based on the human visual quality control

% Load the visual qc results
data = readtable('path/video_segmentation/segmentation/segmentation_qc/segmentation_qc.xlsx');
data = data(:, {'predicted_class', 'true_class'});

prediction = data.predicted_class;
true_class = data.true_class;

% Fill the true classes
for i = 1:size(true_class)
    if(isempty(true_class{i})) % The prediction was correct
        true_class{i} = prediction{i};
    end
end

% Calculate sensitivity and specificity for each class
category = {'eyes';'mouth';'face';'person';'animal';'object';'background';'unknown'};

results = zeros(size(category,1),9);
mis = zeros(size(category,1),8);
for c = 1:size(category,1)
    
    % total number of predictions
    t = sum(strcmp(prediction,category{c}));
    results(c,1) = t;
    
    % False positives
    fp = sum(strcmp(prediction,category{c}) & ~strcmp(true_class,category{c}));
    fp_idx = (strcmp(prediction,category{c}) & ~strcmp(true_class,category{c}));
    results(c,2) = fp;
    
    % False negative
    fn = sum(~strcmp(prediction,category{c}) & strcmp(true_class,category{c}));
    results(c,3) = fn;
    
    % True positives
    tp = sum(strcmp(prediction,category{c}) & strcmp(true_class,category{c}));
    results(c,4) = tp;
    
    % True negatives
    tn = sum(~strcmp(prediction,category{c}) & ~strcmp(true_class,category{c}));
    results(c,5) = tn;
    
    % Positive predictive value
    ppv = tp/t;
    results(c,6) = ppv;
    
    % Negative predictive value
    npv = tn/size(prediction,1);
    results(c,7) = npv;
    
    % Specificity
    specificity = tn/(tn+fp);
    results(c,8) = specificity;
    
    % Sensitivity
    sensitivity = tp/(tp+fn);
    results(c,9) = sensitivity;
    
    % Misclassification rates
    for cc = 1:size(category,1)
        mis(c,cc) = sum(strcmp(prediction,category{c}) & strcmp(true_class,category{cc}));
    end
end

% Save results
results = horzcat(array2table(category),array2table(results));
results.Properties.VariableNames = {'class','total_predictions','fp','fn','tp','tn','ppv','npv','specificity','sensitivity'};
mis = array2table(mis);
mis.Properties.VariableNames = category;
results_final = horzcat(results,mis);

writetable(results_final,'path/video_segmentation/segmentation/segmentation_qc/segmentation_qc_results_final.csv');



%% Kasky (Experiment 2): Extract lowlevel data and combine it with the mid-level semantic feature data
%
% Process
%       1. Extact low level information for every frame (Luminance,Entropy,SpatialEnergyLF,spatialEnergyHF)
%       3. Extract optic flow and differential energy for every consecutive frame pair
%       4. Calculate difference in framewise information between
%          consecutive frames
%       5. Save average visual low-level data for each frame (for regression)
%       7. Save pixelwise data averaged within specified time windows and
%          combine the data with the midlevel object information (for gaze prediction)
%       8. Extract auditory low-level information for every frame and
%          save with average visual low-level data (for regression)
%       9. Create 1ms time-series of the predictos that can be then analyzed
%          in different time windows
%
% Severi Santavirta 27.11.2023


dset = 'kasky'; % Experiment 2
stimulusDir = sprintf('path/stimulus/%s/%s_eyetracking_mp4',dset,dset); % Where are the stimulus video files?
objectFrameDir = sprintf('path/video_segmentation/%s/face/data',dset); % Where is the semantic segmentation data?
class_catalog = load(sprintf('path/video_segmentation/%s/face/catalog_class.mat',dset)); % Semantic segmentation class catalog
lowlevelTarget = sprintf('path/lowlevel/%s',dset); % % Where to store the results?
tmp = 'path/tmp'; % Folder for temp files

% To save memory read only a subset of frames in each parallel worker.
% Define the number of workers and the number of frames for each worker to
% read. Define also the resolution for low-level feature extraction.
% Downsampling saves space and time with little effect in data accuracy
npool = 14;
packSize = 24;
processResolution = [320,320]; % Feature extraction resolution
targetResolution = [64,64]; % Final analysis resolution

% Time windows where to average the data for gaze prediction analysis
tw = [200,500,1000]; % Regression analysis uses the 1 ms time series, but for gaze prediction analysis the script creates time wndow models for three different time windows

% Coordinate information
stimulus_frame = [720,1280]; % Stimulus video size (stimulus frame size, not the area of the display, nor the true borderless video area)
stimulus_presentation_area_y = 97:676; % Stimulus presentation area y coordinates (within the frame)
display_frame = [768,1024];

% Information for heatmap kernel calculations
% Localizer: sigma=1, viewDist = 90, viewWidth = 31, viewWidthPixel = 720, viewHeightPixel = 576, VideoArea: x = [153,873], y = [97,672], Stimulus size = [576,720]
% Kasky: sigma=1, viewDist = 70 (assumption), viewWidth = 61, viewWidthPixel = 1024, viewHeightPixel = 580, VideoArea: x = [1,1024], y = [96,676], Stimulus size = [720,1280]
% Conjuring: sigma=1, viewDist = 70 (assumption), viewWidth = ,60 viewWidthPixel = 1000, viewHeightPixel = 564, VideoArea: x = [13,1012], y = [105,668], Stimulus size = [750,1000]

% Get the broad class catalog
class_catalog = class_catalog.catalog_class;
cats_object = {'eyes';'mouth';'face';'person';'animal';'object';'background';'outside_video_area';'unknown'};
cats_lowlevel = {'Luminance','Entropy','SpatialEnergyLF','SpatialEnergyHF','LuminanceDiff','EntropyDiff','SpatialEnergyLFDiff','SpatialEnergyHFDiff','OpticFlow','DifferentialEnergy'};
cats_auditory = {'RMS','ZeroCrossing','Centroid','Spread','AuditoryEntropy','Rolloff85','Roughness','RMSDiff','ZeroCrossingDiff','CentroidDiff','SpreadDiff','AuditoryEntropyDiff','Rolloff85Diff','RoughnessDiff'};
%% Framewise feature extraction

% Determine out of presentation area pixels
xOut = [];
yOut = setdiff(1:stimulus_frame(1),stimulus_presentation_area_y);

% Open parallel pool
p = gcp('nocreate'); % If no pool, do not create new one.
if(isempty(p))
    p = parpool(npool);
end

vids = find_files(stimulusDir,'*.mp4');
% Loop over trials
for tr = 1:size(vids,1)
    formatTrialLastFile = sprintf('%s/models_1000ms/model_%s_tw1000_trial_%%0%dd.mat',lowlevelTarget,dset,3);
    if(~exist(sprintf(formatTrialLastFile,tr),'file'))
        tic;
        fprintf('Filtering frames: tr %d\n',tr);
        % Frame pack indices
        vidIn = VideoReader(vids{tr});
        nFrames = vidIn.NumFrames;
        framePackStartIdx = (1:packSize:nFrames)';
        
        % Loop over frame packs
        avgData = cell(size(framePackStartIdx,1),1); % Collect frame average data
        parfor pack = 1:size(framePackStartIdx,1)
        
            % Define pack frames
            packStart = framePackStartIdx(pack);
            if(pack<size(framePackStartIdx,1))
                packEnd = framePackStartIdx(pack+1)-1;
            else
                packEnd = nFrames;
            end
            packFrames = read(vidIn,[packStart,packEnd]);

            % For kasky, the x axis does not fit to the diplay screen
            % fully => need to cut from the sides to match the presentation
            % area
            xCutLeft = ((stimulus_frame(2)-display_frame(2))/2)+1;
            xCutRight = stimulus_frame(2)-xCutLeft+1;
            packFrames = packFrames(:,xCutLeft:xCutRight,:,:);

            % Resize to processing resolution
            packFrames= imresize(packFrames,processResolution,'bicubic');
        
            % Process frames one-by-one.
            avgDataFramePack = zeros(size(packFrames,4),4);
            for frameIdx = 1:size(packFrames,4)
                frameNumber = packStart+frameIdx-1;
                [filteredImage,avgDataFramePack(frameIdx,:),~] = image_filter(squeeze(packFrames(:,:,:,frameIdx)),true,true);
                save_filtered(filteredImage,tmp,frameNumber,nFrames);
            end 
            avgData{pack} = avgDataFramePack;
        end
        
        %% Between frames feature extraction
            
        % Parallel differential energy and optic flow calculations
        avgDataBetweenFrames = cell(2,1);
        parfor I = 1:2
            if(I==1)
                avgDataBetweenFrames{I} = video_opticFlow_filter(vids{tr},tmp,processResolution,true,true);
            else
                avgDataBetweenFrames{I} = video_differentialEnergy_filter(vids{tr},tmp,processResolution,true,true);
            end
        end
        
        %% Unlist average data, calculate derivative for the every frame features and save table
            
        avgDataFrame = zeros(nFrames,10,'single');
        avgDataFrame(:,9) = avgDataBetweenFrames{2};
        avgDataFrame(:,10) = avgDataBetweenFrames{1};
        
        for pack = 1:size(avgData,1)
        
            % Define pack frames
            packStart = framePackStartIdx(pack);
            if(pack<size(framePackStartIdx,1))
                packEnd = framePackStartIdx(pack+1)-1;
            else
                packEnd = nFrames;
            end
        
            avgDataFrame(packStart:packEnd,1:4) = avgData{pack}; 
        end
    
        % Calculate 1st derivative fot the features 1-4
        avgDataFrame(2:end,5:8) = diff(avgDataFrame(:,1:4));
        
        % Save table
        avgDataFrame = array2table(avgDataFrame);
        avgDataFrame.Properties.VariableNames = cats_lowlevel;
        formatStr = sprintf('%s/avgData/avgLowlevel_%s_%%0%dd.mat',lowlevelTarget,dset,3);
        save(sprintf(formatStr,tr),'avgDataFrame');
    
        %%  Create time window models for gaze prediction
        % Load trial information
        trials = load(sprintf('%s/dataset_trial_indices.mat',lowlevelTarget));
        if(strcmp(dset,'localizer'))
            trial = trials.trials{1,1};
        elseif(strcmp(dset,'kasky'))
            trial = trials.trials{2,1};
        else
            trial = trials.trials{3,1};
        end
        
        fprintf('Creating gaze prediction model Tr: %d\n',tr);
        for ww = 1:size(tw,2)
    
                
            % Define time windows
            t = (0:tw(ww):sum(trial==tr))';
    
            % Combine last and second last tw of the trial if the last tw would be under half
            % of the desired tw
            if((sum(trial==tr)-t(end))<(tw(ww)/2))
                t = t(1:end-1);
            end
    
            % Define the frame end timepoints (frames that end after the eye tracking trial has ended are excluded)
            frames = (40:40:floor((vidIn.FrameRate*sum(trial==tr)/1000))*40)';
    
            % Frame indices
            framesIdx = 1:size(frames,1);
    
            % Loop over time windows
            feature_frames_trial = cell(size(t,1),1);
            
            % Paths for temporary frames
            digit = numel(num2str(nFrames));
            [~,vidName,~] = fileparts(vids{tr});
            formatStrObject = sprintf('%s/%s/frame_%%0%dd.mat',objectFrameDir,vidName,digit);
            formatStrWithinFrame = sprintf('%s/filtered_%%0%dd.mat',tmp,digit);
            formatStrOpticFlow = sprintf('%s/opticflow_%%0%dd.mat',tmp,digit);
            formatStrDnrg = sprintf('%s/dnrg_%%0%dd.mat',tmp,digit);
    
            parfor w = 1:size(t,1)
    
                % Time frame for this tw
                t0_window =t(w)+1;
                if(w==size(t,1))
                    t1_window = sum(trial==tr);
                else 
                    t1_window = t(w+1);
                end
    
                % Frame end times that are within this tw
                frames_tw = frames(frames > t0_window & frames <=t1_window);
                framesIdx_tw = framesIdx(frames > t0_window & frames <=t1_window);
    
                % Loop over the frames
                object_frames = zeros(vidIn.Height,vidIn.Width,size(frames_tw,1),'uint8');
                within_frames = zeros(processResolution(1),processResolution(2),4,size(frames_tw,1),'single');
                opticflow_frames = zeros(processResolution(1),processResolution(2),size(frames_tw,1),'single');
                dnrg_frames = zeros(processResolution(1),processResolution(2),size(frames_tw,1),'single');
    
                n = 0;
                for K = framesIdx_tw(1):framesIdx_tw(end)
                    n=n+1;
                    % Collect the object information from the frame
                    frame = url_parLoad(sprintf(formatStrObject,K));
                    object_frames(:,:,n) = frame.mask;
    
                    % Collect lowlevel information from the frame
                    frame = url_parLoad(sprintf(formatStrWithinFrame,K));
                    within_frames(:,:,:,n) = frame.filteredImage;
                    frame = url_parLoad(sprintf(formatStrOpticFlow,K));
                    opticflow_frames(:,:,n) = frame.opticFlow;
                    frame = url_parLoad(sprintf(formatStrDnrg,K));
                    dnrg_frames(:,:,n) = frame.dnrg;
    
                end
    
                % Add the unknown category
                object_frames(object_frames == 0) = 139;
    
                % Add the info whether the pixel is within the presentation area
                object_frames(yOut,:,:) = 138;
                object_frames(:,xOut,:) = 138;
    
                % Transform category indices to broad class indices
                object_frames = assignClassIndex(object_frames,class_catalog);
    
                % Take the mode for each pixel over the frames to get the class
                % that was present most of the time
                object_frames_tw = mode(object_frames,3);
    
                % Collect results & downsample
                feature_frames_tw = zeros([targetResolution,11]);
    
                % Mean within frame
                feature_frames_tw(:,:,1:4) = imresize(mean(within_frames,4), targetResolution, 'bicubic');
    
                % Mean difference of framewise features over time
                feature_frames_tw(:,:,5:8) = imresize(mean(diff(within_frames,1,4),4), targetResolution, 'bicubic');
    
                % Optic flow
                feature_frames_tw(:,:,9) = imresize(mean(opticflow_frames,3), targetResolution, 'bicubic');
    
                % Differential energy
                feature_frames_tw(:,:,10) = imresize(mean(dnrg_frames,3), targetResolution, 'bicubic');
                
                % Object
                feature_frames_tw(:,:,11) = imresize(object_frames_tw, targetResolution, 'nearest');
    
                %Collect the results
                feature_frames_trial{w} = single(feature_frames_tw);
            end
    
            % Save the data
            formatStrModel = sprintf('%s/models_%dms/model_%s_tw%d_trial_%%0%dd.mat',lowlevelTarget,tw(ww),dset,tw(ww),3);
            save(sprintf(formatStrModel,tr),'feature_frames_trial','cats_lowlevel','cats_object');
        end
    
        % Trial completed, delete the files in the tmp folder
        delete(sprintf('%s/*',tmp));
        toc;
    end
end

%% Auditory feature extraction and match the average data with the eye-tracking duration

vids = find_files(stimulusDir,'*.mp4');
% Loop over trials
for tr = 1:size(vids,1)
    fprintf('Estimating auditory features: Tr %d\n',tr);

    % Load average visual dat of the trial to get the frame count
    formatStr = sprintf('%s/avgData/avgLowlevel_%s_%%0%dd.mat',lowlevelTarget,dset,3);
    data_visual = url_parLoad(sprintf(formatStr,tr));
    data_visual = data_visual.avgDataFrame;
    nFrames_visual(tr,1) = size(data_visual,1);

    % Extract lowlevel features
    data_auditory = video_lowlevelAudio(vids{tr},0.04,0.04); % Calculate auditory features in 40ms windows and the next windows start at the point where the previous ends.
    nFrames_auditory(tr,1) = size(data_auditory,1);

    % Calculate the 1st dericative between samples
    data_auditory_diff = zeros(size(data_auditory));
    data_auditory_diff(2:end,:) = diff(data_auditory);
    data_auditory = horzcat(data_auditory,data_auditory_diff);

    % In Some video files the auditory stream appears to be a
    % 40ms shorter than the video stream nad MIraudio gives a warning:
    % "The end of file was reached before the requested samples were
    % read completely". This suggests that there is a minor inaccuracy
    % in the Miraudio timewindow timings or the last window is missing
    % if it is not exactly as long as the requested window. We
    % duplicate the last smaple to match the number of frames. For one
    % trial, there is one more audio sample than there are frames. The
    % alst sample is discarded
    missingSamples = nFrames_visual(tr,1)-nFrames_auditory(tr,1);
    if(missingSamples>=0)
        data_auditory = vertcat(data_auditory,repmat(data_auditory(end,:),missingSamples,1));
    else
        data_auditory(end+missingSamples+1,:) = [];
    end
    data_auditory = array2table(data_auditory);
    data_auditory.Properties.VariableNames = cats_auditory;

    % The eye-tracking trials are as long as the video stream or a bit
    % shorter, due to inaccuaracy in avi to mp4 conversion. The mp4
    % file continues a short amount of time after the avi has alreqady
    % ended even though the timings are synchronized (most likely last last keyframe issue)
    % This is corrected so that the resulting data is cut from the end to match the eye-tracking duration
    data_lowlevel = horzcat(data_visual,data_auditory);

    % Get the trial eyetracking timing
    durEyeTrial = sum(trial==tr);

    % Define frame times that are within the eyetracking experiment
    t = (40:40:durEyeTrial)';
    nFramesEye = size(t,1);

    % Select right samples
    data_lowlevel = data_lowlevel(1:nFramesEye,:);

    % Save the table
    formatStr = sprintf('%s/avgLowlevel/lowlevel_%s_%%0%dd.mat',lowlevelTarget,dset,3);
    url_parSave(sprintf(formatStr,tr),data_lowlevel);
end

% Check the miss match of samples between auditory and visual features
nFr = horzcat(nFrames_visual,nFrames_auditory);

%% Create 1ms time series of the low-level data that can be the averaged to different time windows in the analyses

% Load framewise data and trial information
lowlevelFiles = find_files(sprintf('%s/avgLowlevel',lowlevelTarget),'*.mat');
trials = load(sprintf('%s/dataset_trial_indices.mat',lowlevelTarget));
if(strcmp(dset,'localizer'))
    trial = trials.trials{1,1};
elseif(strcmp(dset,'kasky'))
    trial = trials.trials{2,1};
else
    trial = trials.trials{3,1};
end

% Loop through trials
for tr = 1:size(lowlevelFiles,1)
    
    lowlevel_trial = load(lowlevelFiles{tr});
    
    % Lowlevel data is collected in 40ms time windows. Replicate each
    % elemnt 40 times to get the millisecond time series.
    data_trial_ms = repelem(lowlevel_trial.data_lowlevel,40,1);
    
    % The last frame may not be full 40ms correct with the trial length
    tEnd = sum(trial==tr);
    if(tEnd<=size(data_trial_ms)) % The last time window is not complete
        data_trial_ms = data_trial_ms(1:tEnd,:);
    elseif(tEnd<=size(data_trial_ms,1)+40) % The last time windows is <40m longer longer (replicate last to match the timing)
        df = tEnd-size(data_trial_ms,1);
        data_trial_ms = vertcat(data_trial_ms,repmat(data_trial_ms(end,:),df,1));
    elseif(tEnd>size(data_trial_ms))
        error('investigate');
    end
    
    if(tr==1)
        data_ms = data_trial_ms;
    else
        data_ms = vertcat(data_ms,data_trial_ms);
    end
end

lowlevel = struct;
lowlevel.data = data_ms;
lowlevel.trial = trial;

% Save results
save(sprintf('%s/lowlevel_data_1ms.mat',lowlevelTarget),'lowlevel');

%% Functions

function save_filtered(filteredImage,path,frameNumber,totalFrames)

    digit = numel(num2str(totalFrames));
    formatStr = sprintf('%s/filtered_%%0%dd.mat',path,digit);
    fileName = sprintf(formatStr,frameNumber);
    save(fileName,'filteredImage');
end
function framesClass = assignClassIndex(framesCategory,class_catalog)
    
    % Tranform categories to broad classes 
    framesClass = framesCategory;
    for c = 1:9
        idxClass = find(class_catalog.class_idx==(c));
        for cc = 1:size(idxClass,1)
            framesClass(framesCategory==idxClass(cc)) = c;
        end
    end

end
function [filteredImage,avgFiltered,filters] = image_filter(img,compress,scale)
% Function filters the input frame in multiple ways, stores the data into
% the "path" folder and returns the frame average
%
% Filters:
%       1. Luminance
%       2. Entropy
%       3. Spatial energy high frequency (Fourier transform -> low-pass filter -> 1% cut-off from the highest frequencies)
%       4. Spatial energy low frequency (Fourier transform -> low-pass filter -> 10% cut-off from the highest frequencies)
%
% Severi Santavirta 23.11.2023
filters = {'Luminance','Entropy','SpatialEnergyHF','SpatialEnergyLF'};

% Image size
siz = size(img);

% Initialize the output variables
if(compress)
    filteredImage = zeros([siz(2),siz(1),4],'single');
else
    filteredImage = zeros([siz(2),siz(1),4]);
end

% Filter image
avgFiltered = zeros(1,4);
[filteredImage(:,:,1),filteredImage(:,:,2),avgFiltered(1),avgFiltered(2)] = filter_luminance_entropy(img,siz,5,5,compress,scale);
[filteredImage(:,:,3),avgFiltered(3)] = filter_spatialEnergy(img,siz,1,compress,scale);
[filteredImage(:,:,4),avgFiltered(4)] = filter_spatialEnergy(img,siz,10,compress,scale);

end
function [luminance,visualEntropy,avgLuminance,avgEntropy] = filter_luminance_entropy(img,siz,stepSize,radius,compress,scale)

    % Get value from HSV for luminance estimation
    hsv_img = rgb2hsv(img);
    v = squeeze(hsv_img(:,:,3));
    v = v(:);
    
    % Greyscale for entropy estimation
    imgGrey = rgb2gray(img);
    
    % Create a meshgrid to represent the image coordinates
    [Y,X] = meshgrid(1:siz(1), 1:siz(2));
    
    % Step through image and calculate entropy for only subset of pixels to save time.
    xSteps = 1:stepSize:siz(2);
    ySteps = 1:stepSize:siz(1);
    luminance = zeros([size(ySteps,2),size(xSteps,2)]);
    visualEntropy = zeros([size(ySteps,2),size(xSteps,2)]);
    for y = 1:size(xSteps,2)
        for x = 1:size(xSteps,2)
            mask = (Y-ySteps(y)).^2 + (X-xSteps(x)).^2 <= radius^2;
            pixelsLuminance = v(mask(:));      
            pixelsEntropy = imgGrey(mask);
            
            luminance(x,y) = mean(pixelsLuminance);
            visualEntropy(x,y) = entropy(pixelsEntropy);
        end
    end
    
    % Resize back to the original size to get a value for each pixel
    luminance = imresize(luminance,[siz(1),siz(2)],'bicubic');
    visualEntropy = imresize(visualEntropy,[siz(1),siz(2)],'bicubic');
    avgLuminance = mean(luminance(:));
    avgEntropy = mean(visualEntropy(:));

    % Normalize
    if(scale)
        luminance = luminance./mean(luminance(:));
        visualEntropy = visualEntropy./mean(visualEntropy(:));
    end

    % Compress to save space
    % Compress data 
    if(compress)
        luminance = single(luminance);
        visualEntropy = single(visualEntropy);
    end
end
function [spatialEnergy,avgSpatialEnergy] = filter_spatialEnergy(img,siz,fourierRadius,compress,scale)

    % Fourier filter
    % Define Low-pass filter
    [Y,X]=ndgrid(1:siz(1),1:siz(2));
    rad = fourierRadius/100*siz(1);
    tmp = max((Y.^2+X.^2<rad^2),max((flipud(Y).^2+X.^2<rad^2),max((Y.^2+fliplr(X).^2<rad^2),(flipud(Y).^2+fliplr(X).^2<rad^2))));
    fourierMask = ones(siz(1),siz(2));
    fourierMask(tmp) = 0;
    
    % Calculate snrg
    F = fft2(mean(img,3,'omitnan')); % Fourier transform
    spatialEnergy = abs(ifft2(fourierMask.*F)); % Filtered image
    avgSpatialEnergy = mean(spatialEnergy(:));

    % Normalize
    if(scale)
        spatialEnergy = spatialEnergy./mean(spatialEnergy(:));
    end

    % Compress to save space
    if(compress)
        spatialEnergy = single(spatialEnergy);
    end
end
function avgOpticFlow = video_opticFlow_filter(videoFile,path,downsample,compress,scale)
% Estimate optic flow based on LK algorithm and basic options. Optic flow is a single
% measure of absolute movement between adjacent frames. For long videos,
% the memory does not allow to store such a big matrix. We need to store
% the data somewhere.
%
% Severi Santavirta

vidIn = VideoReader(videoFile);

% Methods
o = opticalFlowLK;

t = 0;
frames = vidIn.NumFrames;
digit = numel(num2str(frames));
avgOpticFlow = zeros(frames,1);
while hasFrame(vidIn)
    t=t+1;
    
    % Downsample image the given resolution to speed up the computations
    frame = imresize(readFrame(vidIn), downsample, 'bicubic');
    
    % Calculate optic flow
    flow = estimateFlow(o,im2gray(frame)); % Function calculates optic flow by comparing current frame to the previous (previous frame is assumed black for the first frame)
    opticFlow = flow.Vx.^2 + flow.Vy.^2; % Sum of movements in the both axes
    avgOpticFlow(t,1) = mean(opticFlow(:));

    % Normalize
    if(scale)
        opticFlow = opticFlow./mean(opticFlow(:));
    end

    if(compress)
        opticFlow = single(opticFlow); 
    end
    
    %Save
    formatStr = sprintf('%s/opticflow_%%0%dd.mat',path,digit);
    fileName = sprintf(formatStr,t);
    save(fileName,'opticFlow');
end
end
function avgDnrg = video_differentialEnergy_filter(videoFile,path,downsample,compress,scale)
% Function calculates the difference between voxels between adjacent frames
% Estimate of "movement/change" of image. Save images to path, since they
% cannot be returned if the video is very long.
%
% Severi Santavirta & Juha Lahnakoski

vidIn = VideoReader(videoFile);

imLast = zeros(downsample(1),downsample(2),3);
t=0;
frames = vidIn.NumFrames;
digit = numel(num2str(frames));
avgDnrg = zeros(frames,1);
while hasFrame(vidIn)
    t=t+1;
    im = double(vidIn.readFrame);
    
    % Downsample image the given resolution to speed up the computations
    im = imresize(im, downsample, 'bicubic');
    
    % Calculate dnrg
    dnrg = sqrt(mean((im-imLast).^2,3,'omitnan'));
    avgDnrg(t,1) = mean(dnrg(:));

    % Normalize
    if(scale)
        dnrg = dnrg./mean(dnrg(:));
    end

    if(compress)
        dnrg = single(dnrg);
    end
    
    %Save
    formatStr = sprintf('%s/dnrg_%%0%dd.mat',path,digit);
    fileName= sprintf(formatStr,t);
    save(fileName,'dnrg');
    
    imLast = im;
end

end
function [data,cats,dur_tw] = video_lowlevelAudio(videofile,time_window,hop)
% This function utilizes MIRtoolbox to extract some predefined low-level
% features mainly important for fMRI/eye-tracking analysis
% INPUT
%       videofile   = myVideo.mp4
%       time_window = temporal length of each time window (in seconds)
%       hop         = how far from the start of previous time-window to start the next
%                     time-window (in seconds). If you like to have
%                     interleaved time-windows then hop < time_window
% OUTPUT
%       data        = extracted auditory features
%       cats        = column names in data matrix
%       dur_tw      = Duration of the audio stream
%
% Severi Santavirta, last modified 27th May 2022


audioIn = mirframe(videofile,time_window,'s',hop,'s'); % Calculate everything in time-windows, windows are not interleaved
n_tw = size(mirgetdata(audioIn),2);
dur_tw = n_tw*time_window;


cats = {'rms','zerocrossing','centroid','spread','entropy','rollof85','roughness'};
data = zeros(n_tw,7);

data(:,1) = mirgetdata(mirrms(audioIn)); % rms: RMS / "intensity"

data(:,2) = mirgetdata(mirzerocross(audioIn)); % Zero crossing: Zero crossings of the audio wawe / "noisiness"

mu = mirgetdata(mircentroid(audioIn)); % Centroid: Mean of the frequency spectrum / "average frequency"  
mu(isnan(mu)) = nanmean(mu); % if silence, has NaN value, substitute with the average frequency over the whole video
data(:,3) = mu;  

sd = mirgetdata(mirspread(audioIn)); % Spread: SD of the frequency spectrum / "spread of frequencies"
sd(isnan(sd)) = nanmean(sd); % if silence, has NaN value, substitute with the average value in the sequence;
data(:,4) = sd;  

ent = mirgetdata(mirentropy(audioIn)); % Entropy of the frequency spectrum / "dominant peaks (low entropy) vs. heterogenous spectra (high entropy)"
ent(isnan(ent)) = nanmean(ent); % if silence, has NaN value, substitute with the average value in the sequence;
data(:,5) = ent;  

high_nrg = mirgetdata(mirrolloff(audioIn)); % Rolloff: "85% of energy is under this frequency"
high_nrg(isnan(high_nrg)) = nanmean(high_nrg); % if silence, has NaN value, substitute with the average value in the sequence;
data(:,6) = high_nrg;  

data(:,7) = mirgetdata(mirroughness(audioIn)); % Roughness: Sensory dissonance / "Roughness of the sound"

end
function var = url_parLoad(path)
    var = load(path);
end
function url_parSave(savefile,varargin)
% parsave v1.0.0 (June 2016).
% parsave allows to save variables to a .mat-file while in a parfor loop.
% This is not possible using the regular 'save' command.
%
% SYNTAX:   parsave(FileName,Variable1,Variable2,...)
%
% NOTE: Unlike 'save', do NOT pass the variable names to this function
% (e.g. 'Variable') but instead the variable itself, so without using the
% quotes. An example of correct usage is:
% CORRECT: parsave('file.mat',x,y,z);
%
% This would be INCORRECT: parsave('file.mat','x','y','z'); %Incorrect!
%
%Copyright (c) 2016 Joost H. Weijs
%ENS Lyon, France
%<jhweijs@gmail.com>

for i=1:nargin-1
    %Get name of variable
    name{i}=inputname(i+1);
    
     %Create variable in function scope
    eval([name{i} '=varargin{' num2str(i) '};']); 
end
%Save all the variables, do this by constructing the appropriate command
%and then use eval to run it.
comstring=['save(''' savefile ''''];
for i=1:nargin-1
    comstring=[comstring ',''' name{i} ''''];
end
comstring=[comstring ');'];
eval(comstring);
end
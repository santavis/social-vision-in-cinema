%% Calculate ISC in selected time-intervals
clear;clc;

% Objective is to calculate the eISC in different time scales to assess how
% the time window selection effects the analysis results.

% Severi Santavirta 1.11.2023

%% INPUT, Experiment 1

dataset = 'localizer'; % localizer (Exp. 1), kasky (Exp. 2), conjuring, (Exp. 3)
tw = [100,200,500,1000,2000,4000];
npool = 4; % Number of workers

excluded = {'C08';'C27';'K05';'K15';'K19';'K20';'K24';'L096'}; % Excluded based on QC
video_area_x = 153; % The first x coordinate that is inside video area
video_area_y = 97; % The first y coordinate that is inside video area

% Information for heatmap kernel calculations
% Localizer: sigma=1, viewDist = 90, viewWidth = 31, viewWidthPixel = 720, viewHeightPixel = 576, VideoArea: x = [153,873], y = [97,672]
% Kasky: sigma=1, viewDist = 70 (assumption), viewWidth = 61, viewWidthPixel = 1024, viewHeightPixel = 580, VideoArea: x = [0,1024], y = [96,676]
% Conjuring: sigma=1, viewDist = 70 (assumption), viewWidth = ,60 viewWidthPixel = 1000, viewHeightPixel = 564, VideoArea: x = [13,1012], y = [105,668]

sigma = 1; % heatmap radius as degrees (Nummenmaa 2014 or Lahnakoski 2014)
viewDist = 90; %cm, viewing distance
viewWidth = 31; %cm, Width of the presentations area in cm, not the whole screen width
viewWidthPixel = 720; % Width of the presentation area as pixels, not the whole screen width
viewHeightPixel = 576; % Height of the presentation area as pixels, not the whole screen width

input = sprintf('path/eyedata/subdata/%s/subjects',dataset); % Where are the eye-tracking data?
output = sprintf('path/isc/isc_changing'); % Where to store the ISC results

%%  Read fixations from preprocessed eye tracker data, Experiment 1

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
    
    if(I==1)
        x = zeros(size(x_sub,1),size(subjects,1));
        y = zeros(size(x_sub,1),size(subjects,1));
        trial = zeros(size(subdata.trial_indices,1),1);
        trial(1:find(subdata.trial_indices==17,1,'last')) = 1;
        trial((find(subdata.trial_indices==17,1,'last')+1):find(subdata.trial_indices==34,1,'last')) = 2;
        trial((find(subdata.trial_indices==34,1,'last')+1):find(subdata.trial_indices==51,1,'last')) = 3;
        trial((find(subdata.trial_indices==51,1,'last')+1):end) = 4;
    end
    x(:,I) = x_sub;
    y(:,I) = y_sub;
end

%% Determine windows using rating timings. Calculate heatmap for every time point and every subject, Experiment 1

p = gcp('nocreate'); % If no pool, do not create new one.
if(isempty(p))
    p = parpool(npool);
end

for w = 1:size(tw,2)
    if(~exist(sprintf('%s/isc_%d_millisecond_tw_%s.csv',output,tw(w),dataset),'file'))
        r = cell(size(unique(trial),1),1);
        trial_idx = cell(size(unique(trial),1),1);
        parfor tr = 1:size(unique(trial),1)            
            
            x_trial = x(trial==tr,:);
            y_trial = y(trial==tr,:);
    
            % Define timepoints
            t = (0:tw(w):size(x_trial,1))';
    
            % Combine last and second last tw of the trial if the last tw would be under half
            % of the desired tw
            if((size(x_trial,1)-t(end))<(tw(w)/2))
                t = t(1:end-1);
            end

            % Collect trial information
            trial_idx{tr} = repmat(tr,size(t,1),1);
    
            for I = 1:size(t,1) % time windows
                fprintf('Calculating ISC: TW size: %d/%d, Trial: %d/%d, TW: %d/%d\n',w,size(tw,2),tr,size(unique(trial),1),I,size(t,1));
                
                % TW indices
                if(I==size(t,1))
                    idx = ((t(I)+1):size(x_trial,1))';
                else
                    idx = ((t(I)+1):t(I+1))';
                end
                duration = zeros(size(idx,1),1);
                duration(:) = 1; % The data is stored in millisecond interval instead of fixations, hence this
                
                fixation_heatmap = zeros(viewHeightPixel,viewWidthPixel,size(subjects,1));
                for J = 1:size(x_trial,2) % subjects
    
                    points = zeros(size(idx,1),2);
                    points(:,1) = y_trial(idx,J);
                    points(:,2) = x_trial(idx,J);
                    fixation_heatmap(:,:,J) = eISC_fixationHeatmap(points,kern,viewWidthPixel,viewHeightPixel,duration);
                end
    
                % Calculate isc from the heatmap
                if(I==1)
                    [r_trial,r_trial_subs] = eISC_spatialSimilarity2(fixation_heatmap);
                    r_trial = horzcat(r_trial,r_trial_subs);
                else
                    [r_tw,r_tw_subs] = eISC_spatialSimilarity2(fixation_heatmap);
                    r_trial = vertcat(r_trial,horzcat(r_tw,r_tw_subs));
                end
            end
    
            %Collect trialwise isc:s
            r{tr} = r_trial;
          
        end
    
        % Unlist
        rr = [];
        trial_col = [];
        for i = 1:size(r,1)
            rr = vertcat(rr,r{i});
            trial_col = vertcat(trial_col,trial_idx{i});
        end
    
        % Save isc for each w separately
        isc = horzcat(array2table(trial_col),array2table(rr));
        isc.Properties.VariableNames = [{'trial'},{'isc'},subjects'];
        writetable(isc,sprintf('%s/isc_%d_millisecond_tw_%s.csv',output,tw(w),dataset));
    end
end

clear; clc;

%% INPUT, Experiment 2

dataset = 'kasky'; % localizer (Exp. 1), kasky (Exp. 2), conjuring, (Exp. 3)
tw = [100,200,500,1000,2000,4000];
npool = 4; % Number of workers

excluded = {'C08';'C27';'K05';'K15';'K19';'K20';'K24';'L096'}; % Excluded based on QC
video_area_x = 0; % The first x coordinate that is inside video area
video_area_y = 97; % The first y coordinate that is inside video area

% Information for heatmap kernel calculations
% Localizer: sigma=1, viewDist = 90, viewWidth = 31, viewWidthPixel = 720, viewHeightPixel = 576, VideoArea: x = [153,873], y = [97,672]
% Kasky: sigma=1, viewDist = 70 (assumption), viewWidth = 61, viewWidthPixel = 1024, viewHeightPixel = 580, VideoArea: x = [0,1024], y = [96,676]
% Conjuring: sigma=1, viewDist = 70 (assumption), viewWidth = ,60 viewWidthPixel = 1000, viewHeightPixel = 564, VideoArea: x = [13,1012], y = [105,668]

sigma = 1; % heatmap radius as degrees (Nummenmaa 2014 or Lahnakoski 2014)
viewDist = 70; %cm, viewing distance
viewWidth = 61; %cm, Width of the presentations area in cm, not the whole screen width
viewWidthPixel = 1024; % Width of the presentation area as pixels, not the whole screen width
viewHeightPixel = 580; % Height of the presentation area as pixels, not the whole screen width

input = sprintf('path/eyedata/subdata/%s/subjects',dataset); % Where are the eye-tracking data?
output = sprintf('path/isc/isc_changing/'); % Where to store the ISC results

%%  Read fixations from preprocessed eye tracker data, Experiment 2

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
    
    if(I==1)
        x = zeros(size(x_sub,1),size(subjects,1));
        y = zeros(size(x_sub,1),size(subjects,1));
        trial = subdata.trial_indices;
    end
    x(:,I) = x_sub;
    y(:,I) = y_sub;
end

%% Create Gaussian kernel (sigma 1 degrees, Nummenmaa 2014 or Lahnakoski 2014))

kern = eISC_gaussKernel(sigma,[],viewDist,viewWidth,viewWidthPixel,1);

%% Determine windows using rating timings. Calculate heatmap for every time point and every subject, Experiment 2

p = gcp('nocreate'); % If no pool, do not create new one.
if(isempty(p))
    p = parpool(npool);
end

for w = 1:size(tw,2)
    a = [];
    if(~exist(sprintf('%s/isc_%d_millisecond_tw_%s.csv',output,tw(w),dataset),'file'))
        r = cell(size(unique(trial),1),1);
        trial_idx = cell(size(unique(trial),1),1);
        parfor tr = 1:size(unique(trial),1)
            
            x_trial = x(trial==tr,:);
            y_trial = y(trial==tr,:);

            % Define timepoints
            t = (0:tw(w):size(x_trial,1))';
    
            % Combine last and second last tw of the trial if the last tw would be under half
            % of the desired tw
            if((size(x_trial,1)-t(end))<(tw(w)/2))
                t = t(1:end-1);
            end

            % Collect trial information
            trial_idx{tr} = repmat(tr,size(t,1),1);
            
            for I = 1:size(t,1) % time windows
                fprintf('Calculating ISC: TW size: %d/%d, Trial: %d/%d, TW: %d/%d\n',w,size(tw,2),tr,size(unique(trial),1),I,size(t,1));
                
                % TW indices
                if(I==size(t,1))
                    idx = ((t(I)+1):size(x_trial,1))';
                else
                    idx = ((t(I)+1):t(I+1))';
                end
                duration = zeros(size(idx,1),1);
                duration(:) = 1; % The data is stored in millisecond interval instead of fixations, hence this
                
                fixation_heatmap = zeros(viewHeightPixel,viewWidthPixel,size(subjects,1));
                for J = 1:size(x_trial,2) % subjects
    
                    points = zeros(size(idx,1),2);
                    points(:,1) = y_trial(idx,J);
                    points(:,2) = x_trial(idx,J);
                    fixation_heatmap(:,:,J) = eISC_fixationHeatmap(points,kern,viewWidthPixel,viewHeightPixel,duration);
                end
    
                % Calculate isc from the heatmap
                if(I==1)
                    [r_trial,r_trial_subs] = eISC_spatialSimilarity2(fixation_heatmap);
                    r_trial = horzcat(r_trial,r_trial_subs);
                else
                    [r_tw,r_tw_subs] = eISC_spatialSimilarity2(fixation_heatmap);
                    r_trial = vertcat(r_trial,horzcat(r_tw,r_tw_subs));
                end
            end
    
            %Collect trialwise isc:s
            r{tr} = r_trial;
          
        end

        % Unlist
        rr = [];
        trial_col = [];
        for i = 1:size(r,1)
            rr = vertcat(rr,r{i});
            trial_col = vertcat(trial_col,trial_idx{i});
        end
    
        % Save isc for each w separately
        isc = horzcat(array2table(trial_col),array2table(rr));
        isc.Properties.VariableNames = [{'trial'},{'isc'},subjects'];
        writetable(isc,sprintf('%s/isc_%d_millisecond_tw_%s.csv',output,tw(w),dataset));

    end
end

clear; clc;

%% INPUT, Experiment 3

dataset = 'conjuring'; % localizer (Exp. 1), kasky (Exp. 2), conjuring, (Exp. 3)
tw = [100,200,500,1000,2000,4000];
npool = 4; % Number of workers

excluded = {'C08';'C27';'K05';'K15';'K19';'K20';'K24';'L096'}; % Excluded based on QC
video_area_x = 13; % The first x coordinate that is inside video area
video_area_y = 105; % The first y coordinate that is inside video area

% Information for heatmap kernel calculations
% Localizer: sigma=1, viewDist = 90, viewWidth = 31, viewWidthPixel = 720, viewHeightPixel = 576, VideoArea: x = [153,873], y = [97,672]
% Kasky: sigma=1, viewDist = 70 (assumption), viewWidth = 61, viewWidthPixel = 1024, viewHeightPixel = 580, VideoArea: x = [0,1024], y = [96,676]
% Conjuring: sigma=1, viewDist = 70 (assumption), viewWidth = ,60 viewWidthPixel = 1000, viewHeightPixel = 564, VideoArea: x = [13,1012], y = [105,668]

sigma = 1; % heatmap radius as degrees (Nummenmaa 2014 or Lahnakoski 2014)
viewDist = 70; %cm, viewing distance
viewWidth = 60; %cm, Width of the presentations area in cm, not the whole screen width
viewWidthPixel = 1000; % Width of the presentation area as pixels, not the whole screen width
viewHeightPixel = 564; % Height of the presentation area as pixels, not the whole screen width

input = sprintf('path/eyedata/subdata/%s/subjects',dataset); % Where are the eye-tracking data?
output = sprintf('path/isc/isc_changing/'); % Where to store the ISC results

%%  Read fixations from preprocessed eye tracker data, Experiment 3

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
    
    if(I==1)
        x = zeros(size(x_sub,1),size(subjects,1));
        y = zeros(size(x_sub,1),size(subjects,1));
        trial = subdata.trial_indices;
    end
    x(:,I) = x_sub;
    y(:,I) = y_sub;
end


%% Create Gaussian kernel (sigma 1 degrees, Nummenmaa 2014 or Lahnakoski 2014))

kern = eISC_gaussKernel(sigma,[],viewDist,viewWidth,viewWidthPixel,1);

%% Determine windows using rating timings. Calculate heatmap for every time point and every subject, Experiment 3

p = gcp('nocreate'); % If no pool, do not create new one.
if(isempty(p))
    p = parpool(npool);
end

for w = 1:size(tw,2)
    if(~exist(sprintf('%s/isc_%d_millisecond_tw_%s.csv',output,tw(w),dataset),'file'))
        r = cell(size(unique(trial),1),1);
        trial_idx = cell(size(unique(trial),1),1);
        parfor tr = 1:size(unique(trial),1)
            
            x_trial = x(trial==tr,:);
            y_trial = y(trial==tr,:);
    
            % Define timepoints
            t = (0:tw(w):size(x_trial,1))';
    
            % Combine last and second last tw of the trial if the last tw would be under half
            % of the desired tw
            if((size(x_trial,1)-t(end))<(tw(w)/2))
                t = t(1:end-1);
            end

            % Collect trial information
            trial_idx{tr} = repmat(tr,size(t,1),1);
    
            for I = 1:size(t,1) % time windows
                fprintf('Calculating ISC: TW size: %d/%d, Trial: %d/%d, TW: %d/%d\n',w,size(tw,2),tr,size(unique(trial),1),I,size(t,1));
                
                % TW indices
                if(I==size(t,1))
                    idx = ((t(I)+1):size(x_trial,1))';
                else
                    idx = ((t(I)+1):t(I+1))';
                end
                duration = zeros(size(idx,1),1);
                duration(:) = 1; % The data is stored in millisecond interval instead of fixations, hence this
                
                fixation_heatmap = zeros(viewHeightPixel,viewWidthPixel,size(subjects,1));
                for J = 1:size(x_trial,2) % subjects
    
                    points = zeros(size(idx,1),2);
                    points(:,1) = y_trial(idx,J);
                    points(:,2) = x_trial(idx,J);
                    fixation_heatmap(:,:,J) = eISC_fixationHeatmap(points,kern,viewWidthPixel,viewHeightPixel,duration);
                end
    
                % Calculate isc from the heatmap
                if(I==1)
                    [r_trial,r_trial_subs] = eISC_spatialSimilarity2(fixation_heatmap);
                    r_trial = horzcat(r_trial,r_trial_subs);
                else
                    [r_tw,r_tw_subs] = eISC_spatialSimilarity2(fixation_heatmap);
                    r_trial = vertcat(r_trial,horzcat(r_tw,r_tw_subs));
                end
            end
    
            %Collect trialwise isc:s
            r{tr} = r_trial;
          
        end
    
        % Unlist
        rr = [];
        trial_col = [];
        for i = 1:size(r,1)
            rr = vertcat(rr,r{i});
            trial_col = vertcat(trial_col,trial_idx{i});
        end
    
        % Save isc for each w separately
        isc = horzcat(array2table(trial_col),array2table(rr));
        isc.Properties.VariableNames = [{'trial'},{'isc'},subjects'];
        writetable(isc,sprintf('%s/isc_%d_millisecond_tw_%s.csv',output,tw(w),dataset));
    end
end

function [r,r_sub] = eISC_spatialSimilarity2(fixMaps)
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
% r_sub:    Average correlation between one subject with all others
%
% Version 0.01
% 10.4.2012 Juha Lahnakoski
% juha.lahnakoski@aalto.fi
%
% Modified by Severi Santavrita on 10.11.2023 

%Calculate the correlations and select the upperm triangle entries without
%the diagonal (i.e. triu(...,1))
cMat=corr(reshape(fixMaps,[],size(fixMaps,3)));
cMat2=cMat(find(triu(ones(size(cMat)),1)));

%Calculate the mean using the Fisher Z-transform first (atanh) and then
%transforming back.
r=tanh(nanmean(atanh(cMat2)));

%Severi's modification, return the subject average correlation instead of
%the upper triangle
cMat(1:size(cMat,2)+1:end) = NaN;
r_sub = tanh(mean(atanh(cMat),'omitnan'));

end
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

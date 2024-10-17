%% Optimize and validate automatic cut detection using the manual annotations of the localizer (Experiment 1) dataset as ground truth
%
% Severi Santavirta 10.10.2023

%% Get the manually detected cuts (three independent annotators did these)

f = find_files('path/detection_accuracy_optimization','*cuts_manual');

% Load cuts and format
s1 = table2array(readtable(f{1},'ReadVariableNames',false));
s2 = table2array(readtable(f{2},'ReadVariableNames',false));
s3 = table2array(readtable(f{3},'ReadVariableNames',false));
t1 = cellfun(@frame_to_sec,s1); 
t2 = cellfun(@millisecond_to_sec,s2); 
t3 = cellfun(@frame_to_sec,s3);

% S1 has indicated the start time of the stimulus as the first cut, delete that
t1 = t1(2:end);

% S3 has reporded two exactly the same cuts twice
% The first is most likely an error inputing the next cut time (identified
% by comparing the time series to other annotators) and the second is just
% a duplicate
t3(33) = nan;
t3(42) = [];

% S2 has missed a cut in index 52
t2 = [t2(1:51);nan;t2(52:end)];

% S3 has missed a cut in index 60
t3 = [t3(1:59);nan;t3(60:end)];

% S3 has missed a cut in index 84
t3 = [t3(1:83);nan;t3(84:end)];

% S3 has made a clear typo in index 128, this is corrected
t3(128) = t3(128)+10;

% S1 has over ~2 sec different time for cut in index 166 than the two
% others. This is clearly a typo
t1(166) = t1(166)+2;

% S3 has identified the end of the video as cut, delete that
t3(end) = [];

% Now the annotators agree with the cut timings
cuts = horzcat(t1,t2,t3);

%Calculate the range between the time points to evaluate how accurately
%people report the timing of the cut.
time_range = range(cuts,2); % Average time inconsistency is ~200ms.

% Take the average of the timings as the ground truth value for the
% automatic algorithm optimization
t_manual = mean(cuts,2,'omitnan');

total_manual = 191;

%% Minimize detection error by changing the ffmpeg detection threshold
% if the detection is +-0,2sec (smaller than the average range between manual annotators) from the manual mean time stamp then the detection is correct
% See Bruckert et al 2023 => shortest shots >= 0.2
if(~exist('path/detection_accuracy_optimization/cut_detection_optimization.mat','file'))
    
    cd('path/detection_accuracy_optimization/tmp');
    detectionThr = 0.1; % Manually screened approximate lowest point to start
    
    cmd= sprintf("path/ffmpeg -i path/detection_accuracy_optimization/clips_presentation_order_fixed_length_for_manual_identification_of_cuts.mp4 -filter_complex ""select='gt(scene,%f)',metadata=print:file=path/detection_accuracy_optimization/tmp/time.txt"" -vsync vfr img%%03d.png",detectionThr);
    status = system(cmd);
    
    if status == 0
         
        % Read the data
        tbl = readtable("path/detection_accuracy_optimization/tmp/time.txt","ReadVariableNames",false);
        time = str2double(cellfun(@(x) x(6:end),table2array(tbl(1:2:end,2),'UniformOutput',false),'UniformOutput',false));
        score = str2double(cellfun(@(x) x(7:end),table2array(tbl(2:2:end,2),'UniformOutput',false),'UniformOutput',false));
    
        % Find the threshold where sensitivity and ppv cross
        cross=false;
        thr = detectionThr;
        first = true;
        while(~cross)
            
            % Increase hight
            thr = thr+0.001;
            
            % Cuts with higher score
            t_automatic_clip = time(score>thr);
            
             % Delete duplicate cuts (a cut cannot be within 0.2 sec apart, we delete the next one)
            idx_duplicate = [];
            n=0;
            lastReliable = 0;
            for J = 1:size(t_automatic_clip,1)
                diff = t_automatic_clip(J)-lastReliable; 
                if(diff<0.2)
                    n=n+1;
                    idx_duplicate(n) = J;
                else
                    lastReliable = t_automatic_clip(J);
                end
            end
            t_automatic_clip(idx_duplicate) = [];

            % Calculate correct predictions
            countCorrect = 0;

            % Loop through each element in vector1 and check for closeness
            for J = 1:size(t_automatic_clip,1)
                % Calculate the absolute differences between value1 and all elements in vector2
                differences = abs(t_automatic_clip(J) - t_manual);

                % Check if any difference is less than 0.2
                if(sum(differences < 0.2)==1)
                    countCorrect = countCorrect + 1;
                end
            end

            % Calculate missing predictionss
            countMiss = 0;

            % Loop through each element in vector1 and check for closeness
            for J = 1:size(t_manual,1)
                % Calculate the absolute differences between value1 and all elements in vector2
                differences = abs(t_manual(J) - t_automatic_clip);

                % Check if any difference is less than 0.2
                if(sum(differences < 0.2)==0)
                    countMiss = countMiss + 1;
                end
            end
            
            % Calculate sensitivity and ppv
            sensitivity = countCorrect/total_manual;
            ppv = countCorrect/size(t_automatic_clip,1);
            
            fprintf('Thr: %d, Correct: %d: Missing: %d, Total predictions: %d, Sensitivity: %f, PPV: %f\n',thr,countCorrect,countMiss,size(t_automatic_clip,1),sensitivity,ppv);
            if(sensitivity<ppv)
                cross=true;
            end
            if(first)
                res = [thr,countCorrect,countMiss,size(t_automatic_clip,1),sensitivity,ppv];
                first=false;
            else
                res = vertcat(res,[thr,countCorrect,countMiss,size(t_automatic_clip,1),sensitivity,ppv]);
            end
        end
    else
        error('System cmd failed.');
    end

    % Save the results
    res = array2table(res);
    res.Properties.VariableNames = {'detection_threshold','correct','missed','total_predictions','sensitivity','ppv'};
    save('path/detection_accuracy_optimization/cut_detection_optimization.mat','res');
end

% The detection threshold of 0.150 has the best accuracy (96% ppv / 95% sensitivity) and it is used to determine cuts in all movie data

%% Identify cuts with optimized cut threshold

input = 'path/stimulus';
output = 'path/video_segmentation/cuts/';
dset = {'localizer';'kasky';'conjuring'};

cd('path/video_segmentation/cuts/tmp');
for d = 1:size(dset,1)
    
    % Read video clip lenghts from the eye tracking data
    eyefolder = find_files(sprintf('path/eyedata/subdata/%s/subjects/',dset{d}),'*.mat');
    eyedata = load(eyefolder{1});
    trial_eye = eyedata.subdata.trial_indices;
        
    % Loop over video clips
    f = find_files(sprintf('/%s/%s/%s_eyetracking_mp4/',input,dset{d},dset{d}),'*.mp4');
    t_automatic = [];
    trial = [];
    t0 = 0;
    for v = 1:size(f,1)
        [~,vidName,~] = fileparts(f{v});
        cmd= sprintf("path/ffmpeg -i %s -filter_complex ""select='gt(scene,%f)',metadata=print:file=%s/%s/timepoints/%s.txt"" -vsync vfr img%%03d.png",f{v},0.150,output,dset{d},vidName);
        status = system(cmd);
        
        if status ~= 0 
            error('%s: Command failed to execute.',vidName);
        else
            
            % Get the detected cut timestamps
            tbl = readtable(sprintf('%s/%s/timepoints/%s.txt',output,dset{d},vidName),'ReadVariableNames',false);
            
            % It is possible that there are no cuts in a clip
            if(~isempty(tbl))
                
                t_automatic_clip = str2double(cellfun(@(x) x(6:end),table2array(tbl(1:2:end,2),'UniformOutput',false),'UniformOutput',false));
                

                % Delete cuts that occur after the eye-tracking stimulus has been ended (there can be one or few of these)
                t_eye = sum(trial_eye==v);
                t_automatic_clip((t_automatic_clip*1000)>t_eye) = [];
                
                % For some clips there may be cut detected right in the
                % end (<100ms to the end). Delete these too, since we are
                % adding the between clip cuts manually
                t_automatic_clip((t_automatic_clip*1000)>(t_eye-100)) = [];

                % Delete duplicate cuts (a cut cannot be within 0.2 sec apart, we delete the next one)
                if(~isempty(t_automatic_clip))
                    idx_duplicate = [];
                    n=0;
                    lastReliable = 0;
                    for J = 1:size(t_automatic_clip,1)
                        diff = t_automatic_clip(J)-lastReliable; 
                        if(diff<0.2)
                            n=n+1;
                            idx_duplicate(n) = J;
                        else
                            lastReliable = t_automatic_clip(J);
                        end
                    end
                    t_automatic_clip(idx_duplicate) = [];
                end
                
                % Convert to milliseconds
                t_automatic_clip = t_automatic_clip*1000;
            else
                t_automatic_clip = [];
            end
            
            % Every clip end with a "cut"
            t_automatic_clip = vertcat(1,t_automatic_clip);
            
            % Convert to cumulative time
            t_automatic_clip = t_automatic_clip+t0;
            t0 = t0+sum(trial_eye==v);
            
            % Store trial indices and collect cut information
            trial = vertcat(trial,repmat(v,size(t_automatic_clip,1),1));
            t_automatic = vertcat(t_automatic,t_automatic_clip);
        end
    end
    
    % Save results
    cuts_automatic = horzcat(t_automatic,trial);
    cuts_automatic = array2table(cuts_automatic);
    cuts_automatic.Properties.VariableNames = {'time','trial'};
    writetable(cuts_automatic,sprintf('path/video_segmentation/cuts/%s/gigatrack_%s_scene_cut_times.csv',dset{d},dset{d}));
end

%% Create 1ms time series of cut time points for analyses

dset = {'localizer';'kasky';'conjuring'};

for d = 1:size(dset,1)
    
    % Read cuts
    cuts = readtable(sprintf('path/video_segmentation/cuts/%s/gigatrack_%s_scene_cut_times.csv',dset{d},dset{d}));
    
    % Read eyetracking data for timings
    f = find_files(sprintf('path/eyedata/subdata/%s/subjects',dset{d}),'*.mat');
    eyedata = load(f{1});
    trial_idx = eyedata.subdata.trial_indices;
    trial = unique(trial_idx);
    
    % Loop over trials
    regressor = [];
    trials = [];
    t0 = 0;
    for tr = 1:size(trial,1)
        regressor_trial = zeros(sum(trial_idx==tr),1);
        
        cuts_trial = cuts.time(cuts.trial==tr)-t0;
        
        % For analyses purposes shift the cut between trials from the start to the end (1ms shift) execpt for the first trial
        cuts_trial(1) = [];
        if(tr==1)
            regressor_trial(1) = 1;
        else
            regressor_trial(end) = 1;
        end
        
        % Add scene cuts within trial
        regressor_trial(cuts_trial) = 1;
        
        % Collect time series
        regressor = vertcat(regressor,regressor_trial);
        trials = vertcat(trials,repmat(tr,size(regressor_trial,1),1));

        t0 = t0+size(regressor_trial,1);
    end
    
    % Save table
    regressor = array2table(regressor);
    regressor.Properties.VariableNames = {'cut'};
    regressor.trial = trials;
    save(sprintf('path/video_segmentation/cuts/%s/gigatrack_%s_scene_cut_regressor.mat',dset{d},dset{d}),'regressor');
end

%% Functions
function sec = frame_to_sec(str)
    sp = strsplit(str,':');
    try
        sec = str2double(sp{1})*60*60 + str2double(sp{2})*60 + str2double(sp{3}) + str2double(sp{4}(1:2))/30;
    catch
        a= 1;
    end
    
end

function sec = millisecond_to_sec(str)
    sp = strsplit(str,':');
    sec = str2double(sp{1})*60*60 + str2double(sp{2})*60 + str2double(sp{3}) + str2double(sp{4})/1000; 
end


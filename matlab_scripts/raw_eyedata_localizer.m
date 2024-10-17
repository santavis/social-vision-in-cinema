%% ------------------------------------------------------------------------------------------------

% GigaTrack: Localizer (Experiment 1): Extract fixation, saccades and blinks as well as pupil size, and fixation coordinates from eye-tracker fixation
% reports. Its imperative that the trial start times have been corrected to
% the start of the stimulus when the report has been generated in EyeLink
% software
%
% 1. Every fixation has been interpolated from the start of the fixation to
%    the start of the next fixation (over the saccade between fixations)
%
% 2. Fixations under 80 ms has been excluded. Last reliable fixation is
%    extrapolated to continue until the start of the next reliable fixation.
%    (over first saccade and over <80ms fixation and over next saccade)
%
% 3. If the first fixation of the trial is under 80 ms, then the start time
%    of the first reliable fixation has been extrapolated to time point zero. 
%
% 4. Last fixation of a trial is extrapolated to the end of the trial,
%    fixations do not cross trials. 

% 5. If a fixation < 80 ms occurs, the previous saccade is extrapolated
%    to continue until the start of next fixation. 
%
% 6. Changes in pupil size and gaze coordinates are calculated
%
% 7. Data is stored separately for every subject as matlab struct files.
%
% All extracted variables are interpolated to equal length (1ms) time series and
% every time series contains values from all presented trials in the
% presentation order. 

% Data for one trial only is easy to extract using an
% index vector specifying trial index for every time point.  
%
% Severi Santavirta, last modification 14th of August 2023
% ------------------------------------------------------------------------------------------------

%% INPUT

dataset = 'localizer'; % Experiment 1
fixation_report_path = 'path/eyedata/raw/localizer/fixrep_localizer.txt'; % Where is the fixation report?
saccade_report_path = 'path/eyedata/raw/localizer/sacrep_localizer.txt'; % Where is the saccade report?
stimulus_path = 'path/stimulus/localizer/localizer_eyetracking'; % Where are the stimulus video files?
func_path = 'path/funcs'; % Where are the needed functions?
output = 'path/eyedata/subdata/localizer'; % Where to save all the data?
addpath('path/funcs');

%% Preprocess reports (rename columns, add columns and sort rows to help data extraction. Also double check for missing data)

% Process fixation report (order table and check for missing data)
[path,~,~] = fileparts(fixation_report_path);
if(~exist(sprintf('%s/fixrep_%s_preprocessed.txt',path,dataset),'file'))
    
    % Read the fixation report and select columns of interest columns 
    fixrep = readtable(fixation_report_path);
    
    % Get an process subject names   
    fixrep.SUBJECT = processSubjectid(fixrep);

    % Sort the table to subject order
    fixrep = sortrows(fixrep,252,'ascend');

    % Save preprocessed fixation report
    writetable(fixrep,sprintf('%s/fixrep_%s_preprocessed.txt',path,dataset));
end

% Process saccade report
if(~exist(sprintf('%s/sacrep_%s_preprocessed.txt',path,dataset),'file'))
    
    % Read the fixation report and select columns of interest columns 
    sacrep = readtable(saccade_report_path);
    
    % Get an process subject names   
    sacrep.SUBJECT = processSubjectid(sacrep);

    % Sort the table to subject order
    sacrep = sortrows(sacrep,224,'ascend');

    % Save preprocessed fixation report
    writetable(sacrep,sprintf('%s/sacrep_%s_preprocessed.txt',path,dataset));

end

%% Create subjectwise interpolated ts over clips

% First, check whether the timings match between the stimulus video files and eye-tracker reports
% Read the stimulus files collect the clip durations
clips = find_files(stimulus_path,'*.mp4');
vid_dur_clip = zeros(size(clips,1),1);
for I = 1:size(clips,1)
    vidInfo = VideoReader(clips{I});
    vid_dur_clip(I,1) = vidInfo.Duration;
end

% Read the duration from fixation report
[path,~,~] = fileparts(fixation_report_path);
fixrep = readtable(sprintf('%s/fixrep_%s_preprocessed.txt',path,dataset));
tmp = fixrep.dur;
vid_dur_eyetracking = zeros(size(clips,1),1);
for I=1:size(clips,1)
    vid_dur_eyetracking(I) = unique(tmp(strcmp(fixrep.SUBJECT,'L001') & fixrep.TRIAL_INDEX==I)); % Hard-coded
end
vid_dur = horzcat(floor(vid_dur_clip*1000),vid_dur_eyetracking);
vid_dur(:,3) = vid_dur(:,1)-vid_dur(:,2);

% CHECKED. For some reason, the eye-tracking length for clip number 3 is ~ 1
% sec longer than the stimulus mp4 file. This is because the video freezes for one second after the video has ended. This is corrected so that 
% the eye-tracking data is cut to the actual point where the video ended. 4 other clips have (< 100ms) longer eye-tracking than mp4 clip and these are corrected similarly.
% All other clips are (<100ms) shorter in eye-tracking than the mp4 clips
% which is due to the avi to mp4 conversion inaccuracy. This is corrected
% by cutting the mp4 clips to eye-tracking length.
trial_durations = vid_dur_eyetracking;
trial_durations(vid_dur(:,3)<0) = vid_dur(vid_dur(:,3)<0,1);

% Load preprocessed reports
[path,~,~] = fileparts(fixation_report_path);
data_fix = readtable(sprintf('%s/fixrep_%s_preprocessed.txt',path,dataset));
data_sac = readtable(sprintf('%s/sacrep_%s_preprocessed.txt',path,dataset));

% Separate data matrix for every subject
subj_list_fix = data_fix.SUBJECT;
subj_list_sac = data_sac.SUBJECT;
subjects = unique(subj_list_fix,'stable');

% Create data folders folder

if(~exist(sprintf('%s/subjects',output),'dir'))
    mkdir(sprintf('%s/subjects',output));
end

if(~exist(sprintf('%s/qc',output),'dir'))
    mkdir(sprintf('%s/qc',output));
end

if(~exist(sprintf('%s/qc/report_warnings',output),'dir'))
    mkdir(sprintf('%s/qc/report_warnings',output));
end

for I = 1:size(subjects,1)
    if(~exist(sprintf('%s/subjects/%s.mat',output,subjects{I}),'file'))
       fprintf('%s/%s\n',num2str(I),num2str(size(subjects,1)));
       
       subdata = struct;
       warnings = {};
       
       subj_idx_fix = find(ismember(subj_list_fix,subjects{I}));
       subj_idx_sac = find(ismember(subj_list_sac,subjects{I}));
       submat_fix = data_fix(subj_idx_fix,:);
       submat_sac = data_sac(subj_idx_sac,:);
       
       % Check that the data matrices contain data for every trial
       check_fix = size(unique(submat_fix.TRIAL_INDEX),1);
       check_sac = size(unique(submat_sac.TRIAL_INDEX),1);
       if(check_fix<68 || check_sac <68) % Hard-coded 68 trials
          warning('Data for subject %s could not be read! Data  is not saved.',subjects{I});
          warnings{1,1} = 'Trial amount presented for subject is different than it should be! Data was not saved';
       else
           % Get basic data and warnings for quality control for fixations, saccades ans blinks.
           [subdata.fixations,subdata.trial_indices,warnings_fix] = getFixations(submat_fix,trial_durations); % Fixations
           warnings = vertcat(warnings,warnings_fix);
           
           [subdata.saccades,warnings_saccades] = getSaccades(submat_sac,trial_durations); % Saccades
           warnings = vertcat(warnings,warnings_saccades);
           
           [subdata.blinks,warnings_blinks] = getBlinks(submat_sac,trial_durations); % Blinks
           warnings = vertcat(warnings,warnings_blinks);
          
           % Get timeseries of interesting data (pupil size, x, y, upper left corner x=0, y=0)
           [subdata.pupil,subdata.fix_x,subdata.fix_y,warnings_fixdata] = getFixationData(submat_fix,trial_durations); % Pupil and coordinates 
           warnings = vertcat(warnings,warnings_fixdata);
           
           % Fixation inside video area: ( x = [153,873], y = [97,672])
           [subdata.fix_in_video_area,warning_videoarea] = getInsideVideoArea(subdata.fix_x,subdata.fix_y,[153,873],[97,672]);  
           warnings = vertcat(warnings,warning_videoarea);
           
           if(~isempty(warnings))
               order = cellfun(@fun1,warnings,'UniformOutput',false);
               [sorted,idx] = sortrows(order);
               warnings = warnings(idx);
           end

           [subdata.pupil_dif,~] = getDerivative(subdata.pupil); % (absolute) pupil size change
           [subdata.x_dif,~] = getDerivative(subdata.fix_x); % (absolute) x-coordinate change
           [subdata.y_dif,~] = getDerivative(subdata.fix_y); % (absolute) y-coordinate change

           h = zeros(size(subdata.y_dif,1),1);
           angle = zeros(size(subdata.y_dif,1),1);
           
           for J=1:size(subdata.y_dif,1)
               [h(J,1),angle(J,1)] = getTriangle(subdata.x_dif(J),subdata.y_dif(J)); % Calculate total xy-change and angle of the change
           end
           
           subdata.next_fix_distance = h;
           subdata.next_fix_direction = angle;
           
           save(sprintf('%s/subjects/%s.mat',output,subjects{I}),'subdata');
       end
       
    if(isempty(warnings))
       warnings{1,1} = sprintf('No warnings.');
    end
    
    warnings = array2table(warnings);
    writetable(warnings,sprintf('%s/qc/report_warnings/%s-warnings.csv',output,subjects{I}));    
    end
end

%% Create subjectwise summary table (for quality control and behavioral analyses)

if(exist(sprintf('%s/summary.csv',output),'file'))
    summary = [];
    subs = {};
    k=1;
    for I = 1:size(subjects,1)
        fprintf('Summary: %s/%s\n',num2str(I),num2str(size(subjects,1)));
        try
            load(sprintf('%s/subjects/%s.mat',output,subjects{I}));
            summary(k,1) = subdata.fixations.count;
            summary(k,2) = mean(subdata.fixations.durations);
            summary(k,3) = subdata.fixations.total_time;
            summary(k,4) = subdata.saccades.count;
            summary(k,5) = mean(subdata.saccades.durations);
            summary(k,6) = subdata.saccades.total_time;
            summary(k,7) = subdata.blinks.count;
            summary(k,8) = mean(subdata.blinks.durations);
            summary(k,9) = subdata.blinks.total_time;
            summary(k,10) = sum(subdata.fix_in_video_area)/size(subdata.fix_in_video_area,1);
            subs{k,1} = subjects{I};
            k=k+1;
        catch
            continue;
        end
     
    end
    
    summary = array2table(summary);
    subjects = array2table(subs);
    summary = horzcat(subjects,summary);
    summary.Properties.VariableNames = {'subjects','fixations','average_fixation_duration','total_time_fixations','saccades','average_saccade_duration','total_time_saccades','blinks','average_blink_duration','total_time_blinks','total_time_in_video_area'};
    
    writetable(summary,sprintf('%s/summary.csv',output));
end
    
%% Functions

function sub_id = processSubjectid(report)
    sessions = report.RECORDING_SESSION_LABEL;
    sessions = upper(sessions);
    subs = cellfun(@strsplit,sessions,cellstr(repmat('A',size(sessions,1),1)), 'UniformOutput', false);
    subs = vertcat(subs{:});
    subs = subs(:,2);
    subs = str2double(subs);
    sub_id = cell(size(subs,1),1);
    sub_id(subs<10) = cellfun(@sprintf,cellstr(repmat('L00%i',size(subs(subs<10),1),1)),num2cell(subs(subs<10)),'UniformOutput',false);
    sub_id(subs>9 & subs<100) = cellfun(@sprintf,cellstr(repmat('L0%i',size(subs(subs>9 & subs<100),1),1)),num2cell(subs(subs>9 & subs<100)),'UniformOutput',false);
    sub_id(subs>99) = cellfun(@sprintf,cellstr(repmat('L%i',size(subs(subs>99),1),1)),num2cell(subs(subs>99)),'UniformOutput',false);
end
function number=fun1(str)
    tmp = strsplit(str,' ');
    number = str2double(tmp{3});
end
function [fixations,trial_indices,warnings] = getFixations(submat_fix,trial_durations)
% Function reads subjectwise fixation reports and returns the fixation data
% Under 80ms fixations are defined as false findings and current saccade
% continues over these. If the first fixation of the trial does not start at 0
% then the ts will start with a saccade. Real trial durations are given as input
% to make sure that it has been previously checked that the durations from
% fixations reports match the durations of the stimulus files. 
%
% INPUT:
%         submat_fix        = subject's fixation data matrix from eye-tracker
%         trial_durations   = a Nx1 numeric vector of trial (stimulus)
%                             durations in milliseconds (in trial number order)
%
% OUTPUT:
%         fixations         = struct with fixation data
%         trial_indices     = time series of trial indices for easy extraction of trial specific data 
%         warnings          = warnings considering the quality of the data
%
% Severi Santavirta, last modification 9th of August 2023

trial_number_list = submat_fix.TRIAL_INDEX;
trial_number = unique(trial_number_list);
trial_durations = floor(trial_durations); % May not be exact if transformed from seconds.

warnings = {};
fix_ts = [];
fix_timestamps = [];
fix_durs = [];
fix_total_time = zeros(size(trial_number,1),1);
fix_count = 0;
fix_count_short = 0;
fix_count_long = 0;
trial_indices = [];

t = 0;
w = 0;
for I = 1:size(trial_number)
    idx = find(trial_number_list==trial_number(I));
    
    % Read the data from a table
    trial_dur = trial_durations(I);
    fix_start_times = submat_fix.CURRENT_FIX_START(idx);
    fix_end_times = submat_fix.CURRENT_FIX_END(idx);
    fix_durations = submat_fix.CURRENT_FIX_DURATION(idx);
    
    % Exclude fixations under 80ms.
    idx_over_80ms = fix_durations>80; 
    fix_start_times = fix_start_times(idx_over_80ms);
    fix_end_times = fix_end_times(idx_over_80ms);
    
    % The fixation has started at the beginning of the trial
    if(fix_start_times(1)==0) 
        fix_start_times(1) = 1;
    end
    
    % Delete fixations that start after the end of the trial (there should not be any, if there has not been problems with the experiment desing. Investigate if these warnings are returned.)
    idx_after_end = fix_start_times>trial_dur;
    if(sum(idx_after_end)>0)
        w=w+1;
        warnings{w,1} = sprintf('Trial number %i - %i fixations start after the trial has ended. Investigate the reason.',I,sum(idx_after_end));
        fix_start_times(idx_after_end) = [];
        fix_end_times(idx_after_end) = [];
    end
    
    % Correct the last fixation of the trial to the end of the trial if the
    % end time is after the stimulus has ended
    if(fix_end_times(end)>trial_dur)
        fix_end_times(end) = trial_dur;
    end
    
    % Create 1ms time series for the trial
    fix_ts_trial = zeros(trial_dur,1);
    trial_indices = vertcat(trial_indices,repmat(I,trial_dur,1));
    for J = 1:size(fix_start_times,1)
        fix_idx = fix_start_times(J):1:fix_end_times(J);
        fix_ts_trial(fix_idx) = 1;
    end

    % Catenate trial timeseries
    fix_ts = vertcat(fix_ts,fix_ts_trial);
    
    % Catenate timestamps
    fix_timestamps_trial = horzcat(fix_start_times,fix_end_times);
    fix_timestamps = vertcat(fix_timestamps,(fix_timestamps_trial+t));
    
    % Fixation durations
    fix_durs_trial = fix_timestamps_trial(:,2)-fix_timestamps_trial(:,1)+1;
    fix_durs = vertcat(fix_durs,fix_durs_trial);
    
    % Count over 80ms long fixations
    fix_count_trial = size(fix_timestamps_trial,1);
    fix_count = fix_count+fix_count_trial;
    
    % Count short fixations
    fix_count_short_trial = sum(fix_durations<80);
    fix_count_short = fix_count_short+fix_count_short_trial;
    
    % Count long fixations
    fix_count_long_trial = sum(fix_durations>5000);
    fix_count_long = fix_count_long+fix_count_long_trial;
    
    % Total fixation time
    fix_total_time_trial = sum(fix_ts_trial)/size(fix_ts_trial,1);
    fix_total_time(I,1) = fix_total_time_trial;
    
    % Warnings 
    if(fix_count_short_trial/fix_count_trial>0.05)
        w=w+1;
        warnings{w,1} = sprintf('Trial number %s - Trial contains %s/%s under 80 ms fixations.',num2str(trial_number(I)),num2str(fix_count_short_trial),num2str(size(fix_durs_trial,1)));
    end

    if(fix_count_long_trial>0)
        w=w+1;
        warnings{w,1} = sprintf('Trial number %s - Trial contains %s/%s over 5000 ms fixations',num2str(trial_number(I)),num2str(fix_count_long_trial),num2str(size(fix_durs_trial,1)));
    end
    
    t = t+trial_dur;
end

% Output results
fixations = struct;
fixations.ts = fix_ts;
fixations.timestamps = fix_timestamps;
fixations.durations = fix_durs;
fixations.count = fix_count;
fixations.count_short = fix_count_short;
fixations.count_long = fix_count_long;
fixations.total_time = sum(fix_ts)/size(fix_ts,1);

end
function [saccades,warnings] = getSaccades(submat_sac,trial_durations)
% Function reads subjectwise saccade reports and returns binary time series of
% saccades over the whole experiment. Real trial durations are given as input
% to make sure that it has been previously checked that the durations from
% fixations reports match the durations of the stimulus files. 
%
% INPUT:
%         submat_sac        = subject's saccade data matrix from eye-tracker
%         trial_durations   = a Nx1 numeric vector of trial (stimulus)
%                             durations in milliseconds (in trial number order)
%
% OUTPUT:
%         saccades          = struct with saccade data
%         warnings          = warnings considering the quality of the data
%
% Severi Santavirta, last modification 7th of December 2023

warnings = {};
w=0;
trial_number_list = submat_sac.TRIAL_INDEX;
trial_number = unique(trial_number_list);

sac_ts = [];
sac_timestamps = [];
sac_durs = [];
sac_total_time = zeros(size(trial_number,1),1);
sac_count = 0;
sac_count_long = 0;

t = 0;
for I = 1:size(trial_number,1)
    idx = find(trial_number_list==trial_number(I));
   
    % Read data from a table
    trial_dur = trial_durations(I);
    sac_start_times = submat_sac.CURRENT_SAC_START_TIME(idx);
    sac_end_times = submat_sac.CURRENT_SAC_END_TIME(idx);
    
    % Delete saccades that start after the end of the trial (there should not be any, if there has not been problems with the experiment design. Investigate if these warnings are returned.)
    idx_after_end = sac_start_times>trial_dur;
    if(sum(idx_after_end)>0)
        w=w+1;
        warnings{w,1} = sprintf('Trial number %i - %i saccades start after the trial has ended. Investigate the reason.',I,sum(idx_after_end));
        sac_start_times(idx_after_end) = [];
        sac_end_times(idx_after_end) = [];
    end
    
    % Correct the last saccade of the trial to the end of the trial if the
    % end time is after the stimulus has ended
    if(sac_end_times(end)>trial_dur)
        sac_end_times(end) = trial_dur;
    end
    
    % Saccade can be recorded to start before the stimulus start time,
    % exclude these
    minus_start_time = sac_start_times<0;
    sac_start_times(minus_start_time) = [];
    sac_end_times(minus_start_time) = [];
    
    % Saccade starts imediately at the beginning
    sac_start_times(sac_start_times==0) = 1;
    
    sac_timestamps_trial = horzcat(sac_start_times,sac_end_times);
    
    % Create saccade time series for the trial
    sac_ts_trial = zeros(trial_dur,1);
    for J = 1:size(sac_timestamps_trial,1)
        sac_ts_trial(sac_timestamps_trial(J,1):sac_timestamps_trial(J,2),1) = 1;
    end

    % Saccade timestamps
    sac_timestamps = vertcat(sac_timestamps,sac_timestamps_trial+t);
    
    % Count saccades
    sac_count_trial = size(sac_timestamps_trial,1);
    sac_count = sac_count+sac_count_trial;
    
    % Saccade durations
    sac_dur_trial = sac_timestamps_trial(:,2)-sac_timestamps_trial(:,1)+1;
    sac_durs = vertcat(sac_durs,sac_dur_trial);
    
    % Catenate trial saccades
    sac_ts = vertcat(sac_ts,sac_ts_trial);
    
    % Number of long saccades
    sac_count_long_trial = sum(sac_dur_trial>500);
    sac_count_long = sac_count_long+sac_count_long_trial;
    
    % Total saccade time
    sac_total_time_trial = sum(sac_ts_trial)/size(sac_ts_trial,1);
    sac_total_time(I,1) = sac_total_time_trial;
    
    if(sac_count_long_trial>0)
       w=w+1;
       warnings{w,1} = sprintf('Trial number %s - Trial contains %s/%s over 500ms saccades',num2str(trial_number(I)),num2str(sac_count_long_trial),num2str(size(sac_dur_trial,1))); 
    end

    if(sac_total_time_trial>0.25)
        w=w+1;
        warnings{w,1} = sprintf('Trial number %s - Total duration of saccades is %s of the whole trial duration.',num2str(trial_number(I)),num2str(sac_total_time_trial));
    end
    t = t+trial_dur;
end

% Output results as struct
saccades = struct;
saccades.ts = sac_ts;
saccades.timestamps = sac_timestamps;
saccades.durations = sac_durs;
saccades.count = sac_count;
saccades.count_long = sac_count_long;
saccades.total_time = sum(sac_ts)/size(sac_ts,1);

end
function [blinks,warnings] = getBlinks(submat_sac,trial_durations)
% Function reads subjectwise saccade reports and returns binary time series of
% blinks over the whole experiment. Real trial durations are given as input
% to make sure that it has been previously checked that the durations from
% fixations reports match the durations of the stimulus files. 
%
% INPUT:
%         submat_sac        = subject's saccade data matrix from eye-tracker
%         trial_durations   = a Nx1 numeric vector of trial (stimulus)
%                             durations in milliseconds (in trial number order)
%
% OUTPUT:
%         blinks            = struct with blink data
%         warnings          = warnings considering the quality of the data
%
% Severi Santavirta, last modification 10th of August 2023

trial_number_list = submat_sac.TRIAL_INDEX;
trial_number = unique(trial_number_list);
trial_durations = floor(trial_durations); % May not be exact if transformed from seconds.

blink_ts = [];
blink_timestamps = [];
blink_durs = [];
blink_total_time = zeros(size(trial_number,1),1);
blink_count = 0;
blink_count_long = 0;

warnings = {};

t = 0;
w = 0;
for I = 1:size(trial_number)
    blink_timestamps_trial = [];
    idx = find(trial_number_list==trial_number(I));
    
    % Read data from a table
    trial_dur = trial_durations(I);
    blink_start_trial = submat_sac.CURRENT_SAC_BLINK_START(idx);
    blink_end_trial = submat_sac.CURRENT_SAC_BLINK_END(idx);
    
    blink_start_trial = str2double(blink_start_trial(~cellfun(@fun,blink_start_trial)));
    blink_end_trial = str2double(blink_end_trial(~cellfun(@fun,blink_end_trial)));

    if(~isempty(blink_start_trial)) % The trial contains at least one blink
        
        
        % First blink starts before stimulus
        if(blink_start_trial(1)<1) 
            blink_start_trial(1) = 1;
        end
        
        % Delete blinks that start after the end of the trial (there should not be any, if there has not been problems with the experiment desing. Investigate if these warnings are returned.)
        idx_after_end = blink_start_trial>trial_dur;
        if(sum(idx_after_end)>0)
            w=w+1;
            warnings{w,1} = sprintf('Trial number %i - %i blinks start after the trial has ended. Investigate the reason.',I,sum(idx_after_end));
            blink_start_trial(idx_after_end) = [];
            blink_end_trial(idx_after_end) = [];
        end
        
        % Last blink ends after trial
        if(blink_end_trial(end)>trial_dur) 
            blink_end_trial(end) = trial_dur;
        end
        
        blink_timestamps_trial(:,1) = blink_start_trial;
        blink_timestamps_trial(:,2) = blink_end_trial;
        
        
        blink_dur_trial = blink_timestamps_trial(:,2) - blink_timestamps_trial(:,1)+1;

        blink_ts_trial = zeros(trial_dur,1);
        for J = 1:size(blink_timestamps_trial,1)
            idx_blink = blink_timestamps_trial(J,1):1:blink_timestamps_trial(J,2);
            blink_ts_trial(idx_blink) = 1;
        end

        % Blink timeseries
        blink_ts = vertcat(blink_ts,blink_ts_trial);

        % Blink timestamps
        blink_timestamps = vertcat(blink_timestamps,blink_timestamps_trial+t);

        % Blink count
        blink_count = blink_count + size(blink_timestamps_trial,1);

        % Blink durations
        blink_durs = vertcat(blink_durs,blink_dur_trial);

        % Long blinks
        blink_count_long_trial = sum(blink_dur_trial>300);
        blink_count_long = blink_count_long + blink_count_long_trial;

        % Blink total time trial
        blink_total_time_trial = sum(blink_dur_trial)/trial_dur;
        blink_total_time(I,1) = blink_total_time_trial;

        % Warnings
        if(blink_count_long_trial>0) % Long blink
            w =w+1;
            warnings{w,1} = sprintf('Trial number %s - Trial contains %s/%s over 300 ms blinks.',num2str(trial_number(I)),num2str(blink_count_long_trial),num2str(size(blink_dur_trial,1)));
        end
    
        if(blink_total_time_trial>0.10) % Blink total time high
            w =w+1;
            warnings{w,1} = sprintf('Trial number %s - Total duration of blinks is %s of the whole trial duration.',num2str(trial_number(I)),num2str(blink_total_time_trial));
        end
    else
        blink_ts_trial = zeros(trial_dur,1);
        
        % Blink timeseries
        blink_ts = vertcat(blink_ts,blink_ts_trial);
        
        % Blink total time
        blink_total_time(I,1) = 0;
    end
    
    t = t + trial_dur;
end

% Output results as struct
blinks = struct;
blinks.ts = blink_ts;
blinks.timestamps = blink_timestamps;
blinks.durations = blink_durs;
blinks.count = blink_count;
blinks.count_long = blink_count_long;
blinks.total_time = sum(blink_ts)/size(blink_ts,1);
    
end
function l = fun(str) % getBlinks needs this
    l = strcmp(str,'.');
end
function [pupil,x,y,warnings] = getFixationData(submat_fix,trial_durations)
% Function reads subjectwise fixation reports and returns the fixation data
% (pupil size, x and y coordinate). Real trial durations are given as input
% to make sure that it has been previously checked that the durations from
% fixations reports match the durations of the stimulus files. 
%
% 1. Every fixation has been interpolated from the start of the fixation to
%    the start of the next fixation (over the saccade between fixations)
%
% 2. Fixations under 80 ms has been excluded. Last reliable fixation is
%    extrapolated to continue until the start of the next reliable fixation.
%    (over first saccade and over <80ms fixation and over next saccade)
%
% 3. If the first fixation of the first trial is under 80 ms or started after the beginning of the trial, then the start time
%    of the first reliable fixation has been extrapolated to time point zero. 
%
% 4. Last fixation of a trial is corrected to end at the same time with the
% stimulus
%
% Since the data are extrapolated over trial boundaries, this function may
% not be suitable if calibration etc. happen between the trials
%
% INPUT:
%         submat_fix        = subject's fixation data matrix from eye-tracker
%         trial_durations   = a Nx1 numeric vector of trial (stimulus)
%                             durations in milliseconds (in trial number order)
%
% OUTPUT:
%         pupil, x, y       = 1 ms timeseries over the experiment
%
% Severi Santavirta, last modification 10th of August 2023

trial_number_list = submat_fix.TRIAL_INDEX;
trial_number = unique(trial_number_list);
trial_durations = floor(trial_durations); % May not be exact if transformed from seconds.

x = [];
y = [];
pupil = [];
warnings = {};
wn = 0;
for I = 1:size(trial_number)
    
    idx = find(trial_number_list==trial_number(I));
    
    % Exclude unreliable under 80 ms fixations
    idx_reliable = idx(submat_fix.CURRENT_FIX_DURATION(idx)>80);
   
    % Read data for reliable fixations
    trial_dur = trial_durations(I);
    fix_start_times = submat_fix.CURRENT_FIX_START(idx_reliable);
    fix_end_times = submat_fix.CURRENT_FIX_END(idx_reliable);
    fix_pupil = submat_fix.CURRENT_FIX_PUPIL(idx_reliable);
    fix_x = submat_fix.CURRENT_FIX_X(idx_reliable);
    fix_y = submat_fix.CURRENT_FIX_Y(idx_reliable);
    
    % Correct the first fixation to start at the beginning of the trial
    if(fix_start_times(1)~=1) 
        fix_start_times(1) = 1;
    end
    
    % Delete fixations that start after the end of the trial (there should not be any, if there has not been problems with the experiment desing. Investigate if these warnings are returned.)
    idx_after_end = fix_start_times>trial_dur;
    if(sum(idx_after_end)>0)
        wn=wn+1;
        warnings{wn,1} = sprintf('Trial number %i - %i fixations start after the trial has ended. Investigate the reason.',I,sum(idx_after_end));
        fix_start_times(idx_after_end) = [];
        fix_end_times(idx_after_end) = [];
        fix_pupil(idx_after_end) = [];
        fix_x(idx_after_end) = [];
        fix_y(idx_after_end) = [];
    end
        
    % Correct the last fixation to the end of the trial
    if(fix_end_times(end)~= trial_dur)
        fix_end_times(end) = trial_dur;
    end
    
    % Create 1 ms time series of pupil size and the fixation coordinates
    fix_pupil_trial = zeros(trial_dur,1);
    fix_x_trial = zeros(trial_dur,1);
    fix_y_trial = zeros(trial_dur,1);    
    for J = 1:size(fix_start_times,1)
        if(J<size(fix_start_times,1)) % Not last fixation
            
            % Extrapolate data to continue from the start of the current
            % fixation to the start of the next fixation 
            fix_pupil_trial(fix_start_times(J):(fix_start_times(J+1)-1)) = fix_pupil(J);
            fix_x_trial(fix_start_times(J):(fix_start_times(J+1)-1)) = fix_x(J);
            fix_y_trial(fix_start_times(J):(fix_start_times(J+1)-1)) = fix_y(J);
            
        else % Last fixation of the trial
            
            % Extrapolate the last fixation data to continue until the end
            % of the trial
            fix_pupil_trial(fix_start_times(J):end) = fix_pupil(J);
            fix_x_trial(fix_start_times(J):end) = fix_x(J);
            fix_y_trial(fix_start_times(J):end) = fix_y(J);
        end
    end
    
    % Catenate the trial data to the end of the previously extracted data
    pupil = vertcat(pupil,fix_pupil_trial);
    x = vertcat(x,fix_x_trial);
    y = vertcat(y,fix_y_trial);
end

end
function [insideVideo,warning] = getInsideVideoArea(x,y,x_area,y_area)

% VideoArea localizer: x = [153,873], y = [97,672]
% VideoArea kasky: x = [0,1024], y = [96,676]
% VideoArea conjuring: x = [13,1012], y = [105,668]

% Function returns a logical vector where 1 = input x and y are inside
% videoArea

if(size(x,1) ~= size(y,1))
    error('Size of x- and y-vector differ.');
end

insideVideo = ((y>y_area(1) & y<y_area(2)) & (x>x_area(1) & x<x_area(2)));
warning = [];
inside_time = sum(insideVideo)/size(x,1);
if(inside_time<0.9)
    warning = sprintf('Fixation is inside VideoArea only %s of the time.',num2str(inside_time));
end

end
function [yD,abs_yD] = getDerivative(y)
% Function calculates derivative or changing speed of given y during saccades
%
% Inputs:
%         y        = pupil_size, x_coordinate, y_coordinate -vector etc,
%
% Outputs:
%         yD       = derivative or changing speed of given y
%       abs_yD     = absolute value of yD
%
% Severi Santavirta, last modification 5th of March 2021

yD = zeros(size(y,1),1);
abs_yD = zeros(size(y,1),1);

y_ref = y(1);
y_ref_idx = 1;
for I = 1:size(y)
    yi = y(I);
    if(yi ~= y_ref) % First idx of a new fixation
        yiD = diff(horzcat(y_ref,yi));
        yD(y_ref_idx:I-1) = yiD;
        y_ref = yi;
        y_ref_idx = I;
    end
end

abs_yD = abs(yD);

end
function [h,angle] = getTriangle(x_dif,y_dif)
%Function calculates the hypotenuse and angle (between -180 and 180 degrees)
%of the gaze.

h = sqrt(x_dif.^2+y_dif.^2); % Distance

abs_x = abs(x_dif); sgn_x = sign(x_dif);
abs_y = abs(y_dif); sgn_y = sign(y_dif);

angle = rad2deg(atan(abs_y/abs_x));

if(sgn_x < 0 && sgn_y > 0)
    angle = -180 + angle;
elseif(sgn_x > 0 && sgn_y > 0)
    angle = -angle;
elseif(sgn_x < 0 && sgn_y < 0)
    angle = 180 - angle;
end

if(isnan(angle)) % fixation on, no change
    angle = 0;
end
end
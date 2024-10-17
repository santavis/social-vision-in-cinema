%% Gaze Prediction analysis, train gaze prediction models
%
% Process:
%
%       Loop through datasets and create independent gaze model for each.
%           1. Load trialwise heatmaps and pixelwise design matrices
%           2. Optimize hyperparameters with Experiments 1 & 2
%           3. Train a optimized random forest model separately for each Experiment to predict gaze heatmaps
%
% Severi Santavirta 16.11.2023

clear;clc;

%% INPUT

% Datasets
dset = {'localizer','kasky','conjuring'}; % Experiments 1-3

% Drop last trials for kasky and conjuring since they contain the end
% texts, no "social" scenes
include_trials = [{1:68},{1:25},{1:29}];

% Choose the time window (in milliseconds 200,500 or 1000)
tw = 200;

% Path where the trialwise lowlevel data are saved
path_models = 'path/lowlevel';

% Path where the high-level social data are saved
path_highlevel = 'path/socialdata';

% Path where the cut information is stored
path_cuts = 'path/video_segmentation/cuts';

% Path where the trialwise heatmaps are saved
path_heatmaps = 'path/gaze_prediction/heatmaps';

% Path where to save trained models
path_output = 'path/gaze_prediction';

% What is the number of pixels in the model frames
pix = 4096;

% What is the frame resolution
resolution = [64,64];

% To save resources select every ith window
selectWindow = 1; % 1 = Include all time windows

% Plot framewise results?
plotResults = true;

%% Optimize models independently for each dataset
for d = 1:size(dset,2)
    fprintf('Hyperparameter optimization: Dset: %d, Reading dat\n',d);
    tic;
    % Load data
    %-------------------------------------------------------------------------------------------------------------------------------
    highlevel = load(sprintf('%s/%s/%s_eyetracking_social_regressors_1ms.mat',path_highlevel,dset{d},dset{d}));
    cuts = load(sprintf('%s/%s/gigatrack_%s_scene_cut_regressor.mat',path_cuts,dset{d},dset{d}));
    lowlevel_auditory = load(sprintf('%s/%s/lowlevel_data_1ms.mat',path_models,dset{d}));
    lowlevel_auditory = lowlevel_auditory.lowlevel.data(:,11:end);
    
    files_heatmap = find_files(sprintf('%s/%s/tw%d',path_heatmaps,dset{d},tw),'*.mat');
    files_models = find_files(sprintf('%s/%s/models_%dms',path_models,dset{d},tw),'*.mat');

    % Get feature names
    cats_highlevel = highlevel.regressors_1ms.Properties.VariableNames(1:8)';
    cats_auditory = lowlevel_auditory.Properties.VariableNames;

    % Loop over trial files
    x_lowlevel = [];
    x_lowlevel_auditory = [];
    x_midlevel = [];
    x_highlevel = [];
    x_cuts = [];
    y = [];
    for tr = 1:size(files_models,1)
        if(ismember(tr,include_trials{d}))
            % Trial heatmaps
            load(files_heatmap{tr});
    
            % Trial lowlevel visual and mid-level models
            load(files_models{tr});

            % Trial lowlevel auditory
            lowlevel_auditory_trial_1ms = table2array(lowlevel_auditory(cuts.regressor.trial==tr,:));

            % Trial highlevel
            highlevel_trial_1ms = table2array(highlevel.regressors_1ms(cuts.regressor.trial==tr,1:8));

            % Trial cuts
            cuts_trial = cuts.regressor.cut(cuts.regressor.trial==tr,1);
    
            % Define time windows
            t = (0:tw:size(cuts_trial,1))';
    
            % Combine last and second last tw of the trial if the last tw would be under half
            % of the desired tw
            if((size(cuts_trial,1)-t(end))<(tw/2))
                t = t(1:end-1);
            end
    
            % Average high-level and cut for the time windows
            x_trial_highlevel =zeros(size(t,1),size(highlevel_trial_1ms,2));
            x_trial_lowlevel_auditory =zeros(size(t,1),size(lowlevel_auditory_trial_1ms,2));    
            x_trial_cuts =zeros(size(t,1),1);    
            for w = 1:size(t,1)
        
                % Time frame for this tw
                t0_window =t(w)+1;
                if(w==size(t,1))
                    t1_window = size(highlevel_trial_1ms,1);
                else 
                    t1_window = t(w+1);
                end
        
                x_trial_highlevel(w,:) = mean(highlevel_trial_1ms(t0_window:t1_window,:));
                x_trial_lowlevel_auditory(w,:) = mean(lowlevel_auditory_trial_1ms(t0_window:t1_window,:));
                if(sum(cuts_trial(t0_window:t1_window))>0)
                    x_trial_cuts(w,1) = 1;
                end
            end

            % Repeat the low-level auditory values for each pixel in the timewindow frame
            x_trial_lowlevel_auditory = repelem(x_trial_lowlevel_auditory,pix,1);
    
            % Repeat the high-level values for each pixel in the timewindow frame
            x_trial_highlevel = repelem(x_trial_highlevel,pix,1);

            % Repeat the cuts for each pixel in the timewindow frame
            x_trial_cuts = repelem(x_trial_cuts,pix,1);
        
            % Vectorize heatmap
            y_sub = reshape(trial_heatmaps,[],size(trial_heatmaps,3));
    
            % Create mid- and low-level models
            for w = 1:size(feature_frames_trial,1)
                % Vectorize tw model
                x = reshape(feature_frames_trial{w},[],size(feature_frames_trial{w},3));
                
                % Define what class information we have in this frame and make
                % dummy predictors for the midlevel features
                x_midlevel_dummy = zeros(size(x,1),9);
                objectClassIdx = unique(x(:,end));
                for c = 1:size(objectClassIdx,1)
                    x_midlevel_dummy(:,objectClassIdx(c)) = (x(:,11)==objectClassIdx(c));
                end
                
                % The object dummy variables add to one. Lets remove the "unknown" and
                % "outside video area" since they offer little socially relevant information
                x_midlevel_dummy = x_midlevel_dummy(:,1:7);
                
                %Collect the design matrix
                idx0 = 1+(w-1)*size(y_sub,1);
                idx1 = idx0+size(y_sub,1)-1;
                if(w==1)
                    x_trial_lowlevel = zeros(size(y_sub,2)*size(y_sub,1),10,'single');
                    x_trial_midlevel = zeros(size(y_sub,2)*size(y_sub,1),7,'single');
                    y_trial = zeros(size(y_sub,2)*size(y_sub,1),1);
                end
                
                if(w==size(feature_frames_trial,1) && size(feature_frames_trial,1)==(size(y_sub,2)+1)) % The last time window which is shorter than the specified tw has been calculated for the features but not for the heatmaps
                    break;
                else
                    x_trial_lowlevel(idx0:idx1,:) = x(:,1:end-1);
                    x_trial_midlevel(idx0:idx1,:) = x_midlevel_dummy;
                    y_trial(idx0:idx1,1) = y_sub(:,w);
                end
            end
    
            % Select windows
            if(selectWindow>1)
                nFrames = size(x_trial_lowlevel,1)/pix;
                idx = repelem(1:nFrames,pix,1);
                idx1 = idx(:);
                selectIdx = mod(idx1, selectWindow) == 1;
                x_trial_lowlevel = x_trial_lowlevel(selectIdx,:);
                x_trial_lowlevel_auditory = x_trial_lowlevel_auditory(selectIdx,:);
                x_trial_midlevel = x_trial_midlevel(selectIdx,:);
                x_trial_highlevel = x_trial_highlevel(selectIdx,:);
                x_trial_cuts = x_trial_cuts(selectIdx,:);
                y_trial = y_trial(selectIdx,:);
            end
            
            % Collect trial data
            x_lowlevel = vertcat(x_lowlevel,x_trial_lowlevel);
            x_lowlevel_auditory = vertcat(x_lowlevel_auditory,single(x_trial_lowlevel_auditory));
            x_midlevel = vertcat(x_midlevel,single(x_trial_midlevel));
            x_highlevel = vertcat(x_highlevel,single(x_trial_highlevel));
            x_cuts = vertcat(x_cuts,single(x_trial_cuts));

            y = vertcat(y,single(y_trial));
        end
    end

    % Combine the design matrix
    x = horzcat(zscore(x_lowlevel,[],'omitnan'),zscore(x_lowlevel_auditory,[],'omitnan'),zscore(x_highlevel,[],'omitnan'),x_midlevel,x_cuts);
    predictors = [cats_lowlevel';cats_auditory';cats_highlevel;cats_object(1:7);{'cuts'}];
    
    % Only few animals in the data and poor positive predictive value for
    % "animals" the category was excluded from regression also. Exclude it
    % here too.
    x(:,37) = [];
    predictors(37) = [];

    % Combine predictors by taking the mean based on the clustering result
    % (in R) of the predictors
    [x,predictors] = createClusterPredictors(x,predictors);
    x = single(x);

    % Normalize yy between 0-1 in the whole data 
    tmp = y;
    prctile_train = prctile(tmp,99.9);
    tmp = tmp./prctile_train;
    tmp(tmp<0) = 0;
    tmp(tmp>1) = 1;
    y = tmp;
    
    clearvars -except r_train models importances hyperparameters x y predictors dset tw path_models path_highlevel path_isc path_cuts path_heatmaps path_output pix resolution selectWindow cats_object cats_lowlevel cats_lowlevel_auditory cats_highlevel d include_trials weight_isc plotResults

    % Opimize Parameters (80/20 split). Optimization was done to Experimetn
    % 1 & 2. The optimization results were highly similar and more complex
    % models did not perform much better. To save computation time, we did
    % not optimize for the Experiment 3 data.
    %-------------------------------------------------------------------------------------------------------------------------------
    maxNumSplit = [15,31,63,127]; % 4,5 or 6 full generations of branches
    numTrees = [50,100]; % Number or individual trained trees

    % Split to train and validate sets
    splitLastIdx = floor(size(y,1)/pix*0.80)*pix;
    y_train = y(1:splitLastIdx,:);
    y_validate = y(splitLastIdx+1:end,:);
    x_train = x(1:splitLastIdx,:);
    x_validate = x(splitLastIdx+1:end,:);

    clearvars -except maxNumSplit numTrees r_train models y_train x_train y_validate x_validate importances hyperparameters predictors dset tw path_models path_highlevel path_isc path_cuts path_heatmaps path_output pix resolution selectWindow cats_object cats_lowlevel cats_lowlevel_auditory cats_highlevel d include_trials weight_isc plotResults

    % Loop though the hyperparameter space and save the test set r
    results_hyperparameters_dset = zeros(size(maxNumSplit,2)*size(numTrees,2),4);
    n=0;
    for splitSize = 1:size(maxNumSplit,2)
        for treeSize = 1:size(numTrees,2)
            tic;
            n=n+1;
            fprintf('Dset: %d, Hyperparameter tuning %d/%d\n',d,n,size(results_hyperparameters_dset,1));

            t = templateTree('MaxNumSplits',maxNumSplit(splitSize));
            mdl = fitrensemble(x_train,y_train,'Method','Bag','NumLearningCycles',numTrees(treeSize),'Learners',t); % Random forest
            yhat = predict(mdl,x_validate);
            results_hyperparameters_dset(n,1) = corr(yhat,y_validate);
            results_hyperparameters_dset(n,2) = maxNumSplit(splitSize);
            results_hyperparameters_dset(n,3) = numTrees(treeSize);
            results_hyperparameters_dset(n,4) = toc/60;
            fprintf('Timer: %.01f minutes\n',results_hyperparameters_dset(n,4));
        end
    end

    % Find the best parameters and save the results
    results_hyperparameters_dset = array2table(results_hyperparameters_dset);
    results_hyperparameters_dset.Properties.VariableNames = {'R_validation','NumSplits','NumTrees','Time'};
    results_hyperparameters_dset = sortrows(results_hyperparameters_dset,'R_validation','descend');

    % Save the results
    model = struct;
    model.Trained = char(datetime);
    model.TimeWindow = tw;
    model.SelectedWindows = selectWindow;
    model.TrainSet = dset{d};
    model.TrialsIncluded = include_trials{d};
    model.HyperparameterOptimization = results_hyperparameters_dset;
    model.OptimSplits = results_hyperparameters_dset.NumTrees(1);
    model.OptimNumTrees = results_hyperparameters_dset.NumSplits(1);
    save(sprintf('%s/optimization/tw%d/trained_model_%s_tw%d.mat',path_output,tw,dset{d},tw),'model');
end

%% Train gaze prediction models independently for each dataset

% The hyperparameters (number of splits and tree size) were optimized with
% the 200 ms time window in Exepriment 1 & 2 data. Increasing tree size to 63 increased
% the prediction accuracy in validation dataset but increasing tree size
% from 50 only increased the computation time. Hence, we choose to use 63
% splits and 50 trees.

OptimSplits = 63;
OptimTrees = 50;

for d = 1:size(dset,2)
    fprintf('Training optimized model: Dset: %d, Reading data\n',d);
    tic;

    % Load data
    %-------------------------------------------------------------------------------------------------------------------------------
    highlevel = load(sprintf('%s/%s/%s_eyetracking_social_regressors_1ms.mat',path_highlevel,dset{d},dset{d}));
    cuts = load(sprintf('%s/%s/gigatrack_%s_scene_cut_regressor.mat',path_cuts,dset{d},dset{d}));
    lowlevel_auditory = load(sprintf('%s/%s/lowlevel_data_1ms.mat',path_models,dset{d}));
    lowlevel_auditory = lowlevel_auditory.lowlevel.data(:,11:end);
    
    files_heatmap = find_files(sprintf('%s/%s/tw%d',path_heatmaps,dset{d},tw),'*.mat');
    files_models = find_files(sprintf('%s/%s/models_%dms',path_models,dset{d},tw),'*.mat');

    % Get feature names
    cats_highlevel = highlevel.regressors_1ms.Properties.VariableNames(1:8)';
    cats_auditory = lowlevel_auditory.Properties.VariableNames;

    % Loop over trial files
    x_lowlevel = [];
    x_lowlevel_auditory = [];
    x_midlevel = [];
    x_highlevel = [];
    x_cuts = [];
    y = [];
    for tr = 1:size(files_models,1)
        if(ismember(tr,include_trials{d}))
            % Trial heatmaps
            load(files_heatmap{tr});
    
            % Trial lowlevel visual and mid-level models
            load(files_models{tr});

            % Trial lowlevel auditory
            lowlevel_auditory_trial_1ms = table2array(lowlevel_auditory(cuts.regressor.trial==tr,:));

            % Trial highlevel
            highlevel_trial_1ms = table2array(highlevel.regressors_1ms(cuts.regressor.trial==tr,1:8));

            % Trial cuts
            cuts_trial = cuts.regressor.cut(cuts.regressor.trial==tr,1);
    
            % Define time windows
            t = (0:tw:size(cuts_trial,1))';
    
            % Combine last and second last tw of the trial if the last tw would be under half
            % of the desired tw
            if((size(cuts_trial,1)-t(end))<(tw/2))
                t = t(1:end-1);
            end
    
            % Average high-level and cut for the time windows
            x_trial_highlevel =zeros(size(t,1),size(highlevel_trial_1ms,2));
            x_trial_lowlevel_auditory =zeros(size(t,1),size(lowlevel_auditory_trial_1ms,2));    
            x_trial_cuts =zeros(size(t,1),1);    
            for w = 1:size(t,1)
        
                % Time frame for this tw
                t0_window =t(w)+1;
                if(w==size(t,1))
                    t1_window = size(highlevel_trial_1ms,1);
                else 
                    t1_window = t(w+1);
                end
        
                x_trial_highlevel(w,:) = mean(highlevel_trial_1ms(t0_window:t1_window,:));
                x_trial_lowlevel_auditory(w,:) = mean(lowlevel_auditory_trial_1ms(t0_window:t1_window,:));
                if(sum(cuts_trial(t0_window:t1_window))>0)
                    x_trial_cuts(w,1) = 1;
                end
            end

            % Repeat the low-level auditory values for each pixel in the timewindow frame
            x_trial_lowlevel_auditory = repelem(x_trial_lowlevel_auditory,pix,1);
    
            % Repeat the high-level values for each pixel in the timewindow frame
            x_trial_highlevel = repelem(x_trial_highlevel,pix,1);

            % Repeat the cuts for each pixel in the timewindow frame
            x_trial_cuts = repelem(x_trial_cuts,pix,1);
        
            % Vectorize heatmap
            y_sub = reshape(trial_heatmaps,[],size(trial_heatmaps,3));
    
            % Create mid- and low-level models
            for w = 1:size(feature_frames_trial,1)
                % Vectorize tw model
                x = reshape(feature_frames_trial{w},[],size(feature_frames_trial{w},3));
                
                % Define what class information we have in this frame and make
                % dummy predictors for the midlevel features
                x_midlevel_dummy = zeros(size(x,1),9);
                objectClassIdx = unique(x(:,end));
                for c = 1:size(objectClassIdx,1)
                    x_midlevel_dummy(:,objectClassIdx(c)) = (x(:,11)==objectClassIdx(c));
                end
                
                % The object dummy variables add to one. Lets remove the "unknown" and
                % "outside video area" since they offer little relevant information
                % column
                x_midlevel_dummy = x_midlevel_dummy(:,1:7);
                
                % Collect the design matrix
                idx0 = 1+(w-1)*size(y_sub,1);
                idx1 = idx0+size(y_sub,1)-1;
                if(w==1)
                    x_trial_lowlevel = zeros(size(y_sub,2)*size(y_sub,1),10,'single');
                    x_trial_midlevel = zeros(size(y_sub,2)*size(y_sub,1),7,'single');
                    y_trial = zeros(size(y_sub,2)*size(y_sub,1),1);
                end
                
                if(w==size(feature_frames_trial,1) && size(feature_frames_trial,1)==(size(y_sub,2)+1)) % The last time window which is shorter than the specified tw has been calculated for the features but not for the heatmaps
                    break;
                else
                    x_trial_lowlevel(idx0:idx1,:) = x(:,1:end-1);
                    x_trial_midlevel(idx0:idx1,:) = x_midlevel_dummy;
                    y_trial(idx0:idx1,1) = y_sub(:,w);
                end
            end
    
            % Select windows
            if(selectWindow>1)
                nFrames = size(x_trial_lowlevel,1)/pix;
                idx = repelem(1:nFrames,pix,1);
                idx1 = idx(:);
                selectIdx = mod(idx1, selectWindow) == 1;
                x_trial_lowlevel = x_trial_lowlevel(selectIdx,:);
                x_trial_lowlevel_auditory = x_trial_lowlevel_auditory(selectIdx,:);
                x_trial_midlevel = x_trial_midlevel(selectIdx,:);
                x_trial_highlevel = x_trial_highlevel(selectIdx,:);
                x_trial_cuts = x_trial_cuts(selectIdx,:);
                y_trial = y_trial(selectIdx,:);
            end
            
            % Collect trial data
            x_lowlevel = vertcat(x_lowlevel,x_trial_lowlevel);
            x_lowlevel_auditory = vertcat(x_lowlevel_auditory,single(x_trial_lowlevel_auditory));
            x_midlevel = vertcat(x_midlevel,single(x_trial_midlevel));
            x_highlevel = vertcat(x_highlevel,single(x_trial_highlevel));
            x_cuts = vertcat(x_cuts,single(x_trial_cuts));

            y = vertcat(y,single(y_trial));
        end
    end

    % Combine the design matrix
    x = horzcat(zscore(x_lowlevel,[],'omitnan'),zscore(x_lowlevel_auditory,[],'omitnan'),zscore(x_highlevel,[],'omitnan'),x_midlevel,x_cuts);
    predictors = [cats_lowlevel';cats_auditory';cats_highlevel;cats_object(1:7);{'cuts'}];
    
    % Only few animals in the data and poor positive predictive value for
    % "animals" the category was excluded from regression also. Exclude it
    % here too.
    x(:,37) = [];
    predictors(37) = [];

    % Combine predictors by taking the mean based on the clustering result
    % (in R) of the predictors
    [x,predictors] = createClusterPredictors(x,predictors);
    x = single(x);

    % Normalize yy between 0-1 in the whole data 
    tmp = y;
    prctile_train = prctile(tmp,99.9);
    tmp = tmp./prctile_train;
    tmp(tmp<0) = 0;
    tmp(tmp>1) = 1;
    y = tmp;
    
    clearvars -except r_train models importances hyperparameters x y predictors dset tw path_models path_highlevel path_isc path_cuts path_heatmaps path_output pix resolution selectWindow cats_object cats_lowlevel cats_lowlevel_auditory cats_highlevel d include_trials weight_isc plotResults OptimSplits OptimTrees

    % Fit the optimized model to the dataset
    fprintf('Dset: %d, Training optimized model\n',d);
    t = templateTree('MaxNumSplits',OptimSplits);
    mdl = fitrensemble(x,y,'Method','Bag','NumLearningCycles',OptimTrees,'Learners',t); % Random forest
    mdl = compact(mdl);

    % Estimate the performance in the training dataset
    yhat = predict(mdl,x);
    r = corr(yhat,y);

    % Extract feature importance
    importance = predictorImportance(mdl);
    importance = importance./sum(importance);
    importance = array2table(importance);
    importance.Properties.VariableNames = predictors;

    % Save the results
    model = struct;
    model.Trained = char(datetime);
    model.TimeWindow = tw;
    model.TrainSet = dset{d};
    model.TrialsIncluded = include_trials{d};
    model.OptimSplits = OptimSplits;
    model.OptimNumTrees = OptimTrees;
    model.Model = mdl;
    model.Importance = importance;
    model.TrainR =r;
    save(sprintf('%s/trained_models/tw%d/trained_model_%s_tw%d.mat',path_output,tw,dset{d},tw),'model');
    toc;
end

%% Functions

function [xCluster,predictor_names_clusters] = createClusterPredictors(x,predictor_names)
% The function combines the clustered predictors. Numerical predictors are are averaged. 
% Categorical variables are combined by assigning one (1) if any of the variables are onw (1).

predictor_names_clusters = {'Auditory_RMS_&_roughness_diff','Auditory_spectral_information','Pleasant_situation','Object','Visual_movement','Luminance_&_entropy','Scene_cut','Talking','Body_parts','Eyes','Mouth','Face','Background','Auditory_RMS_&_roughness','Body_movement','Unpleasant_situation','Auditory_spectral_information_diff','Luminance_&_entropy_diff'};

cl1 = horzcat(x(:,strcmp(predictor_names,'RMSDiff')),x(:,strcmp(predictor_names,'RoughnessDiff')));
cl2 = horzcat(x(:,strcmp(predictor_names,'Spread')),x(:,strcmp(predictor_names,'ZeroCrossing')),x(:,strcmp(predictor_names,'AuditoryEntropy')),x(:,strcmp(predictor_names,'Centroid')),x(:,strcmp(predictor_names,'Rolloff85')));
cl3 = horzcat(x(:,strcmp(predictor_names,'playful')),x(:,strcmp(predictor_names,'pleasant_feelings')));
cl4 = horzcat(x(:,strcmp(predictor_names,'object')));
cl5 = horzcat(x(:,strcmp(predictor_names,'OpticFlow')),x(:,strcmp(predictor_names,'DifferentialEnergy')));
cl6 = horzcat(x(:,strcmp(predictor_names,'Luminance')),x(:,strcmp(predictor_names,'Entropy')),x(:,strcmp(predictor_names,'SpatialEnergyLF')),x(:,strcmp(predictor_names,'SpatialEnergyHF')));
cl7 = horzcat(x(:,strcmp(predictor_names,'cuts')));
cl8 = horzcat(x(:,strcmp(predictor_names,'talking')));
cl9 = horzcat(x(:,strcmp(predictor_names,'person')));
cl10 = horzcat(x(:,strcmp(predictor_names,'eyes')));
cl11 = horzcat(x(:,strcmp(predictor_names,'mouth')));
cl12 = horzcat(x(:,strcmp(predictor_names,'face')));
cl13 = horzcat(x(:,strcmp(predictor_names,'background')));
cl14 = horzcat(x(:,strcmp(predictor_names,'RMS')),x(:,strcmp(predictor_names,'Roughness')));
cl15 = horzcat(x(:,strcmp(predictor_names,'body_movement')));
cl16 = horzcat(x(:,strcmp(predictor_names,'aroused')),x(:,strcmp(predictor_names,'unpleasant_feelings')),x(:,strcmp(predictor_names,'aggressive')),x(:,strcmp(predictor_names,'pain')));
cl17 = horzcat(x(:,strcmp(predictor_names,'SpreadDiff')),x(:,strcmp(predictor_names,'ZeroCrossingDiff')),x(:,strcmp(predictor_names,'AuditoryEntropyDiff')),x(:,strcmp(predictor_names,'CentroidDiff')),x(:,strcmp(predictor_names,'Rolloff85Diff')));
cl18 = horzcat(x(:,strcmp(predictor_names,'LuminanceDiff')),x(:,strcmp(predictor_names,'EntropyDiff')),x(:,strcmp(predictor_names,'SpatialEnergyLFDiff')),x(:,strcmp(predictor_names,'SpatialEnergyHFDiff')));

% Combine the cluster features 
xCluster = zeros(size(x,1),18); 
xCluster(:,1) = mean(cl1,2);
xCluster(:,2) = mean(cl2,2);
xCluster(:,3) = mean(cl3,2);
xCluster(:,4) = cl4;
xCluster(:,5) = mean(cl5,2);
xCluster(:,6) = mean(cl6,2);
xCluster(:,7) = cl7;
xCluster(:,8) = cl8;
xCluster(:,9) = cl9;
xCluster(:,10) = cl10;
xCluster(:,11) = cl11;
xCluster(:,12) = cl12;
xCluster(:,13) = cl13;
xCluster(:,14) = mean(cl14,2);
xCluster(:,15) = cl15;
xCluster(:,16) = mean(cl16,2);
xCluster(:,17) = mean(cl17,2);
xCluster(:,18) = mean(cl18,2);

end




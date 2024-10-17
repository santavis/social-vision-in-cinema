%% Gaze Prediction Analysis: Test the trained random forest models
%
% Process:
%       1. Interpret the predictor contributions to the predictions using
%          feature importances.
%       2. Simulate predictions to undrstand how different predictors
%          influence the predictions.
%       3. Test models using cross-validation over datasets.
%           i)  Correlation
%           ii) Euclidean distance between the true most salient point and the most salient prediction 
%
% Severi Santavirta 27.12.2023

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

% What is the number of pixels in the model frames
pix = 4096;

% What is the frame resolution
resolution = [64,64];

% To save resources select every ith window
selectWindow = 1; % 1 = Include all time windows

% How many workers to use when sampling the pixels
npool = 4;

% Where to save the results
path_output = 'path/gaze_prediction/';

%% 1. Interpret models

% Open parallel pool
p = gcp('nocreate'); % If no pool, create new one.
if(isempty(p))
    p = parpool(npool);
end

for d = 1:size(dset,2)
    fprintf('Testing optimized model: Dset: %d, Reading data\n',d);
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
                
                % The object dummy variables add to zero. Lets remove the "unknown" and
                % "outside video area" since they offer little relevant information
                % column
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

    clearvars -except r_train models importances hyperparameters x predictors dset tw path_models path_highlevel path_isc path_cuts path_heatmaps path_output pix resolution selectWindow cats_object cats_lowlevel cats_lowlevel_auditory cats_highlevel d include_trials weight_isc plotResults OptimSplits OptimTrees

    % Load the trained model
    mdl = load(sprintf('path/gaze_prediction/trained_models/tw%d/trained_model_%s_tw%d.mat',tw,dset{d},tw));
    
    % Fetch overall feature importances measuring the overall magnitude of
    % the effect for every predictor
    if(d==1)
        importances = mdl.model.Importance;
    else
        importances = vertcat(importances,mdl.model.Importance);
    end

    % Social features are constant within each pixel in one time
    % window. Hence, their contributions can mainly emerge as
    % interaction with some pixel specific feature. To investigate
    % this, we also check for interaction terms of social features with
    % the 4 most important predictors (luminance, visual movement, eyes, mouth)
    N = 200000;
    M = 20;
    social_idx = [3,8,15,16];
    most_important_idx = [5,6,10,11];

    % For continous predictors choose randomly values from standard normal
    % disribution
    zscores = single((-3:0.001:3)');

    % Select random rows for simulation
    rng(d);
    idx = randi(size(x,1),[N,1]);
    xrand = x(idx,:);
    parfor pred = 1:size(social_idx,2)
        
        % Initialize predictions
        p = zeros(N*M,1,"single");
        xpos1 = zeros(N*M,1,"single");
        xpos2 = zeros(N*M,1,"single");

        % Select tested x values for social. For continuous variable, just select uniformly distributed random values form the zscores list
        rng(pred);
        test_x1 = zscores(randi(size(zscores,1),[N,M]));

        for pred2 = 1:size(most_important_idx,2)

            % Select the social predictor of interest
            interest = xrand(:,social_idx(pred));
            
            % Select the important predictor of interest
            interest2 = xrand(:,most_important_idx(pred2));

            % Select tested x values for important
            rng(pred*pred2+100);
            if(size(unique(interest2),1)==2) % Dummy
                
                % For dummy variable, create a randomly permuted vector
                % with balanced amount ofzeros and ones
                test_x2 = repmat([zeros(1, M/2,"single"), ones(1, M/2,"single")],N,1);
                [~, randomPerms] = sort(rand(N,M),2);
                test_x2 = test_x2(sub2ind([N,M], repmat((1:N)',1,M),randomPerms));

            else
                
                % For continuous variable, just select uniformly distributed random values form the zscores list 
                test_x2 = zscores(randi(size(zscores,1),[N,M]));
            end
            
            % Loop thorugh each randomly selected row
            for r = 1:size(xrand,1)
                fprintf('Predictor: %d, Row: %d/%d\n',pred,r,size(idx,1));
                
                % Select the simulated row
                xrow = xrand(r,:);
                rowdata = repmat(xrow,M,1);
                
                % Predict data with different predictor values in the two selected predictors while keeping
                % other predictors constant
                rowdata(:,social_idx(pred)) = test_x1(r,:)';
                rowdata(:,most_important_idx(pred2)) = test_x2(r,:)';

                prow = predict(mdl.model.Model,rowdata);
    
                % Normalize the predictions
                prow = prow-mean(prow);
    
                % Store
                idx0 = 1+((r-1)*size(test_x1,2));
                idx1 = (r*size(test_x1,2));
                p(idx0:idx1) = prow;
                xpos1(idx0:idx1) = test_x1(r,:)';
                xpos2(idx0:idx1) = test_x2(r,:)';
            end

            % Save the results
            tbl = horzcat(xpos1,xpos2,p);
            tbl = array2table(tbl);
            tbl.Properties.VariableNames = {predictors{social_idx(pred)},predictors{most_important_idx(pred2)},'y'};
            writetable(tbl,sprintf('path/gaze_prediction/interpretation/social_interactions/%s_%s_%s_interaction.csv',dset{d},predictors{social_idx(pred)},predictors{most_important_idx(pred2)}));

        end
    end

    % Loop through all predictors and estimate the main effects. For each predictor sample N rows from the X.
    % For dummy predictor test the effect of the variable by predicting
    % with 0 or 1 keeping others constant. For continous variables sample
    % M number of predictor values from the possible values and make
    % predictions with them.

    % Select randomly N rows from the data and loop over them
    rng(d);
    idx = randi(size(x,1),[N,1]);
    xrand = x(idx,:);
    res = cell(size(predictors,2),1);
    parfor pred = 1:size(predictors,2)

        % Select the predictor of interest
        interest = xrand(:,pred);

        % Initialize predictions
        if(size(unique(interest),1)==2) % Dummy
            p = zeros(2*N,1);
            xpos = zeros(2*N,1);
        else
            p = zeros(N*M,1);
            xpos = zeros(N*M,1);
        end
        
        for r = 1:size(xrand,1)
            fprintf('Predictor: %d, Row: %d/%d\n',pred,r,size(idx,1));
            xrow = xrand(r,:);

            % Select tested x values
            if(size(unique(interest),1)==2) % Dummy

                % For dummy variable, just test both possible values.
                test_x = [0;1];
                rowdata = repmat(xrow,2,1);
            else
                % For continuous variable, test uniformly distributed but
                % randomly selected zscores
                rng(pred*r);
                test_x = zscores(randi(size(zscores,1),[M,1]));
                rowdata = repmat(xrow,M,1);
            end

            % Predict data with different predictor values while keeping
            % other predictors constant
            rowdata(:,pred) = test_x;
            prow = predict(mdl.model.Model,rowdata);

            % Normalize the predictions
            prow = prow-mean(prow);

            % Store
            idx0 = 1+((r-1)*size(test_x,1));
            idx1 = (r*size(test_x,1));
            p(idx0:idx1) = prow;
            xpos(idx0:idx1) = test_x;
        end

        % Save the results
        tbl = horzcat(xpos,p);
        tbl = array2table(tbl);
        tbl.Properties.VariableNames = {'x','y'};
        tbl.predictor = repmat(predictors{pred},size(tbl,1),1);
        res{pred} = tbl;        
    end

    % Save results to a file
    for pred = 1:size(predictors,2)
        writetable(res{pred}(:,1:2),sprintf('%s/interpretation/%s_predictor_%d_influence.csv',path_output,dset{d},pred));
    end
end

 % Save the importance table
 writetable(importances,sprintf('%s/interpretation/predictor_importance.csv',path_output));

%% Evaluate the models agains each other (correlation & peak value distance)

r = zeros(size(dset,2),size(dset,2));
dist = zeros(size(dset,2),size(dset,2));
for d = 1:size(dset,2)
    fprintf('Dset: %d, Evaluating models\n',d);
    tic;

    % Load data
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
    
    clearvars -except r_train models importances hyperparameters x y predictors dset tw path_models path_highlevel path_isc path_cuts path_heatmaps path_output pix resolution selectWindow cats_object cats_lowlevel cats_lowlevel_auditory cats_highlevel d include_trials weight_isc plotResults r dist

    % Test all models agains the loaded dataset
    for model = 1:size(dset,2)
        mdl = load(sprintf('path/gaze_prediction/trained_models/tw%d/trained_model_%s_tw%d.mat',tw,dset{model},tw));

        % Correlation
        yhat = predict(mdl.model.Model,x);
        r(model,d) = corr(y,yhat(:,1));

        % Peak value distance
        frames = size(x,1)/pix;
        dist_frame = zeros(frames,1);
        for w = 1:frames
                
            % Vectorize the frame
            xW = x((w-1)*pix+1:w*pix,:);
            yW = y((w-1)*pix+1:w*pix,:);

            % Make predictions for that frame
            yhatImg = predict(mdl.model.Model,xW);

            % Reshape back to image
            yW = reshape(yW,64,64);
            yhatImg = reshape(yhatImg,64,64);

            % Identify peak values
            [maxTrue, linearIndexTrue] = max(yW(:));
            [rowTrue, colTrue] = ind2sub(size(yW), linearIndexTrue);
            [maxPred, linearIndexPred] = max(yhatImg(:));
            [rowPred, colPred] = ind2sub(size(yhatImg), linearIndexPred);

            % Calculate the absolute distance in pixels
            dist_frame(w,1) = sqrt((rowTrue - rowPred)^2 + (colTrue - colPred)^2);
        end
        dist(model,d) = median(dist_frame);
    end
end

% Save evaluation tables
dist = array2table(dist);
dist.Properties.VariableNames = dset; % The order of the rows is the same  so no need to save row names
writetable(dist,sprintf('%s/evaluation/peak_distance_matrix.csv',path_output));
r = array2table(r);
r.Properties.VariableNames = dset; % The order of the rows is the same  so no need to save row names
writetable(r,sprintf('%s/evaluation/correlation_matrix.csv',path_output));

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


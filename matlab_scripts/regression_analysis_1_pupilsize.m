%% Regression Analysis: Cross-validated multi-step OLS regression (PupilSize X [low-level & mid-level & high-level features)
% The pupil size is predicted with the clustered low-level, mid-level and high level features.
% Precess:
%           1.  Pupil size is independently modelled with each cluster
%               feature. A LOO cross-validated design is selected so that each
%               two experiments are used as the training set and then tested to the
%               leave-one-out experiment data. The features where the beta sign flips between any
%               cross-validation rounds are concluded not to have meaningful effect, since the direction of
%               the association is inconsistent. These features are
%               excluded from the next steps.
%
%           2.  Features with consistent direction of the association are
%               ordered based on the prediction power (correlation in the test set) so that
%               the first feature is the one with the highest prediction
%               power in the previous simple regressions. Next, we will add
%               the features one-by-one into a multiple regression model
%               and the ordering ensures that the features with the
%               best initial predictions are added first into the
%               model. LOO CV design is used here as well. This ensures
%               that each individual predictor would have independent
%               relationship with the variable of interest taking account
%               the possible correlations between the features.
%
%           3.  After each feature addition a permutation test is conducted
%               to assess whether the addition of the feature decreased the
%               prediction error more than would be expected by chance
%                    
%                   i)  To create the null distribution of test set
%                       predictions power (Pearson R)
%                       only the column of the newly added feature is shuffled (circular bootstrap), 
%                       so the previously validated model stays
%                       intact.

%                   ii) Feature is included if the test set R is significanlty lower (p<0.05) than would be
%                       expected by chance.
%
% Severi Santavirta 14.11.2023

tic;

%% INPUT

analysisName = 'pupil_ols_tw1000_shift1000'; % Choose the name based on the time window and selected pupil size shift
dependent = 'Pupil';
dset = {'localizer';'kasky';'conjuring'}; % Experiments 1-3
excluded = {'C08';'C27';'K05';'K15';'K19';'K20';'K24';'L096'}; % Excluded based on QC
input_eyedata = 'path/eyedata/subdata';
input_predictors_lowlevel = 'path/lowlevel';
input_predictors_highlevel = 'path/socialdata';
input_predictors_cuts = 'path/video_segmentation/cuts';
output = 'path/regression'; % where to store the results?

% For kasky and conjuring the last trial contains the end texts (no
% lifelike context), exclude those
include_trials = [{1:68},{1:25},{1:29}];

% Pupil size processing parameters
normalize = 50; % The duration of the normalization period in millisecond at the beginning of each trial (all data points are normalized to the mean of this interval)
shift = 1000; % How many milliseconds the Y should be shifted compared to X?
tw = 1000; % Downsampling time window in milliseconds

% Permutation parameters
nperm = 500; % How many permutations to run

%% Load and process pupil size data and predictors
% Pupil size is normalized within each trial to the initial pupil size, the
% pupil size is also shifted 1 second forward within each trial to correct
% for the pupil-reflex lag. Then the data is downsampled to 1 second
% intervals.

% For localizer the trials=videos are shown continuously and there were only
% 3 calibrations breaks (after trials 17, 34, 51) so the above mentioned procedures are not done for
% each "trial" in Localizer but for the real presentation breaks. For
% Conjuring and Kasky the presentations breaks are between every trial

% Loop over datasets
y = cell(size(dset,1),1);
x = cell(size(dset,1),1);
subjects = cell(size(dset,1),1);
trial_dset = cell(size(dset,1),1);
for d = 1:size(dset,1)

    % Find the data
    f = find_files(sprintf('%s/%s/',input_eyedata,dset{d}),'*.mat');
    
    % Exclude bad subjects
    [path,subs,ext] = fileparts(f);
    subs = setdiff(subs,excluded);
    subjects{d} = subs;
    
    % Loop over subjects
    for i = 1:size(subs,1)
        fprintf('%s: Reading eyedata: %d/%d\n',dset{d},i,size(subs,1));
        eyedata = load(sprintf('%s/%s%s',path{i},subs{i},ext{i}));
        pupil_sub = eyedata.subdata.pupil;
        
        % Define trials
        if(d==1 && i==1) % Localizer has different trials
            trial = zeros(size(eyedata.subdata.trial_indices,1),1);
            trial(1:find(eyedata.subdata.trial_indices==17,1,'last')) = 1;
            trial((find(eyedata.subdata.trial_indices==17,1,'last')+1):find(eyedata.subdata.trial_indices==34,1,'last')) = 2;
            trial((find(eyedata.subdata.trial_indices==34,1,'last')+1):find(eyedata.subdata.trial_indices==51,1,'last')) = 3;
            trial((find(eyedata.subdata.trial_indices==51,1,'last')+1):end) = 4;
        elseif(i==1)
            trial = eyedata.subdata.trial_indices;
        end

        % Drop excluded trials
        trials_included = ismember(trial,include_trials{d});
        trial = trial(trials_included);
        pupil_sub = pupil_sub(trials_included,:);
        
        % Process and collect pupil data
        % Pupil size is normalized seprately for each trial by the average
        % of the "normalize" period. Next the data is shifted "shift" forward.
        % Finally, data is downsampled to "tw" intervals
        if(i==1)
            [pupil_dset,trial_ds] = processPupil(pupil_sub,trial,normalize,shift,tw);
        else
            pupil_dset(:,i) = processPupil(pupil_sub,trial,normalize,shift,tw);
        end
        
        % Load subjectwise gaze class regressors and transform to dummy
        % variables. "Unknown", "Outside video area" and "animals" are
        % excludeded. Not enough animals in the stimulus and others are
        % uninteresting.
        if(i==1)
            gaze = zeros(size(trial,1),size(subs,1),6);
        end
        gaze_class = eyedata.subdata.gaze_class;
        gaze_class = gaze_class(trials_included,:);
        gaze_dummy = zeros(size(gaze_class,1),size(unique(gaze_class),1));
        for c = 1:size(unique(gaze_class),1)
            gaze_dummy(:,c) = gaze_class==c;
        end
        gaze(:,i,:) = gaze_dummy(:,[1:4,6:7]);
    end
    
    % Collect datasets
    y{d} = pupil_dset;
    trial_dset{d} = trial_ds;
    
    % Process lowlevel predictors
    predictors_lowlevel = load(sprintf('%s/%s/lowlevel_data_1ms.mat',input_predictors_lowlevel,dset{d}));
    predictors_lowlevel = table2array(predictors_lowlevel.lowlevel.data);
    predictors_lowlevel = predictors_lowlevel(trials_included,:);
    predictors_lowlevel_processed = processPredictors(predictors_lowlevel,trial,1,shift,tw);
    
    % Process gaze class predictors
    predictors_gaze_processed = processGaze(gaze,trial,shift,tw);
    
    % Process cut predictor
    predictor_cuts = load(sprintf('%s/%s/gigatrack_%s_scene_cut_regressor.mat',input_predictors_cuts,dset{d},dset{d}));
    predictor_cuts = predictor_cuts.regressor;
    predictor_cuts = predictor_cuts(trials_included,:);
    predictor_cuts = processCuts(predictor_cuts.cut,trial,shift,tw);
    
    % Process higlevel predictors
    predictors_highlevel = load(sprintf('%s/%s/%s_eyetracking_social_regressors_1ms.mat',input_predictors_highlevel,dset{d},dset{d}));
    predictors_highlevel = table2array(predictors_highlevel.regressors_1ms(:,1:8));
    predictors_highlevel = predictors_highlevel(trials_included,:);
    predictors_highlevel_processed = processPredictors(predictors_highlevel,trial,1,shift,tw);
    
    % Get the predictor names
    predictors_lowlevel = load(sprintf('%s/%s/lowlevel_data_1ms.mat',input_predictors_lowlevel,dset{1}));
    predictors_highlevel = load(sprintf('%s/%s/%s_eyetracking_social_regressors_1ms.mat',input_predictors_highlevel,dset{1},dset{1}));
    predictors_mid = load(sprintf('%s/%s/subjects/L001.mat',input_eyedata,dset{1}));
    predictors_mid = predictors_mid.subdata.gaze_class_catalog([1:4,6:7])';
    predictor_names = horzcat(predictors_lowlevel.lowlevel.data.Properties.VariableNames,predictors_mid,'cuts_dummy',predictors_highlevel.regressors_1ms.Properties.VariableNames(1:8));
    
    % Individual models for each subject
    for s = 1:size(subs,1)
        xs = horzcat(predictors_lowlevel_processed,squeeze(predictors_gaze_processed(:,s,:)),predictor_cuts,predictors_highlevel_processed);
        
        % Create cluster predictors based on the clustering results (analyszed in R)
        [xsCluster,predictor_names_clusters] = createClusterPredictors(xs,predictor_names);
        
        if(s==1)
            xss = zeros(size(xs,1),size(subs,1),size(xsCluster,2));
        end
        xss(:,s,:) = xsCluster;   
    end
    
    x{d} = single(xss);

end

%% Run simple OLS regression for each regressor to find whether the regressor has any consistent effect
% Select only regressors whit consistent CV results (same sign for the beta
% for all rergression rounds) for further analyses

include = zeros(1,size(predictor_names_clusters,2)); % Collect consistent regressors
betaFeature = [];
R = [];
n=0;
for r = 1:size(predictor_names_clusters,2)
    fprintf('Running simple regressions %d/%d\n',r,size(predictor_names_clusters,2));
    xr{1,1} = x{1}(:,:,r);
    xr{2,1} = x{2}(:,:,r); 
    xr{3,1} = x{3}(:,:,r); 
    results = gigatrack_ols_cv(y,xr,dset,predictor_names_clusters(r),subjects,true);

    % Check whether the test beta is consistent over all CVs
    for cv = 1:size(results,1)
        cvres = results{cv};
        cv_performance(1,cv) = cvres.TestBetas(2);
        cv_performance(2,cv) = cvres.TestMSE;
        cv_performance(3,cv) = cvres.TestR;
        cv_performance(4,cv) = cvres.TrainMSE;
        cv_performance(5,cv) = cvres.TrainR;
    end
    if(sum(cv_performance(1,:)>0)==size(results,1) || sum(cv_performance(1,:)<0)==size(results,1) ) % Consistent betas
        n=n+1;
        include(1,r) = 1;
        betaFeature(n,1) = mean(cv_performance(1,:));
        R = vertcat(R,cv_performance(3,:));
    end
end

% Select included features
xm{1,1} = x{1}(:,:,logical(include));
xm{2,1} = x{2}(:,:,logical(include));
xm{3,1} = x{3}(:,:,logical(include));
predictor_names_m  = predictor_names_clusters(logical(include));

% Sort included features based on their R in the test
% dataset (first regressor has the highest R when weighting all CV rounds equally)
rRank = zeros(size(R));
for cv = 1:size(R,2)
    % Get the rank of each predictor in the CV round
    [~,idx] = sort(R(:,cv),'descend');
    rRank(idx,cv) = 1:size(rRank,1);
end

% Take the average of the ranks to define the order of the regressors in
% the next analysis phase
rDiffRank = mean(rRank,2);
[rDiffRank,idx] = sort(rDiffRank,'ascend');
predictor_names_m = predictor_names_m(idx);
betaFeature = betaFeature(idx);
R = R(idx,:);
xm{1,1} = xm{1}(:,:,idx);
xm{2,1} = xm{2}(:,:,idx);
xm{3,1} = xm{3}(:,:,idx);

% Now, the regressors are ordered based on their out-of-sample predictive power

% Collect results
simpleResults = array2table(horzcat(betaFeature,mean(R,2),rDiffRank));
simpleResults.Properties.RowNames = predictor_names_m;
simpleResults.Properties.VariableNames = {'beta','R','R_rank'};

%% Run multiple regression by adding regressors one by one to assess the statistical significance of the features
% After each addition, evaluate whether the R
% in the test set increased more than would be expected by chance using permutation testing.
% If the addition of the given regressor did not increase the R significantly,
% the regressor is dropped and ít is concluded that there is not enough
% evidence that the feature has an effect.

include = zeros(1,size(predictor_names_m,2)); % Collect regressors with above chance prediction power
n = 0; % Count significant predictors

RTrueFinal = [];
RChanceFinal = [];
betasTrueFinal = [];
pvals = [];
pvalsAll = ones(1,size(predictor_names_m,2));
counter = 0;
for r = 1:size(predictor_names_m,2)
    
    % Select the included regressors for this round
    include_propose = include;
    include_propose(r) = 1;
    xr{1,1} = xm{1}(:,:,logical(include_propose));
    xr{2,1} = xm{2}(:,:,logical(include_propose));
    xr{3,1} = xm{3}(:,:,logical(include_propose));
    predictor_names_r = predictor_names_m(logical(include_propose));
    
    % Run regression with the selected regressors
    results = gigatrack_ols_cv(y,xr,dset,predictor_names_r,subjects,true);
    
    % Collect betas and R
    RTrue = zeros(1,size(results,1));
    betasRound = zeros(size(results,1),size(predictor_names_r,2)+1);
    for cv = 1:size(results,1)
        cvres = results{cv};
        RTrue(cv) = cvres.TestR;
        betasRound(cv,:) = results{cv}.TestBetas;
    end

    % Define cutpoints where the null data are bootsrapped (one cutpoint for each permutation)
    cutpoints = zeros(size(dset,1),nperm);
    for d = 1:size(dset,1)
        observations = size(xr{d},1);
        counter = counter+1;
        rng(counter); % Different random cut points for each dataset, and each permutation round
        cutpoints(d,:) = randi([1,observations-1],nperm,1);
    end
    
    % Permute regression over the null models
    RChance = zeros(nperm,size(results,1));
    for perm = 1:nperm
        fprintf('Permuting regression data: %d/%d\n',perm,nperm);

        % Create null model
        xNull = createNullModelShuffleLast(xr,cutpoints(:,perm),dset,true);

        % Run cross-validated regression with the shuffled dataset
        resultsChance = gigatrack_ols_cv(y,xNull,dset,predictor_names_r,subjects,true);

        % Assess chance performance
        RChancePerm = zeros(1,size(resultsChance,1));
        for cv = 1:size(resultsChance,1)
            cvres = resultsChance{cv};
            RChancePerm(cv) = cvres.TestR;
        end
        RChance(perm,:) = RChancePerm;
    end
    
    % When adding the first regressors we compare whether the R is increased more than expected by chance. Calculate this over all CVs.
    chanceBetter = 0;
    for cv = 1:size(results,1)
        chanceBetter = chanceBetter + sum(RChance(:,cv)>RTrue(1,cv));
    end
    pval = chanceBetter/(nperm*size(results,1));
    pvalsAll(r) = pval; 
    
    % When adding the first regressor, check only whether the p-value < 0.05. For other rounds check if p-value < 0.05 and the R2 is better than in the last round
    if(pval<0.05)
        n=n+1; % New regressor added
        include = include_propose; % Include this model as the starting point for the next round
        
        % Extract the betas to investigate whether there are sign flips in later
        % rounds when adding new regressors into the model (may happen if the regressors are highly correlated)
        betasTrue = nan(1,size(predictor_names_m,2));
        for rr = 1:size(betasRound,2)
            if(sum(betasRound(:,rr)<0)==size(betasRound,1) || sum(betasRound(:,rr)>0)==size(betasRound,1)) % Still the beta does not chance sign between CVs
                betasTrue(rr) = mean(betasRound(:,rr));
            else
                betasTrue(rr) = Inf; % To mark the sign flip;
            end
        end
        betasTrueFinal = vertcat(betasTrueFinal,betasTrue);
        
        % Collect true and chance level model fit estimates for this model
        RTrueFinal = vertcat(RTrueFinal,RTrue);
        RChanceFinal = vertcat(RChanceFinal,mean(RChance));
        
        % Collect p-value
        pvals = vertcat(pvals,pval);
    end
end

%% Collect and save results

% Final model
xfinal = cell(size(y));
xfinal{1,1} = xm{1}(:,:,logical(include));
xfinal{2,1} = xm{2}(:,:,logical(include));
xfinal{3,1} = xm{3}(:,:,logical(include));

% Significant predictors
predictors_significant = predictor_names_m(logical(include));

% Run final model
results = gigatrack_ols_cv(y,xfinal,dset,predictor_names_r,subjects,true);

% R values
RTrue = array2table(RTrueFinal);
RTrue.Properties.VariableNames = dset;
RTrue.Properties.RowNames = predictors_significant;
RChance = array2table(RChanceFinal);
RChance.Properties.VariableNames = dset;
RChance.Properties.RowNames = predictors_significant;

% Betas
betasFinal = betasTrueFinal;
betasFinal(:,find(isnan(betasTrueFinal(end,:)),1):end) = [];
betasFinal = array2table(betasFinal);
betasFinal.Properties.VariableNames = [{'intercept'},predictors_significant];

% Collect results
resultsFinal = struct;
resultsFinal.AnalysisTime = char(datetime);
resultsFinal.DependentVariable = dependent;
resultsFinal.TimeWindow = tw;
resultsFinal.Shift = shift;
resultsFinal.Permutations = nperm;
resultsFinal.InitialFeatures = predictor_names_clusters;
resultsFinal.ConsistentFeatures = predictor_names_m;
resultsFinal.SimpleRegressionResults = simpleResults;
resultsFinal.SignificantFeatures = predictors_significant;
resultsFinal.SignificantFeaturePvalues = pvals;
resultsFinal.ConsistentFeaturePvalues = pvalsAll;
resultsFinal.StepwiseR = RTrue;
resultsFinal.StepwiseRChance = RChance;
resultsFinal.StepwiseBetas = betasFinal;
resultsFinal.FinalModelResults = results;

% Save
save(sprintf('%s/%s.mat',output,analysisName),'resultsFinal');
toc;

%% Functions

function [pupil_sub_shift_ds,trial_shift_ds] = processPupil(pupil,trial,normalize_dur,shift_dur,downsample_dur)
% Function takes the 1ms "pupil" time series and the "trial" indices for
% each millisecond as input and normalizes the data separately for each trial. The data is
% normalized with the mean pupil size of the first milliseconds specified
% in "normalize_dur". Next the trialwise data is shifted forward the amount 
% specified in "shift_dur" (in milliseconds). Finally, the data is downsampled by
% averaging in time windows specified in "downsample_dur" (in milliseconds)
%
% Severi Santavirta 1.11.2023

    for tr = 1:size(unique(trial),1)

        pupil_trial = pupil(trial==tr);

        % Normalization: Divide with the mean of the first milliseconds
        pupil_trial = pupil_trial./(mean(pupil_trial(1:normalize_dur)));

        % Shift pupil forward (cut from the beginning)
        pupil_trial_shift = pupil_trial((shift_dur+1):end);

        % Downsample
        t = (0:downsample_dur:size(pupil_trial_shift,1))';
        % Combine last and second last tw of the trial if the last tw would be under half
        % of the desired tw
        if((size(pupil_trial_shift,1)-t(end))<(downsample_dur/2))
            t = t(1:end-1);
        end
        pupil_trial_shift_ds = zeros(size(t,1),1);
        for ti = 1:(size(t,1))
            t0 = t(ti)+1;
            if(ti==size(t,1))
                t1 = size(pupil_trial_shift,1);
            else
                t1 = t(ti+1);
            end
            pupil_trial_shift_ds(ti) = mean(pupil_trial_shift(t0:t1));
        end

        % Record trial indices
        if(tr==1)
            trial_shift_ds = repmat(tr,size(pupil_trial_shift_ds,1),1);
            pupil_sub_shift_ds = pupil_trial_shift_ds;
        else
            trial_shift_ds = vertcat(trial_shift_ds,repmat(tr,size(pupil_trial_shift_ds,1),1));
            pupil_sub_shift_ds = vertcat(pupil_sub_shift_ds,pupil_trial_shift_ds);
        end
    end  
end
function [predictors_sub_shift_ds,trial_shift_ds] = processPredictors(predictors,trial,standardize,shift_dur,downsample_dur)
% Function takes the 1ms "predictors" time series and the "trial" indices for
% each millisecond as input and standardizes the data if "standardize"=true. Next the trialwise data is shifted backward the amount 
% specified in "shift_dur" (in milliseconds). Finally, the data is downsampled by
% averaging in time windows specified by "downsample_dur" (in milliseconds)
%
% Severi Santavirta 1.11.2023

    for tr = 1:size(unique(trial),1)

        predictors_trial = predictors(trial==tr,:);

        % Shift predictors backward (cut from the end)
        predictors_trial_shift = predictors_trial(1:(end-shift_dur),:);

        % Downsample
        t = (0:downsample_dur:size(predictors_trial_shift,1))';
        
        % Combine last and second last tw of the trial if the last tw would be under half
        % of the desired tw
        if((size(predictors_trial_shift,1)-t(end))<(downsample_dur/2))
            t = t(1:end-1);
        end
        
        predictors_trial_shift_ds = zeros(size(t,1),size(predictors,2));
        for ti = 1:(size(t,1))
            t0 = t(ti)+1;
            if(ti==size(t,1))
                t1 = size(predictors_trial_shift,1);
            else
                t1 = t(ti+1);
            end
            predictors_trial_shift_ds(ti,:) = mean(predictors_trial_shift(t0:t1,:));
        end

        % Record trial indices
        if(tr==1)
            trial_shift_ds = repmat(tr,size(predictors_trial_shift_ds,1),1);
            predictors_sub_shift_ds = predictors_trial_shift_ds;
        else
            trial_shift_ds = vertcat(trial_shift_ds,repmat(tr,size(predictors_trial_shift_ds,1),1));
            predictors_sub_shift_ds = vertcat(predictors_sub_shift_ds,predictors_trial_shift_ds);
        end
    end
    
    % Standardize
    if(standardize)
        predictors_sub_shift_ds = zscore(predictors_sub_shift_ds);
    end
end
function [predictors_sub_shift_ds,trial_shift_ds] = processCuts(predictor,trial,shift_dur,downsample_dur)
% Function takes the 1ms "cuts" time series and the "trial" indices for
% each millisecond as input. The trialwise data is shifted backward the amount 
% specified in "shift_dur" (in milliseconds). Finally, the data is downsampled by
% in time windows specified in "downsample_dur" (in milliseconds) by
% assigning 1 (cut in the time window) or 0 (no cut in the time window).
%
% Severi Santavirta 1.11.2023

    for tr = 1:size(unique(trial),1)

        predictors_trial = predictor(trial==tr,:);

        % Shift predictors backward (cut from the end)
        predictors_trial_shift = predictors_trial(1:(end-shift_dur),:);

        % Downsample
        t = (0:downsample_dur:size(predictors_trial_shift,1))';
        
        % Combine last and second last tw of the trial if the last tw would be under half
        % of the desired tw
        if((size(predictors_trial_shift,1)-t(end))<(downsample_dur/2))
            t = t(1:end-1);
        end
        
        predictors_trial_shift_ds = zeros(size(t,1),1);
        for ti = 1:(size(t,1))
            t0 = t(ti)+1;
            if(ti==size(t,1))
                t1 = size(predictors_trial_shift,1);
            else
                t1 = t(ti+1);
            end
            % Cut in this tw
            if(any(predictors_trial_shift(t0:t1)))
                
                % Simple regressor 
                predictors_trial_shift_ds(ti,1) = 1; 
                
            end

        end

        % Record trial indices
        if(tr==1)
            trial_shift_ds = repmat(tr,size(predictors_trial_shift_ds,1),1);
            predictors_sub_shift_ds = predictors_trial_shift_ds;
        else
            trial_shift_ds = vertcat(trial_shift_ds,repmat(tr,size(predictors_trial_shift_ds,1),1));
            predictors_sub_shift_ds = vertcat(predictors_sub_shift_ds,predictors_trial_shift_ds);
        end
    end
end
function [predictors_sub_shift_ds,trial_shift_ds] = processGaze(predictors,trial,shift_dur,downsample_dur)
% Function takes the 1ms "gaze" time series and the "trial" indices for
% each millisecond as input. The trialwise data is shifted backward the amount 
% specified in "shift_dur" (in milliseconds). Finally, the data is downsampled
% in time windows specified in "downsample_dur" (in milliseconds) calculating in how many time points within te time window
% the subject was watching the given class (0 = no time wathcing the class, 1 = the subject watched the class for the whole tw).

% Severi Santavirta 1.11.2023

    for tr = 1:size(unique(trial),1)

        predictors_trial = predictors(trial==tr,:,:);

        % Shift predictors backward (cut from the end)
        predictors_trial_shift = predictors_trial(1:(end-shift_dur),:,:);

        % Downsample
        t = (0:downsample_dur:size(predictors_trial_shift,1))';
        
        % Combine last and second last tw of the trial if the last tw would be under half
        % of the desired tw
        if((size(predictors_trial_shift,1)-t(end))<(downsample_dur/2))
            t = t(1:end-1);
        end
        
        predictors_trial_shift_ds = zeros(size(t,1),size(predictors_trial_shift,2),size(predictors_trial_shift,3));
        for ti = 1:(size(t,1))
            t0 = t(ti)+1;
            if(ti==size(t,1))
                t1 = size(predictors_trial_shift,1);
            else
                t1 = t(ti+1);
            end
            % Loop over predictors
            for r = 1:size(predictors,3)
                predictors_trial_shift_ds(ti,:,r) = any(predictors_trial_shift(t0:t1,:,r));
            end
        end

        % Record trial indices
        if(tr==1)
            trial_shift_ds = repmat(tr,size(predictors_trial_shift_ds,1),1);
            predictors_sub_shift_ds = predictors_trial_shift_ds;
        else
            trial_shift_ds = vertcat(trial_shift_ds,repmat(tr,size(predictors_trial_shift_ds,1),1));
            predictors_sub_shift_ds = vertcat(predictors_sub_shift_ds,predictors_trial_shift_ds);
        end
    end
end
function xBootsrapped = createNullModelShuffleLast(x,cutpoint,dset,subjectwiseModels)
% The function bootsraps the last column of the x matric circularly and
% stores the nullModels after each shuffle
    
    % Shuffle circularly and return the bootsrapped dataset
    xBootsrapped = x;

    % Shuffle the predictors
    for d = 1:size(dset,1)

        % x for this dataset
        xBootsrapped_dset = xBootsrapped{d};
        
        % Shuffle the last column circularly
        if(subjectwiseModels && length(size(xBootsrapped_dset))==3) % Individual models for each subject, multiple regression
            xBootsrapped_dset(:,:,end) = vertcat(xBootsrapped_dset(cutpoint+1:end,:,end),xBootsrapped_dset(1:cutpoint,:,end));
            xBootsrapped{d} = single(xBootsrapped_dset);
        elseif(subjectwiseModels) % Individual models for each subject (only one column)
            xBootsrapped_dset = vertcat(xBootsrapped_dset(cutpoint+1:end,:),xBootsrapped_dset(1:cutpoint,:));
            xBootsrapped{d} = single(xBootsrapped_dset);
        else % Simple regression or same model for all subjects (fully stimulus dependent model)
            xBootsrapped_dset(:,end) = vertcat(xBootsrapped_dset(cutpoint+1:end,end),xBootsrapped_dset(1:cutpoint,end));
            xBootsrapped{d} = single(xBootsrapped_dset);
        end
    end
end
function [xCluster,predictor_names_clusters] = createClusterPredictors(x,predictor_names)
% The function combines the clustered predictors. Numerical predictors are are averaged. 
% Categorical variables are combined by assigning one (1) if any of the variables are onw (1).

predictor_names_clusters = {'Auditory_RMS_&_roughness_diff','Auditory_spectral_information','Pleasant_situation','Object','Visual_movement','Luminance_&_entropy','Scene_cut','Talking','Body_parts','Face','Background','Auditory_RMS_&_roughness','Body_movement','Unpleasant_situation','Auditory_spectral_information_diff','Luminance_&_diff'};

cl1 = horzcat(x(:,strcmp(predictor_names,'RMSDiff')),x(:,strcmp(predictor_names,'RoughnessDiff')));
cl2 = horzcat(x(:,strcmp(predictor_names,'Spread')),x(:,strcmp(predictor_names,'ZeroCrossing')),x(:,strcmp(predictor_names,'AuditoryEntropy')),x(:,strcmp(predictor_names,'Centroid')),x(:,strcmp(predictor_names,'Rolloff85')));
cl3 = horzcat(x(:,strcmp(predictor_names,'playful')),x(:,strcmp(predictor_names,'pleasant_feelings')));
cl4 = horzcat(x(:,strcmp(predictor_names,'object')));
cl5 = horzcat(x(:,strcmp(predictor_names,'OpticFlow')),x(:,strcmp(predictor_names,'DifferentialEnergy')));
cl6 = horzcat(x(:,strcmp(predictor_names,'Luminance')),x(:,strcmp(predictor_names,'Entropy')),x(:,strcmp(predictor_names,'SpatialEnergyLF')),x(:,strcmp(predictor_names,'SpatialEnergyHF')));
cl7 = horzcat(x(:,strcmp(predictor_names,'cuts_dummy')));
cl8 = horzcat(x(:,strcmp(predictor_names,'talking')));
cl9 = horzcat(x(:,strcmp(predictor_names,'person')));
cl10 = horzcat(x(:,strcmp(predictor_names,'eyes')),x(:,strcmp(predictor_names,'mouth')),x(:,strcmp(predictor_names,'face')));
cl11 = horzcat(x(:,strcmp(predictor_names,'background')));
cl12 = horzcat(x(:,strcmp(predictor_names,'RMS')),x(:,strcmp(predictor_names,'Roughness')));
cl13 = horzcat(x(:,strcmp(predictor_names,'body_movement')));
cl14 = horzcat(x(:,strcmp(predictor_names,'aroused')),x(:,strcmp(predictor_names,'unpleasant_feelings')),x(:,strcmp(predictor_names,'aggressive')),x(:,strcmp(predictor_names,'pain')));
cl15 = horzcat(x(:,strcmp(predictor_names,'SpreadDiff')),x(:,strcmp(predictor_names,'ZeroCrossingDiff')),x(:,strcmp(predictor_names,'AuditoryEntropyDiff')),x(:,strcmp(predictor_names,'CentroidDiff')),x(:,strcmp(predictor_names,'Rolloff85Diff')));
cl16 = horzcat(x(:,strcmp(predictor_names,'LuminanceDiff')),x(:,strcmp(predictor_names,'EntropyDiff')),x(:,strcmp(predictor_names,'SpatialEnergyLFDiff')),x(:,strcmp(predictor_names,'SpatialEnergyHFDiff')));

% Combine the cluster features 
xCluster = zeros(size(x,1),16); 
xCluster(:,1) = mean(cl1,2);
xCluster(:,2) = mean(cl2,2);
xCluster(:,3) = mean(cl3,2);
xCluster(:,4) = cl4;
xCluster(:,5) = mean(cl5,2);
xCluster(:,6) = mean(cl6,2);
xCluster(:,7) = cl7;
xCluster(:,8) = cl8;
xCluster(:,9) = cl9;
xCluster(:,10) = sum(cl10,2)>0;
xCluster(:,11) = cl11;
xCluster(:,12) = mean(cl12,2);
xCluster(:,13) = cl13;
xCluster(:,14) = mean(cl14,2);
xCluster(:,15) = mean(cl15,2);
xCluster(:,16) = mean(cl16,2);

end
function results = gigatrack_ols_cv(y,x,dset_names,x_names,subjects,subjectwiseModels)
% Leave-one-set out cross-validated OLS regression. At least two datasets are
% needed to split the data into train and test sets. All
% variable normalizations should be done before, since this function does
% not implement any variable modifications
%
% INPUT
%           y                   = Nx1 cell array, where the dependent variable vector of each dataset are stored to their own rows
%           x                   = Nx1 cell array, where the independent predictor matrices of each dataset are stored to their own rows
%           dset_names          = names for the dataset (to store in model details)
%           x_names             = names of the predictors (to store in model details)
%           subjects            = Nx1 names of the subjects (to store in model details)
%           subjectwiseModels   = true, if there are individual models for each subject in x
%
% OUTPUT
%           results = struct, with results and optimization information
%
% Severi Santavirta, 14.11.2023

% Check inputs
if(size(y,1)<2)
    error('Under 2 datasets, not possible to split to train and test datasets');
end
if(size(y,1)~=size(x,1))
    error('X and Y have different number of datasets');
end
for i = 1:size(y,1)
    if(size(y{i},1)~=size(x{i}))
        error('X and Y have different number of observations for at least one dataset');
    end
end

% Test set idx
test_set_idx = (1:size(y,1))';

% Run OLS regression for each CV round
results = cell(size(test_set_idx,1),1);
for cv = 1:size(test_set_idx,1) % Use cv as the test set index
   
    % Define the train and test sets
    trainsets = (1:size(y,1))';
    trainsets = trainsets(test_set_idx~=cv);
    testset = cv;
    y_train = y(trainsets);
    y_test = y(testset);
    x_train = x(trainsets);
    x_test = x(testset);
    subjects_train = subjects(trainsets);
    subjects_test = subjects(testset);
    desc = sprintf('Test set: %s',dset_names{cv});
    
    % Run OLS regression for the CV
    results{cv} = gigatrack_ols_regression(y_train,y_test,x_train,x_test,x_names,subjects_train,subjects_test,subjectwiseModels,desc);
    
end

end
function results = gigatrack_ols_regression(Ytrain,Ytest,Xtrain,Xtest,Xnames,subjectsTrain,subjectsTest,subjectModels,desc)
% Function calculates prediction error of a OLS regression model where the
% model is fit in the training set and then prediction error (MSE or R2) is
% calculated for the predictions in the test set

% INPUT
%       Xtrain        = D(dataset) x 1 cell array containing design matrices for train datasets. Each cell should contain a design matrix for a specidic training dataset.
%                          Use the following formatting
%                             1. time x feature           = same model for each subject
%                             2. time x subject           = one regressor, individual regressor for each subject
%                             3. time x subject x feature = multiple regression, individiaul regressors for each
%
%       Xtest         = 1 x 1 cell array containing the design matric for the
%                       test set. Formatted as described above.
%
%       Ytrain        = dataset x 1 cell array containing dependent
%                       datasets for training sets. Each cell should contain
%                       the dependent variables for each subject in the given
%                       dataset (time x subject)
% 
%       Ytest         = 1 x 1 cell array containing dependent
%                       data for the test set. (time x subject)
%
%       subjectModels = true (formatting of the design matrices is either 2. or 3.)
%                       false (formatting of the design matrices is 1.)
%
% OUTPUT
%       results        = betas and fit measures
%
% Severi Santavirta 14.11.2023

% Loop through all training sets
for dt = 1:size(Ytrain,1)
    
    % Initialize outcome matrices
    if(subjectModels && length(size(Xtrain{dt}))==3) % Individual models for each subject, multiple regression
        betas_dset = zeros(size(Ytrain{dt},2),size(Xtrain{dt},3)+1);
    elseif(subjectModels) % Individual models for each subject, simple regression
        betas_dset = zeros(size(Ytrain{dt},2),2);
    else % Same model for all subjects (fully stimulus dependent model)
        betas_dset = zeros(size(Ytrain{dt},2),size(Xtrain{dt},2));
    end
    mse_dset = zeros(size(Ytrain{dt},2),1);
    r_dset = zeros(size(Ytrain{dt},2),1);
    yhat_dset = zeros(size(Ytrain{dt}));

    % Loop through subjects in the dataset
    for s = 1:size(Ytrain{dt},2)
        
        % Dependent variable for this subjects
        y = Ytrain{dt}(:,s);

        % Design matrix for this subject
        if(subjectModels && length(size(Xtrain{dt}))==3) % Individual models for each subject, multiple regression
            x = squeeze(Xtrain{dt}(:,s,:));
        elseif(subjectModels) % Individual models for each subject, simple regression
            x = Xtrain{dt}(:,s);
        else % Same model for all subjects (fully stimulus dependent model)
            x = Xtrain{dt};
        end
        % Fit the model for the subject data
        mdl = fitlm(x,y);

        % Collect betas and model fits from the model object
        betas_dset(s,:) = mdl.Coefficients.Estimate;
        mse_dset(s,1) = sum((yhat_dset(:,s)-y).^2)/size(yhat_dset(:,s),1); % MSE
        r_dset(s,1) = corr(mdl.Fitted,y); % R
        yhat_dset(:,s) = mdl.Fitted;
    end

    % Collect betas and model fits
    if(dt==1)
        betas_train = betas_dset;
        mse_train = mse_dset;
        r_train = r_dset;
        dset_idx = repmat(dt,size(r_dset,1),1);
        
    else
        betas_train = vertcat(betas_train,betas_dset);
        mse_train = vertcat(mse_train,mse_dset);
        r_train = vertcat(r_train,r_dset);
        dset_idx = vertcat(dset_idx,repmat(dt,size(r_dset,1),1));
    end
    yhat_train{dt,1} = yhat_dset;

end

% Calculate the mean prediction error in the train set
% Some outlier subjects may have a big influence on the averge MSE, and we
% exclude very unlikely subjects from the calculations (MSE abs(zscore)>3.89 =>p<0.0001)
idx_train_reliable = abs(zscore(mse_train))<3.89;
mse_train_reliable = mse_train(idx_train_reliable);
mse_train_mean = mean(mse_train_reliable);

% Calculate mean r (exlude the same subjects than from the mse calculation)
r_train_reliable = r_train(idx_train_reliable);
r_train_mean = mean(r_train_reliable);

% Fit the trained model to the test set (very unreliable excluded)
betas_test = mean(betas_train(idx_train_reliable,:));

% Loop over subjects in the test set
r_test = zeros(size(Ytest{1},2),1);
mse_test = zeros(size(Ytest{1},2),1);
yhat_test = zeros(size(Ytest{1}));
for s = 1:size(Ytest{1},2)
    
    % Dependent variable for this test subject
    y = Ytest{1}(:,s);
    
    % Prediction
    if(subjectModels && length(size(Xtest{1}))==3) % Individual models for each subject, multiple regression
        yhat = sum([ones(size(Xtest{1},1),1) squeeze(Xtest{1}(:,s,:))].*betas_test,2);
    elseif(subjectModels) % Individual models for each subject, simple regression
        yhat = sum([ones(size(Xtest{1},1),1) Xtest{1}(:,s)].*betas_test,2);
    else % Same model for all subjects (fully stimulus dependent model)
        yhat = sum([ones(size(Xtest{1},1),1) Xtest{1}].*betas_test,2);
    end

    % Prediction error
    r_test(s,1) = corr(yhat,y); % R
    mse_test(s,1) = sum((yhat-y).^2)/size(y,1); % MSE
    yhat_test(:,s) = yhat;
end

% Calculate the mean prediction error in the test set
% Some outlier subjects may have a big influence on the averge MSE, and we
% exclude very unlikely subjects from the calculation (MSE abs(zscore)>3.89 =>p<0.0001)
idx_test_reliable = abs(zscore(mse_test))<3.89;
mse_test_reliable = mse_test(idx_test_reliable);
mse_test_mean = mean(mse_test_reliable);

% Calculate mean r (exlude the same subjects than from the mse calculation)
r_test_reliable = r_test(idx_test_reliable);
r_test_mean = mean(r_test_reliable);

% Collect results
results = struct;
results.Description = desc;
results.Predictors = Xnames;
results.TrainBetas = betas_train;
results.TrainMSE = mse_train_mean;
results.TrainMSE_sub = mse_train;
results.TrainMSE_sub_reliable = mse_train_reliable;
results.TrainR = r_train_mean;
results.TrainR_sub = r_train;
results.TrainR_sub_reliable = r_train_reliable;
results.TrainYhat = yhat_train;
results.TrainSubjects = subjectsTrain;
results.TestBetas = betas_test;
results.TestMSE = mse_test_mean;
results.TestMSE_sub = mse_test;
results.TestMSE_sub_reliable = mse_test_reliable;
results.TestR = r_test_mean;
results.TestR_sub = r_test;
results.TestR_sub_reliable = r_test_reliable;
results.TestYhat = yhat_test;
results.TestSubjects = subjectsTest;

end

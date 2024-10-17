%% Regression analysis: Summarize regression results over time window analyses
%
% Severi Santavirta

interest = 'blinkRate'; % 'blinkRate', 'fixationRate', 'pupil' or 'isc'
tw = [200,500,1000,2000,4000];
twNames = {'tw200','tw500','tw1000','tw2000','tw4000'};
shift = 0; % O for 'blinkRate', 'fixationRate' and 'isc', Can be either 1000 or 500 for 'pupil'
input = 'path';
output = 'path/regression/combined_results';

% Read results from all tw analyses

for w = 1:size(tw,2)
    load(sprintf('%s/regression/%s_ols_tw%d_shift%d.mat',input,interest,tw(w),shift));
    predictors = resultsFinal.InitialFeatures';
    predictorsConsistent = resultsFinal.ConsistentFeatures;
    predictorsSignificant = resultsFinal.SignificantFeatures;
    betaConsistent = resultsFinal.SimpleRegressionResults.beta;
    rConsistent = resultsFinal.SimpleRegressionResults.R;
    betaSignificant = table2array(resultsFinal.StepwiseBetas(:,2:end));
    pvalConsistent = resultsFinal.ConsistentFeaturePvalues;
    
    % The goal is to plot all results in one table. For this we first
    % extract all beta & R values from simple regressions. For features that
    % showed no consistent results over the datasets we do not have the
    % value. This is assigned to zero. We also collect the consistency
    % information into another matrix. P-values are collected, for
    % inconsistent features we have not calculated p-values, but we assign
    % p=1 to them.
    if(w==1)
        betaMatrix = zeros(size(predictors,1),size(tw,2));
        rMatrix = zeros(size(predictors,1),size(tw,2));
        consistencyMatrix = zeros(size(predictors,1),size(tw,2));
        consistencyCheckMatrix = zeros(size(predictors,1),size(tw,2));
        pvalMatrix = ones(size(predictors,1),size(tw,2));
        
    end
    
    for pred = 1:size(predictors,1)
        
        % Find the indices of the feature in the consistent and significant
        % predictors
        idxConsistent = find(strcmp(predictorsConsistent,predictors{pred}));
        idxSignificant = find(strcmp(predictorsSignificant,predictors{pred}));
        
        % Add to beta, consistency a pval matrices
        if(isempty(idxConsistent))
            betaMatrix(pred,w) = 0;
            rMatrix(pred,w) = 0; 
        else
            betaMatrix(pred,w) = betaConsistent(idxConsistent);
            rMatrix(pred,w) = rConsistent(idxConsistent);
            consistencyMatrix(pred,w) = 1;
            pvalMatrix(pred,w) = pvalConsistent(idxConsistent);
        end
        
        % Check for inconsistent beta signs between all conducted analyses
        if(~isempty(idxSignificant))

            % Check that the beta sign is consistenct in simple regression and in stepwise regressions
            betasAllPred = [betaConsistent(idxConsistent);betaSignificant(:,idxSignificant)];
            betasAllPred(isnan(betasAllPred)) = []; % The predictor was added to the multiple regression in the later steps
            if(any(isinf(betasAllPred))) % The sign flips between the cross validations in any step of the stepwise multiple regression
                consistencyCheckMatrix(pred,w) = 1;
            end
            if(any(betasAllPred>0) && any(betasAllPred<0)) % The sign flips between any performed analysis
                consistencyCheckMatrix(pred,w) = 1;
            end        
        end
    end
end

% Stop for checking inconsistencies if any
if(any(consistencyCheckMatrix(:)))
    error('Beta flips found. Investigate.');
end

% Save tables for plotting in R
betaMatrix = array2table(betaMatrix);
betaMatrix.Properties.RowNames = predictors;
betaMatrix.Properties.VariableNames = twNames;
writetable(betaMatrix,sprintf('%s/%s_shift%d_betas.csv',output,interest,shift),'WriteRowNames',true);

rMatrix = array2table(rMatrix);
rMatrix.Properties.RowNames = predictors;
rMatrix.Properties.VariableNames = twNames;
writetable(rMatrix,sprintf('%s/%s_shift%d_rs.csv',output,interest,shift),'WriteRowNames',true);

consistencyMatrix = array2table(consistencyMatrix);
consistencyMatrix.Properties.RowNames = predictors;
consistencyMatrix.Properties.VariableNames = twNames;
writetable(consistencyMatrix,sprintf('%s/%s_shift%d_consistencies.csv',output,interest,shift),'WriteRowNames',true);

pvalMatrix = array2table(pvalMatrix);
pvalMatrix.Properties.RowNames = predictors;
pvalMatrix.Properties.VariableNames = twNames;
writetable(pvalMatrix,sprintf('%s/%s_shift%d_pvals.csv',output,interest,shift),'WriteRowNames',true);



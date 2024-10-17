%% Eyedata quality control. 
% Read summmary reports and individual subject eyedata and find outliers in the data quality
%
% Severi Santavirta 10.10.2023

sets = {'localizer','kasky','conjuring'};
path = 'path/eyedata/subdata';

%% Read summary reports and investigate total fixation, saccade, blink and inside video area times.

% Read summary reports
for I = 1:size(sets,2)
    if(I==1)
        sum_tbl = readtable(sprintf('%s/%s/summary.csv',path,sets{I}));
    else
        sum_tbl = vertcat(sum_tbl,readtable(sprintf('%s/%s/summary.csv',path,sets{I})));
    end
end

% We estimate the outliers by first fitting beta distributions to the total fixation, saccade, blink and inside video area times data and 
% then identifying the subjects whose data fall below the cutoffs.
prob = 0.02; % Cutoff probability 1%

% Plot histograms and fitted beta distibutions
subplot(2,2,1)
data = sum_tbl.total_time_fixations;
pd_fixations = fitdist(data, 'Beta');
cutoff_fixations = icdf(pd_fixations, prob); % Calculate CDF values for the fitted distribution
x = linspace(min(data), max(data), 100); % Generate x-values
pdf_values = pdf(pd_fixations, x); % Calculate PDF values for the fitted distribution
hist(data,100);
hold on;
plot(x, pdf_values, 'r', 'LineWidth', 2); % Plot fitted distribution
xline(cutoff_fixations);
title('Total fixation time');
xlabel('Proprotional time');
ylabel('Count');
ax = gca;
ax.FontSize = 14;
subplot(2,2,2)
data = sum_tbl.total_time_blinks;
pd_blinks = fitdist(data, 'Beta');
cutoff_blinks = icdf(pd_blinks, 1-prob); % Calculate CDF values for the fitted distribution
x = linspace(min(data), max(data), 100); % Generate x-values
pdf_values = pdf(pd_blinks, x); % Calculate PDF values for the fitted distribution
hist(data,100);
hold on;
plot(x, pdf_values, 'r', 'LineWidth', 2); % Plot fitted distribution
xline(cutoff_blinks);
title('Total blink time');
xlabel('Proprotional time');
ylabel('Count');
ax = gca;
ax.FontSize = 14;
subplot(2,2,3)
data = sum_tbl.total_time_in_video_area;
pd_videoarea = fitdist(data, 'Beta');
cutoff_videoarea = icdf(pd_videoarea, prob); % Calculate CDF values for the fitted distribution
x = linspace(min(data), max(data), 100); % Generate x-values
pdf_values = pdf(pd_videoarea, x); % Calculate PDF values for the fitted distribution
hist(data,100);
hold on;
plot(x, pdf_values, 'r', 'LineWidth', 2); % Plot fitted distribution
xline(cutoff_videoarea);
title('Gaze position in video area');
xlabel('Proprotional time');
ylabel('Count');
ax = gca;
ax.FontSize = 14;

% Find subjects that are outliers based on the cutoffs
bad_subjects_fixations = sum_tbl.subjects(sum_tbl.total_time_fixations<cutoff_fixations);
bad_subjects_saccades = sum_tbl.subjects(sum_tbl.total_time_saccades>cutoff_saccades);
bad_subjects_blinks = sum_tbl.subjects(sum_tbl.total_time_blinks>cutoff_blinks);
bad_subjects_videoarea = sum_tbl.subjects(sum_tbl.total_time_in_video_area<cutoff_videoarea);

bad_subjects = unique(vertcat(bad_subjects_fixations,bad_subjects_saccades,bad_subjects_blinks,bad_subjects_videoarea));

% Reasons for excludions for the bad subjects
% C08 - lwo fixations, high saccades
% C27 - low video area
% K05 - low fixations, high saccedes, high blinks
% K15 - low fixations, high saccedes, high blinks
% K19 - low fixations, high saccedes, high blinks
% K20 - low video area
% K24 - low fixations, high saccedes, high blinks
% L096 - lwo fixations, high saccades



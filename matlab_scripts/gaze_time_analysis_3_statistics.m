%% Gaze Time Analysis: Statistics for the total gaze time analysis based on previously calculated subjectwise gaze times and permuted null distributions
%
% Severi Santavirta 25.10.2023

dset = {'localizer','kasky','conjuring'}; % Combine results from all three datasets (Expr. 1-3).
excluded = {'C08';'C27';'K05';'K15';'K19';'K20';'K24';'L096'}; % Excluded based on QC

%% Read real total gaze postition data for each subject
time_data = [];
dset_idx = [];
good_subjects = [];
dset_idx_good_subjects = [];
for d = 1:size(dset,2)
        
    % Read total time data
    tbl = readtable(sprintf('path/gaze_object_detection/total_gaze_time/time_class_%s.csv',dset{d}));
    data = table2array(tbl(:,2:end));
    class = tbl.class;
    
    % Get high quality subjects data
    subjects = tbl.Properties.VariableNames(2:end);
    good_subjects = horzcat(good_subjects,~ismember(subjects,excluded));
    data = data(:,~ismember(subjects,excluded));
    if(d==1)
        time_data = data;
    else
        time_data = horzcat(time_data,data);
    end

    dset_idx = horzcat(dset_idx,repmat(d,1,size(data,2)));
    dset_idx_good_subjects = horzcat(dset_idx_good_subjects,repmat(d,1,size(~ismember(subjects,excluded),2)));
end

%% Population means, Experiment 1

time_data_dset = time_data(:,dset_idx==1);
mu = zeros(size(time_data_dset,1),1);
for c = 1:size(time_data_dset,1)
    pd = fitdist(time_data_dset(c,:)','normal');
    mu(c) = pd.mu;
end
results_localizer = array2table(mu);

%% Population means, Experiment 2 

time_data_dset = time_data(:,dset_idx==2);
mu = zeros(size(time_data_dset,1),1);
for c = 1:size(time_data_dset,1)
    pd = fitdist(time_data_dset(c,:)','normal');
    mu(c) = pd.mu;
end
results_kasky = array2table(mu);

%% Population means, Experiment 3

time_data_dset = time_data(:,dset_idx==3);
mu = zeros(size(time_data_dset,1),1);
for c = 1:size(time_data_dset,1)
    pd = fitdist(time_data_dset(c,:)','normal');
    mu(c) = pd.mu;
end
results_conjuring = array2table(mu);

%% Calculate what the total times are in the shuffled data and get a p-value for the null hypothesis that people do not have gaze priority towards semantic classes

% Load permuted random total times and calculate the total times for each
% permutation, Experiment 1
f = find_files('path/gaze_object_detection/nulldata/localizer','*.mat');
total_times_random_localizer = zeros(size(class,1),size(f,1));
for r = 1:size(f,1)
    data = load(f{r});
    data = data.time_class;
    data = data(:,logical(good_subjects(dset_idx_good_subjects==1)));
    total_times_random_localizer(:,r) = mean(data,2);
end

% p-values for the real total times
% Do a two-tailed test since for all categories we do not know whether
% the real should be higher or not
p = ones(size(class,1),1);
mu_chance = ones(size(class,1),1);
for c = 1:size(class,1)
    
    nulldist = total_times_random_localizer(c,:);
    mu_chance(c) = mean(nulldist);
    
    % Zero center nulldist
    nulldist_norm = nulldist-mu_chance(c);
    
    % Zero center real total time and shift to positive side
    real = abs(results_localizer.mu(c)-mu_chance(c));
    
    % Count more extreme total times from the null distribution
    count = sum(nulldist_norm>real)+sum(nulldist_norm<(-real));
    
    % p-value
    p(c) = count/size(total_times_random_localizer,2);
end
a = array2table(results_localizer.mu-mu);
results_localizer = horzcat(results_localizer,array2table(mu_chance),array2table(results_localizer.mu-mu_chance),array2table(p));
results_localizer.Properties.VariableNames = {'Average total time','Average chance total time','Difference','p-value'};
results_localizer.Properties.RowNames = class;

% Load permuted random total times and calculate the total times for each
% permutation, Experiment 2
f = find_files('path/gaze_object_detection/nulldata/kasky','*.mat');
total_times_random_kasky = zeros(size(class,1),size(f,1));
for r = 1:size(f,1)
    data = load(f{r});
    data = data.time_class;
    data = data(:,logical(good_subjects(dset_idx_good_subjects==2)));
    total_times_random_kasky(:,r) = mean(data,2);
end

% p-values for the real total times
% Do a two-tailed test since for all categories we do not know whether
% the real should be higher or not
p = ones(size(class,1),1);
mu_chance = ones(size(class,1),1);
for c = 1:size(class,1)
    
    nulldist = total_times_random_kasky(c,:);
    mu_chance(c) = median(nulldist);
    
    % Zero center nulldist
    nulldist_norm = nulldist-mu_chance(c);
    
    % Zero center real total time and shift to positive side
    real = abs(results_kasky.mu(c)-mu_chance(c));
    
    % Count more extreme total times from the null distribution
    count = sum(nulldist_norm>real)+sum(nulldist_norm<(-real));
    
    % p-value
    p(c) = count/size(total_times_random_localizer,2);
end
results_kasky = horzcat(results_kasky,array2table(mu_chance),array2table(results_kasky.mu-mu_chance),array2table(p));
results_kasky.Properties.VariableNames = {'Average total time','Average chance total time','Difference','p-value'};
results_kasky.Properties.RowNames = class;

% Load permuted random total times and calculate the total times for each
% permutation, Experiment 3
f = find_files('path/gaze_object_detection/nulldata/conjuring','*.mat');
total_times_random_conjuring = zeros(size(class,1),size(f,1));
for r = 1:size(f,1)
    data = load(f{r});
    data = data.time_class;
    data = data(:,logical(good_subjects(dset_idx_good_subjects==3)));
    total_times_random_conjuring(:,r) = mean(data,2);
end

% p-values for the real total times
% Do a two-tailed test since for all categories we do not know whether
% the real should be higher or not
p = ones(size(class,1),1);
mu_chance = ones(size(class,1),1);
for c = 1:size(class,1)
    
    nulldist = total_times_random_conjuring(c,:);
    mu_chance(c) = median(nulldist);
    
    % Zero center nulldist
    nulldist_norm = nulldist-mu_chance(c);
    
    % Zero center real total time and shift to positive side
    real = abs(results_conjuring.mu(c)-mu_chance(c));
    
    % Count more extreme total times from the null distribution
    count = sum(nulldist_norm>real)+sum(nulldist_norm<(-real));
    
    % p-value
    p(c) = count/size(total_times_random_conjuring,2);
end
results_conjuring = horzcat(results_conjuring,array2table(mu_chance),array2table(results_conjuring.mu-mu_chance),array2table(p));
results_conjuring.Properties.VariableNames = {'Average total time','Average chance total time','Difference','p-value'};
results_conjuring.Properties.RowNames = class;

% Save the results
permutations = [size(total_times_random_localizer,2),size(total_times_random_kasky,2),size(total_times_random_conjuring,2)];
save('path/gaze_object_detection/analysis/results.mat','results_localizer','results_kasky','results_conjuring','permutations');
writetable(results_localizer,'path/gaze_object_detection/analysis/results_localizer.csv','WriteRowNames',true);
writetable(results_kasky,'path/gaze_object_detection/analysis/results_kasky.csv','WriteRowNames',true);
writetable(results_conjuring,'path/gaze_object_detection/analysis/results_conjuring.csv','WriteRowNames',true);

total_times_localizer = (horzcat(array2table(class),array2table(total_times_random_localizer)));
total_times_kasky = (horzcat(array2table(class),array2table(total_times_random_kasky)));
total_times_conjuring = (horzcat(array2table(class),array2table(total_times_random_conjuring)));
writetable(total_times_localizer,'path/gaze_object_detection/analysis/nulldist_total_times_localizer.csv');
writetable(total_times_kasky,'path/gaze_object_detection/analysis/nulldist_total_times_kasky.csv');
writetable(total_times_conjuring,'path/gaze_object_detection/analysis/nulldist_total_times_conjuring.csv');


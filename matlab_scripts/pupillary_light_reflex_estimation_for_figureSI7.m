%% Estimate pupillary light reflex in a luminance experiment for Experiment 1 subjects.
 % First two trials are for training and trials 3-22 contain 4 sec trials with changing luminance 
 % (1 = black, 2 = 30% white, 3 = 60% white and 4 = 90% white).
 % A brief black screen is shown between each luminance screens for normalizing the pupillary reflex.

% Severi Santavirta 12.4.2024

clear;

%% INPUT

report_path = 'path/megatrack_A1_FixationPupil_40ms_bin_report.txt'; % Where is the fixation report?
func_path = 'path/funcs'; % Where are the needed functions?
output = 'path/pupillary_light_reflex/';

%% Script
data = readtable(report_path);

% Separate data matrix for every subject
subj_list = table2array(data(:,1));
subjects = unique(subj_list,'stable');

%Check data, at least one subject has NaN-data
bad_subj = {};
k=0;
for I = 1:size(subjects,1)
    subdata = struct;     
    subj_idx = find(ismember(subj_list,subjects{I}));
    submat = data(subj_idx,:);
    
    if(isnan(table2array(submat(1,101))))
        k=k+1;
        bad_subj{k,1} = subjects{I};
    end
end

pupil = zeros(size(subjects,1)-size(bad_subj,1),4,5,100); % subjects x luminance levels x repetitions per luminance level x timepoints;
n = 0;
for I = 1:size(subjects,1)
   subdata = struct;     
   subj_idx = find(ismember(subj_list,subjects{I}));
   submat = data(subj_idx,:);

   % First two trials are for training and then trials 3-22 contain 4
   % sec trials with changing luminance (1 = black, 2 = 30% white, 3 = 60% white and 4 = 90% white). A brief black screen is
   % between each luminance screens for normalizing the pupillary
   % reflex
       
    if(isnan(table2array(submat(1,101))))
        warning('Subject excluded: %s\n',subjects{I});
    else
        n=n+1;
        k = [0,0,0,0];
        trials = submat.Var93;
        p = submat.Var88;
        for J = 3:22 % real trials
           trial_idx = (trials==J);
           trial_rows = submat(trial_idx,:);
           tw_intensity = unique(table2array(trial_rows(1:100,101)));
           k(tw_intensity) = k(tw_intensity)+1;
           trial_pupil = p(trial_idx); trial_pupil = trial_pupil(1:100,1);

           % Some timewindows have NaNs, interpolate
           % pupil size linearly over NaNs
           if(sum(isnan(trial_pupil))>0)
               nanidx = isnan(trial_pupil);
               t = 1:numel(nanidx);
               trial_pupil(nanidx) = interp1(t(~nanidx), trial_pupil(~nanidx), t(nanidx),'spline','extrap');
           end

           pupil(n,tw_intensity,k(tw_intensity),:) = trial_pupil./mean(trial_pupil(6)); % Standardize to the last timepoint before group level light response begins
        end
    end
end

% Calculate population level pupillary response for each luminosity levels.
l1 = squeeze(mean(squeeze(pupil(:,1,:,:)),2))';
l2 = squeeze(mean(squeeze(pupil(:,2,:,:)),2))';
l3 = squeeze(mean(squeeze(pupil(:,3,:,:)),2))';
l4 = squeeze(mean(squeeze(pupil(:,4,:,:)),2))';

%Plot
x = 1:100;
plot(x,mean(l1'),x,mean(l2'),x,mean(l3'),x,mean(l4'));

% Save results for plotting with R
writetable(array2table(l1),sprintf('%s/luminance_0.csv',output));
writetable(array2table(l2),sprintf('%s/luminance_30.csv',output));
writetable(array2table(l3),sprintf('%s/luminance_60.csv',output));
writetable(array2table(l4),sprintf('%s/luminance_90.csv',output));


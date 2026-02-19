close all; clearvars; clc

% set directories
% rootdir = '\\cimec-storage5.unitn.it\MORWUR\Projects\INGMAR\Project1917'; % are them still there?
rootdir = 'smb://cimec-storage6.cimec.unitn.it/ingdev/Projects/Project1917/data';
addpath(genpath(rootdir));

%% behavioural analysis
parms = [];
parms.subjects = [3:12 14];

Project1917_behavioural_analysis(parms);

% preprocessing
%% 1. rereference, filter, and cut out catch trials and breaks
parms = [];
% parms.cluster = 1;
parms.subjects = 3;
parms.runs = 1:6;

parms.script2run = 'Project1917_preproc1_reref_filter_trim';
Project1917_cluster_shell_parSUBRUN(parms);

%% 2. bad channel and segment detection
parms = [];
parms.subjects = 15;
parms.runs = 1:6;

Project1917_preproc2_cleaning(parms) % got here

%% 3. ICA
parms = [];
parms.cluster = 1;
parms.subjects = 15;
parms.runs = 1:6;

parms.script2run = 'Project1917_preproc3_ICA';
Project1917_cluster_shell_parSUBRUN(parms);

%% 4. ICA check
parms = [];
parms.subjects = 15;
parms.runs = 1:6;

Project1917_preproc4_ICAcheck(parms)

%% 5. ROI selection, dealing with bad channels and bad segments, optionally multivariate noise normalization (MNN), combining runs
parms = [];
parms.cluster = 0;
parms.subjects = 3;
parms.ROIs = 1:2;% 1 = all sensors, 2 = occipito-parietal sensors
parms.fsNew = 50;% sampling frequency to downsample to neural data to
parms.neuralSmoothing = 23;% smoothing in samples, because that's what ft_preproc_smooth uses, should be odd number of samples
parms.MNN = 0; % Multivariate Noise Normalization: 0 = no MNN, 1 = MNN using trials as observations, 2 = MNN using time as observations
parms.rej_bad_muscle = 0;% 0 to keep bad muscle segments in, 1 to reject them
parms.rej_bad_lowfreq = 1;% 0 to keep low-freq noise segments in, 1 to reject them
parms.bad_seg_interp = 1;% 0 to replace bad segments with NaNs and ignore in final analysis, 1 to interpolate
parms.rej_bad_comp = 2;% 1 to remove all bad components, 2 to keep eye-movement components in

parms.script2run = 'Project1917_preproc5_ROIselect_interp_MNN';
Project1917_cluster_shell_parSUBROI(parms);

parms.script2run = 'Project1917_preproc5_4TizianoSISSA';


%% 6. Retrieve gaze position from eyetracker output, align with neural data, interpolate blinks
parms = [];
parms.cluster = 1;
parms.subjects = 3:34;
parms.runs = 1:6;
parms.preblinkpadding = 0.03;% padding around blink in seconds
parms.postblinkpadding = 0.07;
parms.preblinkgazetime = 0.02;% time window to average gaze position over before blink, which will be used as gaze position during blink for model, in seconds
parms.fsNew = 50;%23.976;% to fit the original movie frame rate
parms.starttrigger = 2;% which trigger is start for analysis, we skip first 5 sec of movie to ignore movie onset effects, and because of some strange delay there
parms.endtrigger = 255;
parms.ITI = 5;% inter-trigger-interval of 5 sec throughout the movie
parms.lpfreqeye = 30;% smooth eyetracker signal to remove jitter due to noise
parms.avg_delay_eyetracker = 1/120;% in case no eye movement ICA component was found, we use 8.333 msec, which was roughly the average across subjects / runs.
% for real analysis probably better to stick with avg delay because makes
% sense that constant delay is simply 1 screen refresh rate of 8.3 msec,
% and any variation amongst subjects comes from measurement noise in the
% MEG component or eye-tracker signal, correlations are usually below 0.8
% so delay estimation is not super precise. Also, _standardalign script is
% more up to date, so perhaps just use that one.

parms.script2run = 'Project1917_preproc6_gazeposition';% find ICA eye-movement component and align eye-tracker data, slower
parms.script2run = 'Project1917_preproc6_gazeposition_standardalign';% take average delay to align eye-tracker data, faster
Project1917_cluster_shell_parSUBRUN(parms);

%% 7. Create models
parms = [];
parms.runs = 1:6;
parms.sigma = 2.5;% sigma = spatial smoothing factor, from Kriegeskorte 2008 (i.e., 5); after visual inspection comparing original with smoothed, value of 5 seems high, so lowered to 2.5
parms.fsNew = 50;% match to neural sampling rate for dRSA analysis
parms.tstart = 5;% always start at t = 5 sec because we ignore the movie onset effects, and because of sometimes strange delay between first two triggers

% gaze-invariant pixelwise similarity model
parms.sdsf = 50;% spatial downsampling factor, up to 40 is precize enough for our needs (i.e., correlation between original and downsampled at least 0.994 and on average 0.99998)
Project1917_preproc7_models_gazeinvar_pixelwise(parms);

% gaze-invariant optical flow vector magnitude and direction models
parms.OFtempres = 24;% whether to run optical flow models on downsampled 12 Hz movie data, or on original 24 Hz movie data, or on both
parms.sdsf = 50;% spatial downsampling factor, up to 40 is precize enough for our needs (i.e., correlation between original and downsampled at least 0.994 and on average 0.99998)
Project1917_preproc7_models_gazeinvar_OF(parms);

% gaze-dependent pixelwise similarity model
parms.subjects = [16:23 25 28];% subjects for gaze-dependent models
parms.gazeradius = 250;% circle size around gaze location in pixels
parms.sdsf = 20;% gaze dependent models have less features anyway, so less spatial downsampling is necessary
% Project1917_preproc7_models_gazedep_pixelwise(parms);
Project1917_preproc7_models_gazedep_pixelwise_parallel(parms);

% gaze-dependent optical flow vector magnitude and direction models
parms.subjects = [21:23 25 28];% subjects for gaze-dependent models
parms.gazeradius = 250;% circle size around gaze location in pixels
parms.OFtempres = 24;% whether to run optical flow models on downsampled 12 Hz movie data, or on original 24 Hz movie data, or on both
parms.sdsf = 20;% gaze dependent models have less features anyway, so less spatial downsampling is necessary
% Project1917_preproc7_models_gazedep_OF(parms);
Project1917_preproc7_models_gazedep_OF_parallel(parms);

close all; clearvars; clc
addpath('/Users/tizianocausin/Desktop/programs/fieldtrip-20240110')
% set directories
% rootdir = '\\cimec-storage5.unitn.it\MORWUR\Projects\INGMAR\Project1917';
rootdir='/Users/tizianocausin/Desktop/dataRepository/RepDondersInternship/project1917'; %!!CHANGED PATH
addpath(genpath(rootdir));
codeDir='/Users/tizianocausin/Desktop/backUp20240609/summer2024/dondersInternship/code'; %!!CHANGED TO ADD THE PATH TO THE CODE WHICH IS SEPARATE FROM THE DATA REP
addpath(genpath(codeDir))
%% set dRSA parameters
% general parameters
parms = [];
parms.fsNew = 50;% here match the chosen neural sampling rate
% parms.subjects = 3:15;
parms.subjects = 3; %!!CHANGED BC I ONLY HAVE SUB03
parms.repetitions = 1:2;
parms.ROInames = {'allsens', 'occpar'};
parms.ROI = 1:2;% 1 = all MEG sensors, 2 = occipito-parietal
parms.modelnames = {'pixelwise','OFmag','OFdir'};
parms.models2test = 1:3;% 
parms.OFtempres = 24;% whether to use OF models computed on 12 Hz downsampled or 24 Hz original movie data, or on both
parms.gazedep = 0:1;% whether to use  gaze-invariant models (0), or gaze-dependent models (1) or both

% preprocessing parameters
parms.gazeradius = 250;% circle size around gaze location in pixels
parms.rej_bad_muscle = 0;% 0 to keep bad muscle segments in, 1 to reject them
parms.rej_bad_lowfreq = 1;% 0 to keep low-freq noise segments in, 1 to reject them
parms.bad_seg_interp = 1;% 0 to replace bad segments with NaNs and ignore in final analysis, 1 to interpolate
parms.rej_bad_comp = 2;% 1 to remove all bad components, 2 to keep eye-movement components in

% dRSA parameters
parms.MNN = 0;
parms.smoothNeuralRDM = 5;% Smoothing of neural RDM in samples. Has to be odd number, because includes centre point!  
parms.smoothModelRDM = 5;% same for model RDM
parms.similarity = 0;% 0 = correlation, 1 = principal component regression
parms.nPCRcomps = 75;% in case of pcr, maximum amount of PCR components to regress out

% temporal subsampling parameters
parms.nstim = 180; % # of pseudo-stimuli to cut out of movie for each subsampling iteration 
parms.stimlen = 10;% length of pseudo-stimuli in seconds
parms.iterations = 1000;
parms.minISI = 1;% minimum inter-stimulus-interval in seconds; 0 means pseudo-stimuli can touch each other
parms.maxlatency = 5;% max latency to test with dRSA latency plots in sec

% cluster or locally
%parms.cluster = 1; %!!CHANGED BC NO CLUSTER NEEDED 

%% run dRSA
parms.script2run = 'Project1917_dRSA';
Project1917_cluster_shell_parALL(parms);

%% run simulation
% parms.models2sim = 1:3;% models to simulate
parms.ROI = 1;
parms.iterationsSIM = 50;

parms.script2run = 'Project1917_dRSA_simulation';
Project1917_cluster_shell_parALL(parms);

%% plot results
parms.fisherz = 1;%fisher transform for individual subject correlation values before stats and plotting
parms.fs = 8;% fontsize
parms.side = 2;% 1 or 2 sided
parms.pthresh = 0.05;
parms.models2plot = [2 4 6];% pixelwise gaze-invar, pixelwise gaze-dep, OFmag gaze-invar, OFdir gaze-invar
parms.modellabels = {'pixel gaze-invar','pixel gaze-dep','OFmag gaze-invar','OFmag gaze-dep','OFdir gaze-invar','OFdir gaze-dep'};

Project1917_dRSA_PLOT(parms);


% % single subject statistics
% parms.randperms = 1:20;
% 
% % script2run = 'Project1917_dRSA_randperms';
% script2run = 'Project1917_dRSA_randperms_sub0506';
% Project1917_cluster_shell_parSUBMODROIPERM(parms,script2run);
% 
% Project1917_dRSA_singlesubstats(parms)

function cProject1917_dRSA_general(subjects, models,ROIs,igaze, ires, nproc)

% Calls Project_dRSA_core with the right parameters. It's a function so
% that I can call it with the right args from a job script.
% INPUT:
% - subjects::Array -> the idx of subjects

% - models::cell or string -> the models to compute, 
% if it's a ANN, just pass the name, we are adding automatically all the
% layers

% - ROIs::Int or Cell -> the ROIs to compute, if it's all of them, just
% write 1
% - igaze::Int -> 1 if gaze-dep, 0 if gaze indep
% - ires::Int -> 24 if movie resolution , 50 if gaze resolution 


%FIXME add outside the script 
parallel.defaultClusterProfile('local')  % Reset the default
parcluster('local') 
parpool('local',nproc)
addpath("/home/tiziano.causin/adds_on/fieldtrip-20250114")
ft_defaults
% set directories
preproc_dir = '/mnt/storage/tier2/ingdev/projects/TIZIANO/data_preproc';
models_dir = "/mnt/storage/tier2/ingdev/projects/TIZIANO/models";
results_dir = "/mnt/storage/tier2/ingdev/projects/TIZIANO/results";
parms = [];
parms.fsNew = 50;% here match the chosen neural sampling rate
% parms.subjects = 3:15;
parms.subjects =subjects ;
parms.repetitions = 1:2;
disp("dRSA_general")
disp(ROIs)
if strcmp(class(ROIs{1}),'double') % if ROIs == 1 do all
    parms.ROInames = {'allsens', 'occpar','occ', 'par', 'tem', 'fro'}
else % otherwise specify which ones
    parms.ROInames = ROIs
end % if ROIs == 1
    if strcmp(models{1}, 'alexnet')
        parms.modelnames = {"alexnet_conv_layer1", "alexnet_conv_layer4", "alexnet_conv_layer7", "alexnet_conv_layer9", "alexnet_conv_layer11", "alexnet_fc_layer2", "alexnet_fc_layer5"};
    elseif strcmp(models{1}, 'resnet18')
        parms.modelnames = {"resnet18_layer1", "resnet18_layer2", "resnet18_layer3", "resnet18_layer4", "resnet18_fc"};
    elseif strcmp(models{1}, 'ViT')
        parms.modelnames = {"ViT_layer0", "ViT_layer1", "ViT_layer2", "ViT_layer3", "ViT_layer4", "ViT_layer5", "ViT_layer6", "ViT_layer7", "ViT_layer8", "ViT_layer9", "ViT_layer10", "ViT_layer11", "ViT_layer12"};
else
    parms.modelnames = models
end % if strcmp(models{1}, 'alexnet')

disp("models")
disp(models)
disp(parms.modelnames)
disp("ROIs")
disp(ROIs)
disp(parms.ROInames)
parms.OFtempres = 24;% whether to use OF models computed on 12 Hz downsampled or 24 Hz original movie data, or on both
parms.gazedep = 0;% whether to use  gaze-invariant models (0), or gaze-dependent models (1) or both

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
parms.iterations = 100; %00;
parms.minISI = 1;% minimum inter-stimulus-interval in seconds; 0 means pseudo-stimuli can touch each other
parms.maxlatency = 5;% max latency to test with dRSA latency plots in sec
% cluster or locally
%parms.cluster = 1; %!!CHANGED BC NO CLUSTER NEEDED
for isub=parms.subjects
    for irep=parms.repetitions
        for iroi=1:length(parms.ROInames)
            roi_name = parms.ROInames{iroi} 
            parfor imod=1:length(parms.modelnames)
                mod_name = parms.modelnames{imod} % just to show at which ROI we had arrived
                Project1917_dRSA_core(preproc_dir, models_dir, results_dir,  parms,isub,irep,imod,iroi,ires, igaze, roi_name, mod_name)
disp(isub)
disp(irep)
disp(imod)
disp(mod_name)
disp(igaze)
disp(ires)
            end % parfor iroi = 1:6
        end %for imod=1:length(parms.modelnames)
    end %for irep=parms.repetitions
end %for isub=parms.subjects





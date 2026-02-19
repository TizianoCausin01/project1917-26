% function Project1917_preproc3_ICA(parms,isub,irun)
% Project 1917
% preprocessing - ICA
%numCores = min(8, feature('numcores'));
numCores = 6; % to run in parallel the 6 runs
parpool('local',numCores)
addpath("/home/tiziano.causin/adds_on/fieldtrip-20250114")
ft_defaults
rootdir = "/mnt/storage/tier2/ingdev/projects/TIZIANO/data_preproc"
subjects = 5:10;
runs = 1:6;
for isub = subjects
    indir = sprintf('%s/sub-%03d/preprocessing',rootdir,isub);
    parfor irun = 1:length(runs)
       disp(['Processing Subject ' num2str(isub) ', Run ' num2str(irun)]);
       Project1917_preproc3_ICA(indir, isub, irun)
    end % for irun = runs 
end % for isub = subjects


function Project1917_preproc3_ICA(indir, isub, irun)
fn2load = sprintf('%s%sdata_reref_filt_trim_sub%03d_run%02d',indir,filesep,isub,irun);
load(fn2load,'data');

% filter
cfg = [];
cfg.padding = 900;% our video is almost 900 sec long
cfg.padtype = 'mirror';% we need to use mirror because there is no extra data
cfg.hpfilter = 'yes';
cfg.hpfreq = 1;
cfg.hpfiltord = 2;
data = ft_preprocessing(cfg, data);

% load bad channels and bad segments and remove before ICA
fn2load = sprintf('%s%sbadchan_sub%03d_run%02d',indir,filesep,isub,irun);
load(fn2load, 'megchan_keep');

cfg = [];
cfg.channel = megchan_keep(:)';
data = ft_selectdata(cfg, data);

fn2load = sprintf('%s%sbadseg_sub%03d_run%02d',indir,filesep,isub,irun);
load(fn2load, 'BAD_lowfreq','BAD_muscle');

% combine two sources of bad segments
badsegs = [BAD_lowfreq ; BAD_muscle];

for iseg = 1:size(badsegs,1)
    badsegs(iseg,:) = dsearchn(data.time{1}',badsegs(iseg,:)')';
end

% create logical vector of bad segments
samples2keep = true(size(data.time{1}));
for iseg = 1:size(badsegs,1)
    samples2keep(badsegs(iseg,1):badsegs(iseg,2)) = false;
end

data.trial{1}(:,~samples2keep) = NaN;

cfg = [];
cfg.method = 'runica';
cfg.demean = 'no';
cfg.channel = {'MEG'}; % only do ICA on MEG channels, not the refchans

comp = ft_componentanalysis(cfg, data);

unmixing = comp.unmixing;
topolabel = comp.topolabel;

fn2save = sprintf('%s%sica_weights_sub%03d_run%02d',indir,filesep,isub,irun);
save(fn2save,'unmixing', 'topolabel');
end % EOF

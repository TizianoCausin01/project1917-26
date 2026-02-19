function Project1917_preproc3_ICA(parms,isub,irun)
% Project 1917
% preprocessing - ICA

if isfield(parms,'cluster')
    % set paths
    rootdir = '//mnt/storage/tier2/morwur/Projects/INGMAR/Project1917';
    % start up Fieldtrip
    addpath('//mnt/storage/tier2/morwur/Projects/INGMAR/toolboxes/fieldtrip-20231220');
else
    % set paths
    rootdir = '/Volumes';
    % start up Fieldtrip
    addpath('/Users/tizianocausin/Desktop/programs/fieldtrip-20240110')
end
indir = sprintf('%s%sdata%ssub-%03d%spreprocessing',rootdir,filesep,filesep,isub,filesep);

% set Fieldtrip defaults
ft_defaults

fn2load = sprintf('%s%sdata_reref_filt_trim_sub%03d_run%02d',indir,filesep,isub,irun);
load(fn2load,'data');
beep
%% DOWNSAMPLING PART
cfg = [];
cfg.resamplefs = 600
cfg.method = 'resample'
data = ft_resampledata(cfg, data)
beep
%%
% filter
cfg = [];
cfg.padding = 900;% our video is almost 900 sec long
cfg.padtype = 'mirror';% we need to use mirror because there is no extra data
cfg.hpfilter = 'yes';
cfg.hpfreq = 1;
cfg.hpfiltord = 2;
data = ft_preprocessing(cfg, data);
%%
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
%%
% create logical vector of bad segments
samples2keep = true(size(data.time{1}));
for iseg = 1:size(badsegs,1)
    samples2keep(badsegs(iseg,1):badsegs(iseg,2)) = false;
end

data.trial{1}(:,~samples2keep) = NaN;
%%
cfg = [];
cfg.method = 'runica';
cfg.demean = 'no';
cfg.channel = {'MEG'}; % only do ICA on MEG channels, not the refchans

comp = ft_componentanalysis(cfg, data);
beep
unmixing = comp.unmixing;
topolabel = comp.topolabel;
%%
fn2save = sprintf('%s%sica_weights_sub%03d_run%02d',indir,filesep,isub,irun);
save(fn2save,'unmixing', 'topolabel');

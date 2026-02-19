% to recheck: 
% sub 3 run 2 (messed up)
% sub 5 run 4
% sub 6 run 1,2 (compon 1,2 both + compon 17 run 2)
% sub 6 run 5 (go back to bad segments)
% sub 7 run 1
% sub 7 run 5 (messed up)
% sub 8 run 1 (ringing)
% sub 8 run 3 (messed up)
% sub 8 run 4
%%
%sub 6 sensors messed up
%%
% start up Fieldtrip
addpath(['/Users/tizianocausin/Desktop/programs/fieldtrip-20240110'])
ft_defaults
preproc_dir = '/Volumes/TIZIANO/data_preproc/';
subjects = 7;
runs = 1:6;
for isub = subjects
    for irun = runs
        tProject1917_preproc4_ICAcheck(isub, irun, preproc_dir)
    end % for irun = runs
end % for isub % = subjects


function tProject1917_preproc4_ICAcheck(isub, irun, preproc_dir)
% Project 1917
% preprocessing - IC check and rejection

indir = sprintf('%s%ssub-%03d%spreprocessing',preproc_dir,filesep,isub,filesep);

fn2load = sprintf('%s%sdata_reref_filt_trim_sub%03d_run%02d',indir,filesep,isub,irun);
load(fn2load,'data');

% load bad channels and bad segments and remove before ICA checking
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

% load ICA components
fn2load = sprintf('%s%sica_weights_sub%03d_run%02d',indir,filesep,isub,irun);
load(fn2load,'unmixing', 'topolabel');

% now plot components for check
cfg = [];
cfg.method = 'predefined mixing matrix';
cfg.demean = 'no';
cfg.channel = {'MEG'};
cfg.topolabel = topolabel;
cfg.unmixing = unmixing;
beep
comp = ft_componentanalysis(cfg, data);
comp.trial{1}(:,~samples2keep) = NaN;

cfg = [];
cfg.viewmode = 'component';
cfg.layout = 'CTF275_helmet.mat';
cfg.blocksize = 15;
cfg.channel = comp.label(1:20);
cfg.compscale = 'local';
ft_databrowser(cfg, comp)
beep
%%
fprintf('*** SUBJECT %02d : save the identified components!!! ***\n', isub);

% write down and save
badcomps = [];
badcomps_reasons = {};
badcount = 0;
while 1 % press 0 to escape
    newcomp = input('component: ');
    if newcomp == 0
        break
    else
        badcount = badcount + 1;
    end
    badcomps(badcount) = newcomp;
    newreason = input('reason: ','s');
    badcomps_reasons{badcount} = newreason;
end

assert(numel(badcomps) == numel(badcomps_reasons));

fn2save = sprintf('%s%sica_badcomps_sub%03d_run%02d',indir,filesep,isub,irun);
save(fn2save, 'badcomps', 'badcomps_reasons');
clear comp
close all
end %EOF
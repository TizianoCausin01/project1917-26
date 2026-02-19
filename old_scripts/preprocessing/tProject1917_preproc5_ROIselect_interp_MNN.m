function Project1917_preproc5_ROIselect_interp_MNN(parms,isub,iroi)
% Project 1917
% preprocessing script 4 - ROI selection, dealing with bad channels and bad segments, optionally multivariate noise normalization (MNN), combining runs

if isfield(parms,'cluster')
    % set paths
    rootdir = '//mnt/storage/tier2/morwur/Projects/INGMAR/Project1917';
    % start up Fieldtrip
    addpath('//mnt/storage/tier2/morwur/Projects/INGMAR/toolboxes/fieldtrip-20231220');
else
    % set paths
    rootdir = '\\cimec-storage5.unitn.it\MORWUR\Projects\INGMAR\Project1917';
    % start up Fieldtrip
    addpath('\\cimec-storage5.unitn.it\MORWUR\Projects\INGMAR\toolboxes\fieldtrip-20231220')
end

% set Fieldtrip defaults
ft_defaults

% prepare layout
cfg = [];
cfg.layout = 'CTF275.lay';
layout = ft_prepare_layout(cfg);
layout = layout.label;

% prepare neighbourhood structure
cfg = [];
cfg.method = 'template';
cfg.template = 'CTF275_neighb.mat';
neighbours = ft_prepare_neighbours(cfg);

indir = sprintf('%s%sdata%ssub-%03d%spreprocessing',rootdir,filesep,filesep,isub,filesep);
outdir = sprintf('%s%sdata%ssub-%03d%spreprocessed',rootdir,filesep,filesep,isub,filesep);

if ~exist(outdir,'dir')
    mkdir(outdir);
end

% remove 'COMNT' and 'SCALE' channel, and remove channels that are missing
% from all runs for this subject
missing_standard = [125 155];
layout([missing_standard end-1:end]) = [];
neighbours(missing_standard) = [];

if iroi == 1
    ROIname = {'allsens'};
    ROIletter = {'M'};
elseif iroi == 2
    ROIname = {'occpar'};
    ROIletter = {'O','P'};
end

%% load all runs
missingchan = cell(1,6);
for irun = 1:6

    fn2load = sprintf('%s%sdata_reref_filt_trim_sub%03d_run%02d',indir,filesep,isub,irun);
    load(fn2load,'data');

    % remove bad channels
    fn2load = sprintf('%s%sbadchan_sub%03d_run%02d',indir,filesep,isub,irun);
    load(fn2load, 'megchan_keep');

    cfg = [];
    cfg.channel = megchan_keep(:)';
    data = ft_selectdata(cfg, data);

    % remove bad ICA components
    fn2load = sprintf('%s%sica_weights_sub%03d_run%02d',indir,filesep,isub,irun);
    load(fn2load,'unmixing', 'topolabel');

    fn2load = sprintf('%s%sica_badcomps_sub%03d_run%02d',indir,filesep,isub,irun);
    load(fn2load, 'badcomps', 'badcomps_reasons');

    if parms.rej_bad_comp == 2% keep eye movement components in
        eyemovID = strcmp(badcomps_reasons,'eyemov');
        badcomps(eyemovID) = [];
        badcomps_reasons(eyemovID) = [];
    end

    % remove the bad components
    cfg = [];
    cfg.demean = 'no';
    cfg.method = 'predefined unmixing matrix';
    cfg.unmixing = unmixing;
    cfg.topolabel = topolabel;
    data = ft_componentanalysis(cfg, data);

    % reject bad components
    cfg = [];
    cfg.demean = 'no';
    cfg.component = badcomps;
    data = ft_rejectcomponent(cfg, data);

    % smooth with sliding window (Cichy)
    data.trial{1} = ft_preproc_smooth(data.trial{1},parms.neuralSmoothing);

    % downsample
    cfg = [];
    cfg.resamplefs = parms.fsNew;
    cfg.demean = 'no';
    cfg.detrend = 'no';
    data = ft_resampledata(cfg, data);

    % remove bad segments
    fn2load = sprintf('%s%sbadseg_sub%03d_run%02d',indir,filesep,isub,irun);
    load(fn2load, 'BAD_lowfreq','BAD_muscle');

    % combine two sources of bad segments
    badsegs = [BAD_lowfreq*parms.rej_bad_lowfreq ; BAD_muscle*parms.rej_bad_muscle];
    badsegs(badsegs(:,1) == 0,:) = [];

    for iseg = 1:size(badsegs,1)
        badsegs(iseg,:) = dsearchn(data.time{1}',badsegs(iseg,:)')';
    end

    % create logical vector of samples to keep
    samples2keep = true(size(data.time{1}));
    for iseg = 1:size(badsegs,1)
        samples2keep(badsegs(iseg,1):badsegs(iseg,2)) = false;
    end

%     % find bad segments larger than 10 sec so they can be excluded from the dRSA analysis
%     segstart = [];
%     segend = [];
%     for isamp = 1:length(samples2keep)-1
%         if diff(samples2keep(isamp:isamp+1)) == -1
%             segstart = [segstart isamp+1];
%         elseif diff(samples2keep(isamp:isamp+1)) == 1
%             segend = [segend isamp+1];
%         end
%     end
%     largesegID = (segend-segstart)/data.fsample > 10;
%     segstart(~largesegID) = [];
%     segend(~largesegID) = [];

    % either replace bad segments with NaNs and ignore in final analysis, or interpolate
    if ~parms.bad_seg_interp
        data.trial{1}(:,~samples2keep) = NaN;
    elseif parms.bad_seg_interp
        
        tempdata = data.trial{1};
        temptime = data.time{1};
        tempdata(:,~samples2keep) = [];
        temptime(~samples2keep) = [];

        tempdata = interp1(temptime,tempdata',data.time{1},'pchip')';
        data.trial{1} = tempdata;
        clear tempdata temptime
    end

    % make channel selection for this ROI
    chan2sel = contains(layout,ROIletter);

    if any(~contains(layout,data.label))

        % find which channel(s) are missing
        missingchan{irun} = ~contains(layout,data.label);

        % interpolation generally not recommended for MEG data because of different sensor directions relative to a magnetic source, but MNN below
        % doesn't work with NaN, so we interpolate for now. After MNN we can still replace those bad channels with NaNs
        cfg = [];
        cfg.badchannel     = layout(missingchan{irun});
        cfg.method         = 'spline';%'nan';
        cfg.neighbours     = neighbours;
        data = ft_channelrepair(cfg,data);

        % interpolated channel is appended at end of data, move it to
        % correct position
        newid = zeros(size(layout));
        for ichan = 1:length(layout)
            newid(ichan) = find(strcmp(layout{ichan},data.label));
        end

        data.label = data.label(newid);
        data.trial{1} = data.trial{1}(newid,:);

        % select these channels in the missing channel array
        missingchan{irun} = missingchan{irun}(chan2sel);

    end

    % now select channels
    cfg = [];
    cfg.channel = layout(chan2sel);
    data = ft_selectdata(cfg,data);

    if irun == 1
        data_all = data;
    end

    data_all.time{irun} = data.time{1};
    data_all.trial{irun} = data.trial{1};

end% runs

data = data_all;
clear data_all

%% multivariate noise normalisation with shrinkage (Guggenmos et al. 2018 NeuroImage)
n_sensors = length(data.label);% # of sensors for this ROI
n_stim = 2;% # of runs
uniquestim = 1:2;
n_trial = 1;% # of repetitions per part

if parms.MNN == 1

    % using trial dimension as observations for cov1para
    % first compute sigma
    sigma_perpart = nan(n_stim, n_sensors, n_sensors);
    for ipart = 1:n_stim

        data2use = data.trial(ipart:2:end);
        data2use = permute(cat(3,data2use{1},data2use{2},data2use{3}),[3 1 2]);

        % remove timepoints with only NaNs on all 3 repetitions
        data2use(:,:,squeeze(all(isnan(data2use(:,1,:)),1))) = [];% time points

        n_time = size(data2use,3);
        % compute sigma for each time point
        sigma_pertime = nan(n_time, n_sensors, n_sensors);
        for itime = 1:n_time

            temp = data2use(:,:,itime);

            % now remove the trials with NaNs
            temp(any(isnan(temp),2),:) = [];
            sigma_pertime(itime, :, :) = cov1para(temp);

        end
        sigma_perpart(ipart, :, :) = squeeze(mean(sigma_pertime, 1));% average across time
        clear sigma_pertime
    end

elseif parms.MNN == 2

    % using time dimension as observations for cov1para
    % first compute sigma
    sigma_perpart = nan(n_stim, n_sensors, n_sensors);
    for ipart = 1:n_stim

        data2use = data.trial(ipart:2:end);
        data2use = permute(cat(3,data2use{1},data2use{2},data2use{3}),[3 1 2]);

        n_time = size(data2use,3);
        % compute sigma for each time point
        sigma_pertrial = nan(n_trial, n_sensors, n_sensors);
        for itrial = 1:n_trial

            temp = squeeze(data2use(itrial,:,:))';

            % now remove the time points with NaNs
            temp(any(isnan(temp),2),:) = [];
            sigma_pertrial(itrial, :, :) = cov1para(temp);

        end
        sigma_perpart(ipart, :, :) = squeeze(mean(sigma_pertrial, 1));% average across time
        clear sigma_pertrial
    end

end

% apply MNN
if parms.MNN > 0

    sigma = squeeze(mean(sigma_perpart, 1));  % average across runs
    clear sigma_perpart
    sigma_inv = sigma^-0.5;

    for irun = 1:6
        for itime = 1:length(data.time{irun})
            data.trial{irun}(:,itime) = squeeze(data.trial{irun}(:,itime))' * sigma_inv;
        end
    end
end

% after MNN we can call the bad chans NaN again and actually remove
% them in the dRSA analysis, because interpolation not accurate for MEG data.
% Was only done to allow MNN to work across runs
for irun = 1:6

    data.trial{irun}(missingchan{irun},:) = nan;

end

%% decrease file size for storage
% I tested how much we can reduce precision by correlating the uint8 data
% matrix with the original data matrix and the lowest correlation value
% across all time points is is 0.997, but to be safe let's use uint16 precision,
% for which the lowest correlation value was 0.99999995
%
% First we need to rescale the data to the 1-2^16 interval, then it fits in
% the uint16 precision so we can reduce precision (and storage size)
data_final = cell(6,1);
for irun = 1:6

    data_final{irun} = single(data.trial{irun});% uint16(rescale(data.trial{irun},1,2^16));

end

% and then store, with some additional info such as time, labels
time = data.time;
label = data.label;
fn2save = sprintf('%s%ssub%03d_%s_%dHz_MNN%d_badmuscle%d_badlowfreq%d_badsegint%d_badcomp%d',...
    outdir,filesep,isub,ROIname{1},parms.fsNew,parms.MNN,parms.rej_bad_muscle,parms.rej_bad_lowfreq,parms.bad_seg_interp,parms.rej_bad_comp);

save(fn2save,'data_final','time','label','missingchan');


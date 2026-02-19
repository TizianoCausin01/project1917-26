
numCores = 6; % to run in parallel the 6 runs
parpool('local',numCores)
addpath("/home/tiziano.causin/adds_on/fieldtrip-20250114")
rootdir = "/mnt/storage/tier2/ingdev/projects/Project1917"
subjects=5:10
runs=1:6
preproc_dir = "/mnt/storage/tier2/ingdev/projects/TIZIANO/data_preproc"
parfor isub=subjects
    isub
    for irun=runs
        cProject1917_preproc1_reref_filt_trim(isub, irun, rootdir, preproc_dir)
    end
end
%% function definition
function cProject1917_preproc1_reref_filt_trim(isub, irun, data_dir, preproc_dir)
data_dir = sprintf('%s%sdata%ssub-%03d',data_dir,filesep,filesep,isub);
preproc_dir = sprintf('%s%ssub-%03d',preproc_dir,filesep,isub);
if ~exist(preproc_dir,'dir')
    mkdir(preproc_dir);
end
MEGdir = dir(sprintf('%s%sses-meg01%smeg%s*.ds*',data_dir,filesep,filesep,filesep)); % extracting the meatadata from this file
MEGdir = fullfile(MEGdir.folder,MEGdir.name); % appends the parent directory and the filename of the .ds file
outdir = sprintf('%s%spreprocessing',preproc_dir,filesep);
if ~exist(outdir,'dir')
    mkdir(outdir);
end

% start up Fieldtrip
ft_defaults

% parameters
fsample = 1200; % sampling frequency
constant_delay_photodiode = 10; % add additional 10 samples, because photodiode (and therefore stimulus) always 1 screen refresh cycle behind MEG trigger, which at 120Hz is 8.333 msec, or at 1200Hz MEG signal, 10 samples.
starttrigger = 2;
stoptrigger = 255;

% load events
events = ft_read_event(MEGdir); % reads the whole time series of information about trigges/stimuli/catch trials ... in memory
beep
% select only the trigger channel called 'UPPT001'
events = events(strcmp(extractfield(events,'type'),'UPPT001'));

% add additional 10 samples, because photodiode (and therefore stimulus) always 1 screen refresh cycle behind MEG trigger, which at 120Hz is 8.333 msec, or at 1200Hz MEG signal, 10 samples.
temp = num2cell(extractfield(events,'sample')+constant_delay_photodiode);
[events.sample] = temp{:};

% to extract the datapts corresponding to the beginning and end of runs
startID = find(extractfield(events,'value')==starttrigger);% start from second trigger, because strange inconsistent delay between first two, and we want to exclude movie onset anyway
endID = find(extractfield(events,'value') == stoptrigger);% i.e., trigger for movie end
startsample = extractfield(events(startID),'sample');
endsample = extractfield(events(endID),'sample')+1*fsample; % add extra sec

% select channels and segment belonging to irun and load data
cfg = [];
cfg.dataset = MEGdir; % the path of the data
cfg.continuous = 'yes'; % default is cutting into trials based on triggers, but we want to keep continuous
cfg.trl = [startsample(irun) endsample(irun) 0]; % 0 is just padding for the ft_preprocessing func
cfg.channel = {'MEG', 'MEGREF', 'UPPT001'};
% (991 secs)
data = ft_preprocessing(cfg); % just reading the data of the first run here - > data end up being chans by timepoints{[array]}
beep
% in some rare cases something strange like a instantaneous baseline jump
% (sub15, run6), or huge blinks (sub02), we need to remove these BEFORE the
% rereferencing and filtering below as this would make it worse.
if isub == 15 && irun == 6
    badt = [181.6 181.8];
    badID = dsearchn(data.time{1}',badt');

    % subtract baseline separately before and after the jump
    data.trial{1}(:,1:badID(1)) = data.trial{1}(:,1:badID(1)) - repmat(mean(data.trial{1}(:,1:badID(1)),2),1,length(1:badID(1)));
    data.trial{1}(:,badID(2)+1:end) = data.trial{1}(:,badID(2)+1:end) - repmat(mean(data.trial{1}(:,badID(2)+1:end),2),1,length(data.time{1})-badID(2));
    data.trial{1}(:,badID(1)+1:badID(2)) = 0;
end

% 3rd order gradient correction (?) (48sec)
cfg = [];
cfg.gradient = 'G3BR'; % reference to external electrodes
data = ft_denoise_synthetic(cfg, data);
beep
% demean and filter
cfg = [];
cfg.bsfreq = [49.9 50.1; 99.9 100.1; 149.9 150.1];
cfg.demean = 'yes';
cfg.padding = 1000; % new length of the video in seconds (we are adding 50s at the beginning and end) our video is almost 900 sec long, and we need a lot of extra padding for such a low cutoff of 0.01 Hz ->
cfg.padtype = 'mirror';% we need to use mirror because there is no extra data
cfg.hpfilter = 'yes'; % high-pass filter
cfg.hpfreq = 0.01; % takes off freq below it
cfg.hpfiltord = 2; % (?)
cfg.bsfilter = 'yes'; % band-stop (notch) filter
cfg.bsfiltord = 2;
%%
data = ft_preprocessing(cfg, data); % applies parms
whos("data")
beep
%%
% if the movie was paused during a run, the pause start and end are indicated with
% triggers 251 and 252, respectively. This data segment should be
% removed from further analyses, which we do here. Same for catch trials,
% start and end of which are indicated with 241 and 242 respectively
currentevents = events(startID(irun):endID(irun));

if any(extractfield(currentevents,'value') == 251)

    error('did not check the code for cutting out breaks yet!');

    breakstartID = find(extractfield(currentevents,'value') == 251);

    % if during the break the break button was accidentally pressed
    % again (i.e., giving multiple 251 in a row), we want to remove those because
    % they're not real break starts
    if any(diff(breakstartID) == 1)
        doublestartID = (diff(breakstartID) == 1)+1;
        breakstartID(doublestartID) = [];
    end

    % loop over breaks if multiple
    breaks2remove = [];
    for ibreak = 1:length(breakstartID)

        breakstart = extractfield(currentevents(breakstartID(ibreak)),'sample');

        % check how many 'good' samples we can keep before break start,
        % so we can calculate how many samples to keep after break end
        prebreaksamples = breakstart - extractfield(currentevents(breakstartID(ibreak)-1),'sample');
        postbreaksamples = 4*data.fsample - prebreaksamples;% 4 seconds

        % find first event after break start that is not a break-type
        % event, i.e., has a value lower than 250, because that's the
        % first good event after the break
        nextgoodevent = breakstartID(ibreak) + find(extractfield(currentevents(breakstartID(ibreak)+1:end),'value')<250,1);

        % remove segment from break start, until the next correct event
        % minus the amount of samples left after the break, so the time
        % between the surrounding segments remains 4 sec / 4800 samples
        breaks2remove = [breaks2remove breakstart:extractfield(currentevents(nextgoodevent),'sample') - postbreaksamples];

    end

    % a run doesn't start at sample 1, but at data.sampleinfo(1),
    % so we still need to subtract that
    breaks2remove = breaks2remove - data.sampleinfo(1)+1;

    % now remove the breaks from the data
    data.trial{1}(:,breaks2remove) = [];
    data.time{1}(end-length(breaks2remove)+1:end) = [];
    data.sampleinfo(2) = data.sampleinfo(2) - length(breaks2remove);
end

% and the catch trials (picking the timepts of the catch trials for later cutting)
occstartID = find(extractfield(currentevents,'value') == 241);
occendID = find(extractfield(currentevents,'value') == 242);

% loop over breaks if multiple
occ2remove = [];
for iocc = 1:length(occstartID)

    occstartsamp = extractfield(currentevents(occstartID(iocc)),'sample');

    % check how many 'good' samples we can keep before task start,
    % so we can count backwards from first good event after task
    preoccsamples = occstartsamp - extractfield(currentevents(occstartID(iocc)-1),'sample');
    postoccsamples = 5*data.fsample - preoccsamples;% 5 seconds

    % find first good event after occlusion end, excluding duplicates
    nextgoodevent = occendID(iocc) + find(extractfield(currentevents(occendID(iocc)+1:end),'value')==extractfield(currentevents(occstartID(iocc)-1),'value') + 1,1);

    % remove segment from occlusion start, until the next correct event
    % minus the amount of samples left after the occlusion, so the time
    % between the surrounding segments remains 5 sec / 6000 samples
    occ2remove = [occ2remove occstartsamp:extractfield(currentevents(nextgoodevent),'sample') - postoccsamples];

end

% a run doesn't start at sample 1, but at data.sampleinfo(1),
% so we still need to subtract that
occ2remove = occ2remove - data.sampleinfo(1)+1;

% now remove the breaks from the data
data.trial{1}(:,occ2remove) = [];
data.time{1}(end-length(occ2remove)+1:end) = [];
data.sampleinfo(2) = data.sampleinfo(2) - length(occ2remove);

fn2save = sprintf('%s%sdata_reref_filt_trim_sub%03d_run%02d',outdir,filesep,isub,irun);
% DOWNSAMPLING PART
cfg = [];
cfg.resamplefs = 600;
cfg.method = 'resample';
data = ft_resampledata(cfg, data);
save(fn2save,'data', '-v7.3');
clear data
end

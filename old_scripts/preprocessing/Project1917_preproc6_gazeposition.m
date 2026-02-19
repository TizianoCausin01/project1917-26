function Project1917_preproc6_gazeposition(parms,isub,irun)
% Project 1917
% gaze position based on eye-tracker data

if isfield(parms,'cluster')
    % set paths
    rootdir = '//mnt/storage/tier2/morwur/Projects/INGMAR/Project1917';
    % start up Fieldtrip
    addpath('//mnt/storage/tier2/morwur/Projects/INGMAR/toolboxes/fieldtrip-20231220');
else
    % set paths
   rootdir='/Users/tizianocausin/Desktop/dataRepository/RepDondersInternship/project1917/example data'; %!!CHANGED PATH
    % start up Fieldtrip
    addpath('/Users/tizianocausin/Desktop/programs/fieldtrip-20240110')
end

% set Fieldtrip defaults
ft_defaults

% hardcoded parameters
resHor = 1920;
resVer = 1080;
avg_delay_eyetracker = 10;% in case no eye movement ICA component was found, we use 10 samples, which was roughly the average across subjects / runs.

datadir = sprintf('%s%sdata%ssub-%03d',rootdir,filesep,filesep,isub);
EYEdir = sprintf('%s%sses-meg01%seye',datadir,filesep,filesep);

outdir = sprintf('%s%sdata%ssub-%03d%sgaze',rootdir,filesep,filesep,isub,filesep);
if ~exist(outdir,'dir')
    mkdir(outdir);
end

% temporarily load MEG file to get eye movement component for temporal alignment
fn2load = sprintf('%s%spreprocessing%sdata_reref_filt_trim_sub%03d_run%02d',datadir,filesep,filesep,isub,irun);
load(fn2load,'data');

% load ICA components
fn2load = sprintf('%s%spreprocessing%sica_weights_sub%03d_run%02d',datadir,filesep,filesep,isub,irun);
load(fn2load,'unmixing', 'topolabel');

% load bad channels and bad segments and remove before ICA checking
fn2load = sprintf('%s%spreprocessing%sbadchan_sub%03d_run%02d',datadir,filesep,filesep,isub,irun);
load(fn2load, 'megchan_keep');

cfg = [];
cfg.channel = megchan_keep(:)';
data = ft_selectdata(cfg, data);

% now plot components for check
cfg = [];
cfg.method = 'predefined mixing matrix';
cfg.demean = 'no';
cfg.channel = {'MEG'};
cfg.topolabel = topolabel;
cfg.unmixing = unmixing;
comp = ft_componentanalysis(cfg, data);
clear data

% find eye movement component %!!COMMENTED BC I DON'T HAVE BAD COMPONENTS
% fn2load = sprintf('%s%spreprocessing%sica_badcomps_sub%03d_run%02d',datadir,filesep,filesep,isub,irun);
% load(fn2load, 'badcomps', 'badcomps_reasons');
% 
% eyecompID = badcomps(strcmp(badcomps_reasons,'eyemov'));
% 
% cfg = [];
% cfg.channel = eyecompID;
% comp = ft_selectdata(cfg,comp);
% 
% demean and filter
% cfg = [];
% cfg.demean = 'yes';
% cfg.padding = 1200;% our video is almost 900 sec long, and we need a lot of extra padding for such a low cutoff of 0.01 Hz
% cfg.padtype = 'mirror';% we need to use mirror because there is no extra data
% cfg.hpfilter = 'yes';
% cfg.hpfreq = 0.05;
% cfg.hpfiltord = 2;
% cfg.lpfilter = 'yes';
% cfg.lpfreq = 30;
% cfg.lpfiltord = 2;
% cfg.bsfilter = 'yes';
% cfg.bsfreq = [49.9 50.1 ; 99.9 100.1 ; 149.9 150.1];
% cfg.bsfiltord = 2;
% comp = ft_preprocessing(cfg, comp);

% load gaze position in eyelink file
fileEYE = sprintf('%s%sIdVs%02dr%d.asc',EYEdir,filesep,isub,irun);

% extract events
event_eye = ft_read_event(fileEYE);
hdr_eye = ft_read_header(fileEYE);

% event samples start from start of eyetracker recording, but we want them to start at 'starttrigger', which is where our MEG data starts, so here we shift the sample values accordingly
startsamp = event_eye(strcmp(extractfield(event_eye,'type'),'INPUT'));
startsamp = extractfield(startsamp(extractfield(startsamp,'value')==parms.starttrigger),'sample');
temp = num2cell(extractfield(event_eye,'sample')-startsamp);
[event_eye.sample] = temp{:};

% select triggers
triggers_eye = event_eye(strcmp('INPUT',extractfield(event_eye,'type')));
triggers_eye = triggers_eye(extractfield(triggers_eye,'value')~=0);

% load asc file
cfg = [];
cfg.dataset = fileEYE;
cfg.trl = [startsamp hdr_eye.nSamples 0];
cfg.channel = {'2','3','4'};
eye = ft_preprocessing(cfg);

% find blinks
startevent = find(extractfield(event_eye,'sample') == 0,1);
blinkID = zeros(size(eye.time{1}));
for ievent = startevent:length(event_eye)

    if strcmp(extractfield(event_eye(ievent),'type'),'BLINK') %&& strcmp(extractfield(event_eye(ievent-1),'type'),'SACC')

        % find first SACC before the blink, which is when blink
        % actually started happening
        saccID = ievent - find(strcmp(extractfield(event_eye(ievent-1:-1:ievent-10),'type'),'SACC'),1);

        blinkonset = extractfield(event_eye(saccID),'sample');
        blinkoffset = blinkonset + extractfield(event_eye(saccID),'duration');

        blinkID(blinkonset:blinkoffset) = 1;
    end

end

% additionally, find blinks by finding zeros in the pupil dilation signal, or jumps to strange high values that fall outside of the screen
blinkID(eye.trial{1}(3,:) == 0 | eye.trial{1}(1,:) > resHor | eye.trial{1}(2,:) > resVer) = 1;
% additional jumps
jumps = false(size(blinkID));
jumps(2:end) = diff(eye.trial{1}(3,:)) < -50;
blinkID(jumps) = 1;

% change from logical vector into indices vector
blinkID = find(blinkID);

% find blink onsets and offsets, so we can add some padding around the blink
blinkonsets = blinkID(1);% first blink onset is first blink ID
blinkoffsets = [];

for iblink = 1:length(blinkID)-1

    if blinkID(iblink+1) - blinkID(iblink) > 1
        blinkonsets = [blinkonsets blinkID(iblink+1)];
        blinkoffsets = [blinkoffsets blinkID(iblink)];
    end

end

blinkoffsets(end+1) = blinkID(end);

% add padding around blinks, coz signal not stable in that interval
blinkonsets = blinkonsets - parms.preblinkpadding*eye.fsample;
blinkoffsets = blinkoffsets + parms.postblinkpadding*eye.fsample;

% sometimes after padding, first blink starts before t = 0, and last blink ends after t = max, so we need to cut that off again
if blinkonsets(1) < 1
    blinkonsets(1) = 1;
end
if blinkoffsets(end) > length(eye.time{1})
    blinkoffsets(end) = length(eye.time{1});
end

% remove blinks from data by replacing them with average gaze position in past 'preblinkgazetime' seconds
eye.trial{1}(4,:) = 0;% here we keep track of missing data, and after cutting out catch trials and breaks we compute final percentage of missing data for subject rejection criteria.
for iblink = 1:length(blinkonsets)

    blinklen = blinkoffsets(iblink)-blinkonsets(iblink)+1;

    % check to make sure no blink is larger than 5 sec. This would anyway not be a blink, but more likely some missing eyetracker data. Up to 5 sec we accept and still use the same method to estimate
    % eye position, but more than 5 sec we just take the middle of the screen as gaze position. Additionally, we count how much missing eyetracker data there is in total, and if above a certain 
    % threshold we have to consider rejecting this subject altogether, because no accurate gaze-dependent models can be created.    
    % Last, if first blink is too close to start of trial and there's not enough 'preblinkgazetime' to average over, we also take the centre of the screen (resHor / 2 and resVer / 2)
    if blinklen > 5*eye.fsample || blinkonsets(iblink) < 1 + parms.preblinkgazetime*eye.fsample
        eye.trial{1}(:,blinkonsets(iblink):blinkoffsets(iblink)) = repmat([resHor/2 ; resVer/2 ; 5000 ; 1],1,blinklen);
    else
        eye.trial{1}(1:3,blinkonsets(iblink):blinkoffsets(iblink)) = repmat(mean(eye.trial{1}(1:3,blinkonsets(iblink)-eye.fsample*parms.preblinkgazetime:blinkonsets(iblink)),2),1,blinklen);
        eye.trial{1}(4,blinkonsets(iblink):blinkoffsets(iblink)) = 1;
    end

end

% if the movie was paused during a run, the pause start and end are indicated with
% triggers 251 and 252, respectively. This data segment should be
% removed from further analyses, which we do here. Same for catch trials,
% start and end of which are indicated with 241 and 242 respectively
if any(extractfield(triggers_eye,'value') == 251)

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
    % breaks2remove = breaks2remove - data.sampleinfo(1)+1;

    % now remove the breaks from the data
    data.trial{1}(:,breaks2remove) = [];
    data.time{1}(end-length(breaks2remove)+1:end) = [];
    data.sampleinfo(2) = data.sampleinfo(2) - length(breaks2remove);
end

% and the catch trials
occstartID = find(extractfield(triggers_eye,'value') == 241);
occendID = find(extractfield(triggers_eye,'value') == 242);

% loop over catch trials
occ2remove = [];
for iocc = 1:length(occstartID)

    occstartsamp = extractfield(triggers_eye(occstartID(iocc)),'sample');

    % check how many 'good' samples we can keep before task start,
    % so we can count backwards from first good event after task
    preoccsamples = occstartsamp - extractfield(triggers_eye(occstartID(iocc)-1),'sample');
    postoccsamples = parms.ITI*eye.fsample - preoccsamples;% 5 seconds

    % find first good event after occlusion end, excluding duplicates
    nextgoodevent = occendID(iocc) + find(extractfield(triggers_eye(occendID(iocc)+1:end),'value')==extractfield(triggers_eye(occstartID(iocc)-1),'value') + 1,1);

    % remove segment from occlusion start, until the next correct event
    % minus the amount of samples left after the occlusion, so the time
    % between the surrounding triggers remains 5 sec
    occ2remove = [occ2remove occstartsamp:extractfield(triggers_eye(nextgoodevent),'sample') - postoccsamples];

end

% now remove the breaks from the data
eye.trial{1}(:,occ2remove) = [];
eye.time{1}(end-length(occ2remove)+1:end) = [];
eye.sampleinfo(2) = eye.sampleinfo(2) - length(occ2remove);

% smoothen gaze position to get rid of artificial jitter during fixation due to noise
eye.trial{1} = ft_preproc_lowpassfilter(eye.trial{1}, eye.fsample, parms.lpfreqeye);
% 
% % we loaded in eyetracker data until end of the file, but MEG data until last trigger + 1 sec, which is a bit longer
% % so before any comparisons, we should remove this extra bit from the MEG data
% endtime = dsearchn(comp.time{1}',eye.time{1}(end));
% comp.time{1}(endtime+1:end)=[];
% comp.trial{1}(:,endtime+1:end)=[];
% 
% % resample to match MEG data
% tOld = eye.time{1};
% tNew = comp.time{1};
% 
% eye.trial{1} = interp1(tOld,eye.trial{1}',tNew,'pchip')';
% eye.time{1} = tNew;
% eye.fsample = comp.fsample;
% 
% % now remove all bad segments from both signals to improve delay estimation
% fn2load = sprintf('%s%spreprocessing%sbadseg_sub%03d_run%02d',datadir,filesep,filesep,isub,irun);
% load(fn2load, 'BAD_lowfreq');
% 
% % combine two sources of bad segments
% badsegs = BAD_lowfreq;
% 
% for iseg = 1:size(badsegs,1)
%     badsegs(iseg,:) = dsearchn(comp.time{1}',badsegs(iseg,:)')';
% end
% 
% % create logical vector of samples to keep
% samples2keep = true(size(comp.time{1}));
% for iseg = 1:size(badsegs,1)
%     samples2keep(badsegs(iseg,1):badsegs(iseg,2)) = false;
% end
% 
% comp.trial{1}(:,~samples2keep) = 0;
% 
% 
% % find delay between signals, positive d means eyetracker data delayed by d samples compared to MEG data
% % if isempty(eyecompID)%!!CHAGED BC I DON'T HAVE it
% % if ~exist(eyecompID) %!!CHANGED bc I dont have a eyecomponent
% % d = avg_delay_eyetracker;
% % else
%     d = finddelay(comp.trial{1}',eye.trial{1}(1,:));
% % end
% 
% % make sure d is not very strange. If so, probably because eyemov component
% % was not clean, so better use the average delay then
% if d < 0 || d > 20
%     d = avg_delay_eyetracker;
% end

% if multiple eyemov ICAs, average delay
% d = round(mean(d));
d=0
% shift eyetracker signal by d, and fill end of signal so it keeps same length
eye.trial{1}(:,end+1:end+d) = repmat(eye.trial{1}(:,end),1,d);
eye.trial{1} = eye.trial{1}(:,d+1:end);
% 
% % if isempty(eyecompID)
% % if ~exist(eyecompID) %!!CHANGED BC I DON'T HAVE IT
% %     finalcorr = nan;
% %     finald = 0;
% % else
%     finalcorr = corr(eye.trial{1}(1,:)',comp.trial{1}');
%     finald = finddelay(comp.trial{1}',eye.trial{1}(1,:));
% %end

% resample the data to the original movie frame rate used for the models
tNew = 0:1/parms.fsNew:eye.time{1}(end);
tOld = eye.time{1};

gaze = interp1(tOld,eye.trial{1}',tNew,'nearest','extrap')';

percentage_missing_data = 100*sum(gaze(4,:))/size(gaze,2);

fn2save = sprintf('%s%sgaze_sub%03d_run%02d_%dHz',outdir,filesep,isub,irun,round(parms.fsNew));

fsNew = parms.fsNew;
% save(fn2save, 'gaze', 'tNew', 'fsNew', 'd', 'finalcorr', 'finald', 'percentage_missing_data', '-v7.3');
save(fn2save, 'gaze', 'tNew', 'fsNew', 'd',   'percentage_missing_data', '-v7.3');


clear data


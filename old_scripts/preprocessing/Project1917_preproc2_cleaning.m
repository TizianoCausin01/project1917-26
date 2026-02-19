addpath('/Users/tizianocausin/Desktop/programs/fieldtrip-20240110')
ft_defaults
preproc_dir = '/Volumes/TIZIANO/data_preproc';
subjects = 10;
target_runs = 1:6;
for isub = subjects
    tProject1917_preproc2_cleaning(preproc_dir, isub, target_runs)
end % for isub = subjects

% function
function tProject1917_preproc2_cleaning(preproc_dir, isub, target_runs)

% Project 1917
% preprocessing script 1 - bad channel and segment detection

% loop over subjects
    % set paths 
    % start up Fieldtrip
    preproc_dir = sprintf('%s%ssub-%03d',preproc_dir,filesep,isub);
    indir = sprintf('%s%spreprocessing',preproc_dir,filesep);
    % loop over runs
    for irun = target_runs
        fn2load = sprintf('%s%sdata_reref_filt_trim_sub%03d_run%02d',indir,filesep,isub,irun);
        load(fn2load,'data');
        % filter for segment outliers in the frequency range we're interested
        % in for now (i.e., max beta, or below 50)
        cfg = [];
        cfg.lpfilter = 'yes';
        cfg.lpfreq = 50; % look at +low freq bc ur inter.d into slow fluctuations

        data_filt = ft_preprocessing(cfg, data); % ft_preprocessing to apply filters

        % segment continuous data into 500 msec 'fake trials'
        cfg = [];
        cfg.length               = .5;
        data_segmented           = ft_redefinetrial(cfg, data_filt);
        clear data_filt

        cfg = [];
        cfg.channel = {'MEG'};
        cfg.method = 'summary';
        cfg.layout = 'CTF275.lay';
        cfg.keeptrial   = 'yes';
        beep
        data_segmented = ft_rejectvisual(cfg, data_segmented); % automatic artifact rejection
        % store which channels to keep
        megchan_keep = data_segmented.label; % only the channels that were not rejected from rejectvisual (?)
        fn2save = sprintf('%s%sbadchan_sub%03d_run%02d',indir,filesep,isub,irun);
        save(fn2save, 'megchan_keep'); % outputs both channels and segments and then select bad channels
        % extract bad segments from first test
        BAD_lowfreq = data_segmented.cfg.artfctdef.summary.artifact-data_segmented.sampleinfo(1,1)+1;
        clear data_segmented
        % filter for muscle activity, cut into 500 msec segments, and run ft_rejecvisual summary again
        cfg = [];
        cfg.bpfilter = 'yes';
        cfg.bpfreq = [110 140]; % keep these frequencies to detect muscle activity ->
        cfg.bpfilttype = 'but'; % butterworth filter
        cfg.bpfiltord = 4;
        cfg.hilbert = 'yes';
        cfg.channel = megchan_keep;
        data_filt = ft_preprocessing(cfg, data);
        % segment continuous data into 500 msec 'fake trials'
        cfg                      = [];
        cfg.length               = .5;
        data_segmented           = ft_redefinetrial(cfg, data_filt); % and we'll remove these small "trials" when they present heavy muscle artifacts w/ rejectvisual 
        clear data_filt

        cfg = [];
        cfg.channel = {'MEG'};
        cfg.method = 'summary';
        cfg.layout = 'CTF275.lay';
        cfg.keepchannel = 'yes';
        cfg.keeptrial   = 'yes';
        beep
        data_segmented = ft_rejectvisual(cfg, data_segmented); % again rejectvisual after the bandpass 110-140
        % extract bad segments
        BAD_muscle = data_segmented.cfg.artfctdef.summary.artifact-data_segmented.sampleinfo(1,1)+1;
        clear data_segmented
        % convert from indices to time to prevent mistakes later on when
        % removing bad segments on downsampled data
        BAD_lowfreq = data.time{1}(BAD_lowfreq);
        BAD_muscle = data.time{1}(BAD_muscle);
        fn2save = sprintf('%s%sbadseg_sub%03d_run%02d',indir,filesep,isub,irun);
        save(fn2save,'BAD_lowfreq','BAD_muscle');
    end
end % EOF

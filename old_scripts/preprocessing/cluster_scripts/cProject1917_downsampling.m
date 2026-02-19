addpath("/home/tiziano.causin/adds_on/fieldtrip-20250114")
ft_defaults
% rootdir = "/mnt/storage/tier2/ingdev/projects/Project1917"
subjects=[3:10];
runs=1:6
preproc_dir = "/mnt/storage/tier2/ingdev/projects/TIZIANO/data_preproc";
for isub=subjects
    isub
    sub_dir = sprintf("%s/sub-%03d/preprocessing", preproc_dir, isub);
    for irun=runs
        fn2load = sprintf('%s/data_reref_filt_trim_sub%03d_run%02d.mat',sub_dir,isub,irun);
        load(fn2load,'data');
        cfg.resamplefs = 600;
        cfg.method = 'resample';
        data = ft_resampledata(cfg, data);
        save(fn2load,'data', '-v7.3'); % overwrites the existing file
    end
end

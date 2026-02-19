function Project1917_dRSA_singlesubstats(cfg)

rootdir = '\\cimec-storage5.unitn.it\MORWUR\Projects\INGMAR\Project1917';
addpath(genpath(rootdir));
addpath(genpath('\\cimec-storage5.unitn.it\MORWUR\Projects\INGMAR\ProjectMovie\code\dRSA\boundedline-toolbox'));

if cfg.similarity == 0
    simstring = 'corr';
elseif cfg.similarity == 1
    simstring = ['pcr_' num2str(cfg.nPCRcomps) 'comps'];
end

indir = sprintf('%s%sresults%sdRSA%s%s_%dHz_%dstim_%dsec_%diterations_%dMNN_randperms_hp02hz',rootdir,filesep,filesep,filesep,simstring,cfg.fsNew,cfg.nstim,cfg.stimlen,cfg.iterations,cfg.MNN);
outdir = sprintf('%s%sresults%sdRSA%s%s_%dHz_%dstim_%dsec_%diterations_%dMNN_hp02hz',rootdir,filesep,filesep,filesep,simstring,cfg.fsNew,cfg.nstim,cfg.stimlen,cfg.iterations,cfg.MNN);

%% load data
for iroi = cfg.ROI
    
    dRSAperm = zeros(length(cfg.randperms),length(cfg.subjects),length(cfg.models2test),cfg.maxlatency*cfg.fsNew*2+1);
    for itest = 1:length(cfg.models2test)

        for isub = 1:length(cfg.subjects)

            for iperm = cfg.randperms

                % load real data
                fn = sprintf('%s%cSUB%02d_%s_%s_perm%04d', indir, filesep, cfg.subjects(isub), cfg.ROInames{iroi}, cfg.modelnames{itest}, iperm);
                load(fn,'dRSA');

                dRSAperm(iperm,isub,itest,:) = dRSA;

            end

        end

    end

    fn2save = sprintf('%s%cALLSUB_%s_ALLMOD_%04dperms', outdir, filesep, cfg.ROInames{iroi}, length(cfg.randperms));
    save(fn2save,'dRSAperm');

end

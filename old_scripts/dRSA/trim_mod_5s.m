models = ["CE", "SC", "beta", "gamma"];
path2mod = "/Volumes/TIZIANO/models/Project1917_%s_run%02d_movie24Hz.mat";
for imod = 1:length(models)
    for irun = 1:3
        path2currmod = sprintf(path2mod, models(imod), irun);
        load(path2currmod)
        seg2cut = 119;
        vecrep(:, 1:seg2cut) = [];
        S.vecrep = vecrep;
        S.tVid = tVid;
        S.fsVid = fsVid;
        save(path2currmod,'-fromstruct', S,'-v7.3')
    end % for irun = 1:3
end % for imod = 1:length(models)

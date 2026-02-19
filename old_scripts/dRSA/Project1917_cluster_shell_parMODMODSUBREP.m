function Project1917_cluster_shell_parMODMODSUBREP(parms)
% determine whether to send job to cluster, or run script locally

if isfield(parms,'cluster')% send to cluster
    
    matlab_version = sscanf(version('-release'),'%d%s');
    if matlab_version(1) ~= 2020
        error('The package for sending jobs to the cluster only works on matlab 2020, please use that');
    end

    cluster = parcluster;
    job = createJob(cluster);
    
    % Assign tasks using your function
    for irep = parms.con
        for isub = parms.subjects
            for imod=parms.models2test
                for isim = parms.models2sim
                    createTask(job,str2func(parms.script2run), 0,{parms, isim, imod, isub, irep});
                end
            end
        end
    end
    % run task
    submit(job);
    
else% run locally
    
    feval(parms.script2run,parms,0,0);%if run locally, looping over subjects happens within function (saves time as you only have to load the data of a subject once for all ROIs)
    
end
function Project1917_preproc7_models_gazedep_OF_parallel(parms)

% set directories
rootdir = '\\cimec-storage5.unitn.it\MORWUR\Projects\INGMAR\Project1917';
addpath(genpath(rootdir));

% path to the movie data (mp4 or mov), and path where to store RDM
stimdir = fullfile(rootdir,'experiment','stimuli');

% hardcoded parameters
resHor = 1920;% horizontal screen resolution
resVer = 1080;% vertical screen resolution
[columnsInImage, rowsInImage] = meshgrid(1:resHor, 1:resVer);% for the circle around the gaze location, we need to temporarily create this info
wholescreen = zeros(resVer,resHor,'single');

vecrep_runs = cell(length(parms.runs),1);
fsVid_runs = cell(length(parms.runs),1);
tVid_runs = cell(length(parms.runs),1);
parfor irun = parms.runs

    % on runs 1 and 4 select first 15 minute fragment
    % on runs 2 and 5 select second 15 minute fragment
    % on runs 3 and 6 select third 15 minute fragment
    if any(irun == [1 4])
        ipart = 1;
    elseif any(irun == [2 5])
        ipart = 2;
    elseif any(irun == [3 6])
        ipart = 3;
    end

    gaze = cell(length(parms.subjects),1);
    maxlen = zeros(length(parms.subjects),1);
    for isub = 1:length(parms.subjects)

        gazedir = sprintf('%s%sdata%ssub-%03d%sgaze',rootdir,filesep,filesep,parms.subjects(isub),filesep);

        % load gaze position
        fn2load = sprintf('%s%sgaze_sub%03d_run%02d_24Hz',gazedir,filesep,parms.subjects(isub),irun);
        gazepos = load(fn2load);

        gaze{isub} = gazepos.gaze(1:2,:);

        maxlen(isub) = size(gaze{isub},2);

    end

    temp = zeros(length(parms.subjects),size(gaze{1},1),min(maxlen));
    for isub = 1:length(parms.subjects)
        temp(isub,:,:) = gaze{isub}(:,1:min(maxlen));
    end 
    gaze = temp;

    fn2load = sprintf('%s%sProject1917_movie_part%d_24Hz.mp4',stimdir,filesep,ipart);

    % load video header, and reset and prepare optical flow vector
    videoHeader = VideoReader(fn2load);
    videoHeader.CurrentTime = parms.tstart;
    opticFlow = opticalFlowFarneback;% use Farneback algorithm to compute optical flow field

    % prepare gray background screen
    bufferVer = (resVer-videoHeader.height)/2;
    bufferHor = (resHor-videoHeader.width)/2;

    % initialize variables
    ntime = floor(videoHeader.NumFrames-parms.tstart*videoHeader.FrameRate);
    nfeature = floor(pi*parms.gazeradius.^2/parms.sdsf);
    vecrep = zeros(length(parms.subjects),nfeature,ntime,3,'uint8');

    iframe = 0;
    while hasFrame(videoHeader)

        iframe = iframe+1;

        clc;
        disp(['computing vector representations of frame ' num2str(iframe)]);

        frame = readFrame(videoHeader);

        frame = rgb2gray(frame);

        % estimate optical flow vectors
        flow = estimateFlow(opticFlow,frame);
        % concatenate magnitude, and x and y direction using cosine and sine of angle orientation to make sure the vector length is 1, i.e., without magnitude
        matrep = cat(3,flow.Magnitude,cos(flow.Orientation),sin(flow.Orientation));

        bgscreen = repmat(wholescreen,1,1,3);
        bgscreen(bufferVer+1:end-bufferVer,bufferHor+1:end-bufferHor,:) = matrep;
        
        % Add gray boundaries around frame in case gaze was at edge, then create the circle surrounding gaze location
        centerHor = round(gaze(:,1,iframe));
        centerVer = round(gaze(:,2,iframe));

        for isub = 1:length(parms.subjects)

            gazeCircle = (rowsInImage - centerVer(isub)).^2 + (columnsInImage - centerHor(isub)).^2 <= parms.gazeradius.^2;
            gazeCircle = repmat(gazeCircle,1,1,3);% 3 models

            % select gaze circle and vectorize
            vecrep_frame = zeros(sum(sum(gazeCircle(:,:,1))),size(matrep,3));
            for imod = 1:3
                temp = squeeze(bgscreen(:,:,imod));
                vecrep_frame(:,imod) = temp(squeeze(gazeCircle(:,:,imod)));
            end

            % spatially downsample
            vecrep_frame = vecrep_frame(1:parms.sdsf:end,:);

            % reduce file size, separately per model (i.e., scale is very different for magnitude compared to direction)
            temp = zeros(size(vecrep_frame),'uint8');
            for imod = 1:3
                temp(:,imod) = uint8(rescale(vecrep_frame(:,imod),1,2^8));
            end

            vecrep(isub,1:size(vecrep_frame,1),iframe,:) = temp;

        end

    end

    % remove extra empty pixels in vecrep
    vecrep(:,vecrep(1,:,1,1)==0,:,:) = [];

    vecrep_runs{irun} = vecrep;

    % resample to new sampling rate in parms.fsNew, only 'nearest' makes sense for movie frames
    fsVid_runs{irun} = videoHeader.framerate;
    tVid_runs{irun} = 0:1/videoHeader.framerate:size(vecrep,3)/videoHeader.framerate-1/videoHeader.framerate;
    % tNew = 0:1/parms.fsNew:size(vecrep,2)/fsVid-1/parms.fsNew;

    % for each new time point, find index of nearest old time point
    % old2newID = dsearchn(tVid',tNew');
    % vecrep = vecrep(:,old2newID);

end
clear vecrep

% save vector representation to use in dRSA script
for irun = parms.runs
    for isub = 1:length(parms.subjects)

        outdir = sprintf('%s%sdata%ssub-%03d%smodels',rootdir,filesep,filesep,parms.subjects(isub),filesep);
        if ~exist(outdir,'dir')
            mkdir(outdir);
        end

        OFmag = squeeze(vecrep_runs{irun}(isub,:,:,1));
        OFdir = [squeeze(vecrep_runs{irun}(isub,:,:,2)) ; squeeze(vecrep_runs{irun}(isub,:,:,3))];
        tVid = tVid_runs{irun};
        fsVid = fsVid_runs{irun};

        % save vector representation to use in dRSA script
        fn2save = sprintf('%s%sProject1917_OFmag_sub%03d_run%02d_movie24Hz_gazerad%d',...
            outdir,filesep,parms.subjects(isub),irun,parms.gazeradius);

        vecrep = OFmag;
        save(fn2save,'vecrep','tVid','fsVid','-v7.3');
            
        fn2save = sprintf('%s%sProject1917_OFdir_sub%03d_run%02d_movie24Hz_gazerad%d',...
            outdir,filesep,parms.subjects(isub),irun,parms.gazeradius);

        vecrep = OFdir;
        save(fn2save,'vecrep','tVid','fsVid','-v7.3');

    end
end% run loop



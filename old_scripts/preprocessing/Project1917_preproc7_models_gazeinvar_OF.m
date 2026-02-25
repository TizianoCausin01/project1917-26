function Project1917_preproc7_models_gazeinvar_OF(parms)

% set directories
rootdir='/Users/tizianocausin/Desktop/dataRepository/RepDondersInternship/project1917/'; %!!CHANGED PATH , DIFFERENT FROM USUAL BC STIM ARE IN ANOTHER FOLDER
addpath(genpath(rootdir));

% path to the movie data (mp4 or mov), and path where to store RDM
stimdir = fullfile(rootdir,'stimuli'); %!!CHANGED BC I DON'T HAVE EXPERIMENT FOLDER

outdir = sprintf('%s%sdata%smodels',rootdir,filesep,filesep);
if ~exist(outdir,'dir')
    mkdir(outdir);
end

% on runs 1 and 4 select first 15 minute fragment
% on runs 2 and 5 select second 15 minute fragment
% on runs 3 and 6 select third 15 minute fragment
for ipart = 1 %!!it was 1:3

    % for dynamic models, run on both high (24 Hz) and low (12 Hz) temporal resolution movie
    for iresolution = 1:length(parms.OFtempres)

        fn2load = sprintf('%s%sProject1917_movie_part%d_%dHz.mp4',stimdir,filesep,ipart,parms.OFtempres(iresolution));

        % load video header, and reset and prepare optical flow vector
        videoHeader = VideoReader(fn2load);
        videoHeader.CurrentTime = parms.tstart;
        clear flow opticFlow
        opticFlow = opticalFlowFarneback;% use Farneback algorithm to compute optical flow field

        % initialize variables
        ntime = floor(videoHeader.NumFrames-parms.tstart*videoHeader.FrameRate);
        nfeature = floor(videoHeader.Width*videoHeader.Height/parms.sdsf);
        vecrep = zeros(nfeature,ntime,3,'uint8');

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
            vecrep_frame = reshape(matrep,[],size(matrep,3));

            % spatially downsample
            vecrep_frame = vecrep_frame(1:parms.sdsf:end,:);

            % reduce file size, separately per model (i.e., scale is very different for magnitude compared to direction)
            temp = zeros(size(vecrep_frame),'uint8');
            for imod = 1:3
                temp(:,imod) = uint8(rescale(vecrep_frame(:,imod),1,2^8));
            end

            vecrep(1:length(vecrep_frame),iframe,:) = temp;

            clear frame matrep vecrep_frame flow temp

        end

        % remove extra empty pixels in vecrep
        vecrep(vecrep(:,1)==0,:,:) = [];

        % resample to new sampling rate in parms.fsNew, only 'nearest' makes sense for movie frames
        fsVid = videoHeader.framerate;
        tVid = 0:1/fsVid:size(vecrep,2)/fsVid-1/fsVid;
        % tNew = 0:1/parms.fsNew:size(vecrep,2)/fsVid-1/parms.fsNew;

        % for each new time point, find index of nearest old time point
        % old2newID = dsearchn(tVid',tNew');
        % vecrep = vecrep(:,old2newID,:);

        OFmag = squeeze(vecrep(:,:,1));
        OFdir = [vecrep(:,:,2) ; vecrep(:,:,3)];

        % save vector representation to use in dRSA script
        fn2save = sprintf('%s%sProject1917_%s_run%02d_movie%dHz',...
            outdir,filesep,'OFmag',ipart,parms.OFtempres(iresolution));

        vecrep = OFmag;
        save(fn2save,'vecrep','tVid','fsVid','-v7.3');

        fn2save = sprintf('%s%sProject1917_%s_run%02d_movie%dHz',...
            outdir,filesep,'OFdir',ipart,parms.OFtempres(iresolution));

        vecrep = OFdir;
        save(fn2save,'vecrep','tVid','fsVid','-v7.3');

    end% temporal resolution of movies loop

end% run loop


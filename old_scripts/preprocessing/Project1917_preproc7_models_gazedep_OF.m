function Project1917_preproc7_models_gazedep_OF(parms)

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

for isub = parms.subjects

    gazedir = sprintf('%s%sdata%ssub-%03d%sgaze',rootdir,filesep,filesep,isub,filesep);

    outdir = sprintf('%s%sdata%ssub-%03d%smodels',rootdir,filesep,filesep,isub,filesep);
    if ~exist(outdir,'dir')
        mkdir(outdir);
    end

    for irun = parms.runs

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

        % for dynamic models, run on both high (24 Hz) and low (12 Hz) temporal resolution movie
        for iresolution = 1:length(parms.OFtempres)

            % load gaze position and downsample if necessary
            fn2load = sprintf('%s%sgaze_sub%03d_run%02d_24Hz',gazedir,filesep,isub,irun);
            gazepos = load(fn2load);

            if parms.OFtempres(iresolution) == 12% downsample gaze position
                tNew = 0:1/parms.OFtempres(iresolution):gazepos.tNew(end);
                tOld = gazepos.tNew;

                gazepos.gaze = interp1(tOld,gazepos.gaze',tNew,'nearest','extrap')';
                gazepos.tNew = tNew;
            end

            fn2load = sprintf('%s%sProject1917_movie_part%d_%dHz.mp4',stimdir,filesep,ipart,parms.OFtempres(iresolution));

            % load video header, and reset and prepare optical flow vector
            videoHeader = VideoReader(fn2load);
            videoHeader.CurrentTime = parms.tstart-1/videoHeader.FrameRate;
            clear flow opticFlow
            opticFlow = opticalFlowFarneback;% use Farneback algorithm to compute optical flow field

            % prepare gray background screen
            bufferVer = (resVer-videoHeader.height)/2;
            bufferHor = (resHor-videoHeader.width)/2;

            % initialize variables
            ntime = floor(videoHeader.NumFrames-parms.tstart*videoHeader.FrameRate);
            nfeature = floor(pi*parms.gazeradius.^2/parms.sdsf);
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

                % Add gray boundaries around frame in case gaze was at edge, then create the circle surrounding gaze location
                centerHor = round(gazepos.gaze(1,iframe));
                centerVer = round(gazepos.gaze(2,iframe));
                gazeCircle = (rowsInImage - centerVer).^2 + (columnsInImage - centerHor).^2 <= parms.gazeradius.^2;

                vecrep_frame = zeros(sum(sum(gazeCircle)),size(matrep,3));
                for imod = 1:size(matrep,3)
                    temp = wholescreen;
                    temp(bufferVer+1:end-bufferVer,bufferHor+1:end-bufferHor) = matrep(:,:,imod);
                    % select gaze circle and vectorize
                    vecrep_frame(:,imod) = temp(gazeCircle);
                    clear temp
                end

                % % store video section with gaze circle on top for check
                % matrep(~gazeCircle) = 0;
                % video4demo(iframe,:,:) = matrep;

                % spatially downsample
                vecrep_frame = vecrep_frame(1:parms.sdsf:end,:);

                % reduce file size, separately per model (i.e., scale is very different for magnitude compared to direction)
                temp = zeros(size(vecrep_frame),'uint8');
                for imod = 1:3
                    temp(:,imod) = uint8(rescale(vecrep_frame(:,imod),1,2^8));
                end

                vecrep(1:size(vecrep_frame,1),iframe,:) = temp;

                clear frame matrep vecrep_frame flow temp

            end

            % remove extra empty pixels in vecrep
            vecrep(vecrep(:,1,1)==0,:,:) = [];

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
            fn2save = sprintf('%s%sProject1917_%s_sub%03d_run%02d_movie%dHz_gazerad%d',...
                outdir,filesep,'OFmag',isub,irun,parms.OFtempres(iresolution),parms.gazeradius);

            vecrep = OFmag;
            save(fn2save,'vecrep','tVid','fsVid','-v7.3');

            fn2save = sprintf('%s%sProject1917_%s_sub%03d_run%02d_movie%dHz_gazerad%d',...
                outdir,filesep,'OFdir',isub,irun,parms.OFtempres(iresolution),parms.gazeradius);

            vecrep = OFdir;
            save(fn2save,'vecrep','tVid','fsVid','-v7.3');

        end% temporal resolution of movies loop

    end% run loop

end% subject loop


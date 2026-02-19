mod2load = "/Volumes/TIZIANO/models/Project1917_ViTPose_run01.h5";
vecrep = h5read(mod2load, "/kpts");
%% load vid and store it
stim_dir = "/Volumes/TIZIANO/stimuli/";
ipart = 1;
fn2load = sprintf('%sProject1917_movie_part%d_24Hz.mp4',stim_dir,ipart);
% load video header, and reset and prepare optical flow vector
videoHeader = VideoReader(fn2load);
height = videoHeader.Height;
width = videoHeader.Width;
n_frames = 100;
vid = uint8(zeros(height, width, 3, n_frames));
iframe=0;
frames2discard = 119;
vecrep(:,1:frames2discard) = [];
for i = 1:frames2discard
    frame=readFrame(videoHeader);
end

count = 0;
for i = 1:n_frames
    count = count +1;
    frame = readFrame(videoHeader);
    vid(:,:,:,i) = frame;
    % imshow(frame)
    % pause(.5)
    % hold on
    % x = data_final{1}(1,i) - bufferVer;
    % y = data_final{1}(2,i) - bufferHor;
    % plot(x,y, 'ro', 'MarkerSize', 7, 'MarkerFaceColor', 'r','LineWidth', 2);
    % hold off
end
%% load eyetracking data and downsample them to the fs of the vid
cfg = [];
cfg.resamplefs = 23.976;       % New sampling frequency in Hz (e.g., downsample to 100 Hz)
cfg.detrend     = 'no';     % Optional: remove linear trend
cfg.demean      = 'no';     % Optional: remove mean
parms.fsNew = 50;
% size_runs = [20953, 20761, 18961];
dataDir = "/Volumes/TIZIANO/eyetracking_data";
isub = 5;
irep = 1;
irun = 1;
fn2load=sprintf('%s/gaze_sub%03d_run%02d_50Hz.mat',dataDir,isub,irun);
load(fn2load);
data = gaze([1 2],:);
data_ft = [];
data_ft.label = {'x', 'y'};                 % one label per channel
data_ft.fsample = parms.fsNew;             % original sampling rate (e.g., 50 Hz)
data_ft.trial = {double(data)};            % trial data in a cell array
data_ft.time = {(0:size(data, 2)-1) / parms.fsNew};
data_resampled = ft_resampledata(cfg, data_ft);
data = data_resampled.trial{1};
%% takes off the gray margins
resHor = 1920;% horizontal screen resolution
resVer = 1080;% vertical screen resolution
bufferVer = (resVer-videoHeader.height)/2;
bufferHor = (resHor-videoHeader.width)/2;
%% TODO find a way to compute the closest body-posture and to use the model already at high fs
for i = 1:100
    figure
    imshow(vid(:,:,:,i))
    pause(.2)
    hold on
    x = data(1,i) - bufferVer;
    y = data(2,i) - bufferHor;
    % kpts = reshape(vecrep(:,i), 17,2,5);
    % for j=1:5
    %     plot(kpts(:,1,j),kpts(:,2,j), 'ro', 'MarkerSize', 3, 'MarkerFaceColor', 'r','LineWidth', 2);
    % end
    kpts = get_closest_kpts(vecrep(:,i), x, y, "min_dist")
    plot(kpts(:,1),kpts(:,2), 'ro', 'MarkerSize', 3, 'MarkerFaceColor', 'r','LineWidth', 2);
    plot(x,y, 'rx', 'MarkerSize', 20, 'MarkerFaceColor', 'r','LineWidth', 2);
    hold off
    pause(.3)
    close(gcf)
end
%% alternatives include: weighting by scores, using min dist, use only a subset of pts, normalize by obj size
function [curr_kpts, idx] = get_closest_kpts(mod_frame, x, y, crit)
    curr_frame = reshape(mod_frame, 17,2,5)
    if strcmp(crit, "mean")
        m = squeeze(mean(curr_frame));
        pt_mat = repmat([x;y], 1, 5); % brings the x,y gaze coords into the same shape as the mean
        d = sqrt(sum((m - pt_mat).^2, 1)); % computes the euclidean dist between gaze and each other pt

    elseif strcmp(crit, "min_dist")
        pt_mat = repmat([x, y], 17, 1, 5); % brings the x,y gaze coords into the same shape as the kpts
        dist_mat = squeeze(sum((curr_frame - pt_mat).^2, 2)); % it computes the euclidean dist of each kpt from gaze coords
        d = min(dist_mat); % for each pers, it computes the mindist 
    end
    [pt_dist, idx] = min(d);
    curr_kpts = curr_frame(:,:,idx)

end % EOF

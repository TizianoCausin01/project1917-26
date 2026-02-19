
addpath(genpath("/Volumes/TIZIANO/Project1917/code/LGNstatistics"))
%%
stim_dir = "/Volumes/TIZIANO/stimuli/";
out_dir = "/Volumes/TIZIANO/models/";
path2OF = "/Volumes/TIZIANO/models/Project1917_OFdir_run%02d_movie24Hz.mat";
img_len = [522, 1280];
n_squares = 3; % height, width
square_size = [floor(img_len(1)/n_squares), floor(img_len(2)/ n_squares)]; % floor to make sure we don't overshoot with the indexing
%% defines the onsets of the squares

onsets = zeros(2, n_squares); % onset rows on the 1st row, onset cols on the 2nd row
onsets(:, 1) = 1;  % initializing the onsets
for i = 2:n_squares
    onsets(:, i) = onsets(:, i-1) + square_size'; % adding square_size as a column vector on the rows of onsets
end % for i = 1:n_squares
%%
% in order the columns are : CE (contrast energy), SC (spatial coherence), beta (Weibull paramt), gamma (Weibull paramt)
models = ["CE", "SC", "beta", "gamma"];
n_mod = length(models);
SC_mat = zeros(n_squares^2, 3);
CE_mat = zeros(n_squares^2, 3);
beta_mat = zeros(n_squares^2, 3);
gamma_mat = zeros(n_squares^2, 3);
viewing_dist = .75;
% size_of_movie = .222; % in mt
% movie_size_factor = .7;
% num_of_pix_w_presented = img_len(2)*movie_size_factor/n_squares;
% dot_pitch = size_of_movie/num_of_pix_w_presented ;
parts = 1:3;
%%
parfor ipart=parts %loops over the parts of the movie
    sing_frame_mod = {zeros(n_squares^2, 3)};
    temp_mod = repmat(sing_frame_mod, n_mod, 1);
    mod_dim = 3*n_squares^2; % it outputs 3 parameters for each of the 9 squares
    OF_mod = sprintf(path2OF, ipart);
    S = load(OF_mod, 'tVid', 'fsVid');
    tVid = S.tVid;
    fsVid = S.fsVid;
    mod_len = length(tVid);
    sing_mod = {zeros(mod_dim, mod_len)}; % creates a temporary cell array with a model
    tot_mod = repmat(sing_mod, n_mod, 1); % creates the cell array where to store the models
    fn2load = sprintf('%sProject1917_movie_part%d_24Hz.mp4',stim_dir,ipart);
    % load video header, and reset and prepare optical flow vector
    videoHeader = VideoReader(fn2load);
    iframe=0;
    while hasFrame(videoHeader) %loops over all the frames of the movie part
        
        iframe=iframe+1;
        clc
        fprintf("%s frame num %d of part %d \n", string(datetime('now')), iframe, ipart)
        frame=readFrame(videoHeader);
        imshow(frame)
        count_row = 0;
        count_col = 0;
        isquare = 0;
        for i_row = onsets(1,:)
            
            for i_col = onsets(2,:)
                isquare = isquare +1;
                img_sq = frame(i_row: i_row+square_size(1)-1,i_col:i_col+square_size(2)-1, :);
                % imshow(img_sq)
                [temp_mod{1}(isquare, :), temp_mod{2}(isquare, :), temp_mod{3}(isquare, :), temp_mod{4}(isquare, :)] = LGNstatistics(img_sq, viewing_dist); % res_mat(i_sq, :) = % it outputs a 1x4 vec of 1x3 arrays
            end %for i_row = 1: n_squares
        end %for i_col = 1: n_squares

        for imod= 1:n_mod
            tot_mod{imod}(:,iframe) =temp_mod{imod}(:);
        end
    end
    beep %to tell you that one part is over
    disp(['part ' num2str(ipart) ' finished'])
    for imod = 1:n_mod
        vecrep = tot_mod{imod};
        vecrep(isnan(vecrep)) = 0; % sometimes beta is missing, we'll substitute it with 0s
        path2save=sprintf('%sProject1917_%s_run%02d_movie24Hz.mat',out_dir,models(imod),ipart);
        S = [];
        S.vecrep = vecrep;
        S.tVid = tVid;
        S.fsVid = fsVid;
        save(path2save,'-fromstruct', S,'-v7.3')
    end % for imod = 1:n_mod
    % clear tVid fsVid vecrep tot_mod
end
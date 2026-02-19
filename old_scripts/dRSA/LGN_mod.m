addpath(genpath("/Volumes/TIZIANO/Project1917/code/LGNstatistics"))
%%
path2img = "~/Desktop/leonia.jpg"
img = imread(path2img);
%%
img_len = size(img)
n_squares = 3; % height, width
square_size = [floor(img_len(1)/n_squares), floor(img_len(2)/ n_squares)]; % floor to make sure we don't overshoot with the indexing
%%

onsets = zeros(2, n_squares); % onset rows on the 1st row, onset cols on the 2nd row
onsets(:, 1) = 1;  % initializing the onsets
for i = 2:n_squares
    onsets(:, i) = onsets(:, i-1) + square_size'; % adding square_size as a column vector on the rows of onsets
end % for i = 1:n_squares

%%
% in order the columns are : CE (contrast energy), SC (spatial coherence), beta (Weibull paramt), gamma (Weibull paramt)
SC_mat = zeros(n_squares^2, 3);  
CE_mat = zeros(n_squares^2, 3);
beta_mat = zeros(n_squares^2, 3);
gamma_mat = zeros(n_squares^2, 3);
viewing_dist = .75;
size_of_movie = .222; % in mt
num_of_pix_w = 1280;
movie_size_factor = .7;
num_of_pix_w_presented = num_of_pix_w*movie_size_factor/n_squares;
dot_pitch = size_of_movie/num_of_pix_w_presented ;


%%

count_row = 0;
count_col = 0;
count = 0

for i_row = onsets(1,:)
    disp(i_row)
    for i_col = onsets(2,:)
        count = count +1;
        img_sq = img(i_row: i_row+square_size(1)-1,i_col:i_col+square_size(2)-1, :);
        % imshow(img_sq)
        [CE_mat(count, :), SC_mat(count, :), beta_mat(count, :), gamma_mat(count, :)] = LGNstatistics(img_sq, viewing_dist,dot_pitch); % res_mat(i_sq, :) = % it outputs a 1x4 vec of 1x3 arrays
    end %for i_row = 1: n_squares
end %for i_col = 1: n_squares

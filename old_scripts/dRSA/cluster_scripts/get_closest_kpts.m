function [curr_kpts, idx] = get_closest_kpts(mod_frame, x, y, crit)
    curr_frame = reshape(mod_frame, 17,2,5);
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
    curr_kpts = curr_frame(:,:,idx);

end % EOF

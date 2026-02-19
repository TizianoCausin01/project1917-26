%layers = 3:11;
layers = 12 % because the logits where computed separately from the other features
mod_dir = "/Volumes/TIZIANO/models/";
for irun = 1:3
    t_stamps_dir = sprintf("%sProject1917_OFdir_run%02d_movie24Hz.mat", mod_dir, irun);
    load(t_stamps_dir, "fsVid", "tVid")
    for ilayer = layers
        %mod2load=sprintf("%sProject1917_ViT_run%02d.h5", mod_dir, irun)
        mod2load=sprintf("%sProject1917_ViT_logits_run%02d.h5", mod_dir, irun) % because the logits where computed separately from the other features
        layer2load = sprintf("/%d", ilayer); %it need to be like this /conv_layer1
        disp(layer2load)
        vecrep = h5read(mod2load, layer2load);
        vecrep(:,1:floor(5*fsVid)) =[];
        if size(vecrep,2)~=size(tVid)
            error(["error: the sizes of the timestamps and of the model don't coincide for this run"])
        end

        fn2save =sprintf("%sProject1917_ViT_layer%d_run%02d_movie24Hz.mat", mod_dir, ilayer, irun);
        save(fn2save, 'vecrep', 'fsVid', 'tVid')
    end
end


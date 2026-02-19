layers = ["layer1", "layer2", "layer3", "layer4", "fc"];
mod_dir = "/Volumes/TIZIANO/models/";
for irun = 1:3
    t_stamps_dir = sprintf("%sProject1917_OFdir_run%02d_movie24Hz.mat", mod_dir, irun);
    load(t_stamps_dir, "fsVid", "tVid")
    for ilayer = 1:length(layers)
        mod2load=sprintf("%sProject1917_resnet18_run%02d.h5", mod_dir, irun)
        layer2load = sprintf("/%s", layers(ilayer)); %it need to be like this /conv_layer1
        vecrep = h5read(mod2load, layer2load);

        vecrep(:,1:floor(5*fsVid)) =[];
        if size(vecrep,2)~=size(tVid)
            error(["error: the sizes of the timestamps and of the model don't coincide for this run"])
        end

        fn2save =sprintf("%sProject1917_resnet18_%s_run%02d_movie24Hz.mat", mod_dir, layers(ilayer), irun);
        save(fn2save, 'vecrep', 'fsVid', 'tVid')
    end
end


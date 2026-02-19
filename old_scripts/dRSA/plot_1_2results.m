% "alexnet_fc_layer5"
% 

parms=[]
parms.subjects = 3:10;
parms.fsNew = 500;
results_dir = sprintf("/Volumes/TIZIANO/results/corr/%dHz_180stim_10sec_100iter_0MNN", parms.fsNew);
% parms.modelnames = {"OFdir", "dg_map", "dg_map_KLD","alexnet_conv_layer1", "alexnet_conv_layer4", "alexnet_conv_layer7", "alexnet_conv_layer9", "alexnet_conv_layer11", "alexnet_fc_layer2", "alexnet_fc_layer5", "gbvs_map", "gbvs_map_KLD", "pixelwise","OFmag"};
%parms.modelnames = {"real_alexnet_real_conv_layer1", "real_alexnet_real_conv_layer4", "real_alexnet_real_conv_layer7", "real_alexnet_real_conv_layer9", "real_alexnet_real_conv_layer11", "real_alexnet_real_fc_layer2", "real_alexnet_real_fc_layer5"};
parms.modelnames = {"resnet18_layer1","resnet18_layer2","resnet18_layer3","resnet18_layer4", "resnet18_fc"}
iroi = "par";
upper_y_lim = 0.03;
plot_resultss(results_dir,iroi, parms, upper_y_lim, 1)
function plot_resultss(results_directory,iroi,parms, upper_y_lim, save)
% inputs:
% - results_directory
% - imod
% - parms : .subjects, .fsNew
% - image_dir (where to store the image) ??


figure
freq = round(parms.fsNew);

    count=0;
    for isub=parms.subjects
        count=count+1;
        % if dist == 0
        fn2load = sprintf('%s/sub%03d/dRSA_corr_sub%03d_%s_rep1_2_%dHz.mat', results_directory, isub, isub,iroi,freq);
        disp(fn2load)
        %fn2load = sprintf('/Users/tizianocausin/Desktop/dataRepository/RepDondersInternship/results_sanity_check/%02d_23freq_simulation_rep12_%d_iterations_50Hz_no_smooth',isub,100);
        % else
        %     fn2load = sprintf('%s%cdRSA_dist_sub%03d_%s_rep%d_%dHz.mat', results_directory, filesep, isub,imod,irep,freq);
        % end
        load(fn2load)
        storeMod(count,:)=dRSA;
        dRSApeak1(isub)=latencytime(dRSA==max(dRSA));
    end
    avgMod=mean(storeMod);
    sem_mod=std(storeMod)/sqrt(size(parms.subjects,2));
    peakLatency=latencytime(avgMod==max(avgMod));
[hl1 hp1] = boundedline(latencytime,avgMod,sem_mod, 'alpha','transparency',.3);
set(hl1,'LineWidth',5);
set(hl1,'color',[0,0.4,.7])
set(hp1,'FaceColor',[0,0.4,.7])
yticks(0:.01:.2)
ylim([0 upper_y_lim])
xlim([-5 5])
xline([0 0]) %,'w')
title(sprintf("%s_rep1_2", iroi))
ax = gca;
% ax.XColor = 'w';
% ax.YColor = 'w';
% ax.Color = 'k';  % axes background
% set(gcf, 'InvertHardcopy', 'off')
annotation('textbox',[0.15, 0.8, 0.1, 0.1], 'String',['rep1 = ' num2str(peakLatency)])
hold off
if save ==1
path2save = "/Volumes/TIZIANO/figures_1917";
fig2save = sprintf("%s/%s_%dHz_%s.png", path2save, "1_2", parms.fsNew, iroi)
exportgraphics(gcf, fig2save, 'BackgroundColor', 'white')
end
end %EOF

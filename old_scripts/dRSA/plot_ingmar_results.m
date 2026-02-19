% "alexnet_fc_layer5"
%
results_dir = "/Volumes/TIZIANO/results_Ingmar/50Hz_180stim_10sec_1000iter_0MNN";
parms=[]
parms.subjects = [3:15];
parms.fsNew = 24;
upper_y_lim = 0.006;
rois = ["allsens"]
mods = ["pixelwise", "OFdir", "OFmag"] %, "dg_map"]
imod_tit = ["pixelwise", "OFdir", "OFmag"] %, "Deep-Gaze II"]
sm = {}
roi_counter = 0
for iroi = rois
    roi_counter = roi_counter+1;
    counter = 0
    for imod = mods
        counter = counter+1;
        plot_resultss(results_dir, imod,imod_tit(counter), iroi, parms, upper_y_lim)
    end
end
function plot_resultss(results_directory,imod,imod_tit, iroi,parms, upper_y_lim)
% inputs:
% - results_directory
% - imod
% - parms : .subjects, .fsNew
% - image_dir (where to store the image) ??
figure
freq = round(parms.fsNew);
for irep=[1 2]
    count=0;
    for isub=parms.subjects
        count=count+1;
        fn2load = sprintf('%s/sub%03d/dRSA_%s_%s_%dHz_rep%d_gazedep1_gazerad250.mat', results_directory, isub, iroi, imod, freq, irep);
        disp(fn2load)
        load(fn2load)
        storeMod{irep}(count,:)=dRSA;
        dRSApeak1{irep}(isub)=latencytime(dRSA==max(dRSA));
    end
    avgMod{irep}=mean(storeMod{irep});
    sem_mod{irep}=std(storeMod{irep})/sqrt(size(parms.subjects,2));
    peakLatency{irep}=latencytime(avgMod{irep}==max(avgMod{irep}));
end
[hl1 hp1] = boundedline(latencytime,avgMod{1},sem_mod{1}, 'alpha','transparency',.3);
set(hl1,'LineWidth',5);
set(hl1,'color',[0,0.4,.7])
set(hp1,'FaceColor',[0,0.4,.7])
hold on
[hl2, hp2] = boundedline(latencytime,avgMod{2},sem_mod{2},'alpha','transparency',.2);
set(hl2,'LineWidth',5);
set(hl2, 'color',[.6,0,.8]);
set(hp2, 'FaceColor',[.6,0,.8]);
xlabel("Lag[sec]", 'FontName', 'Arial', 'FontSize', 24, 'FontWeight', 'normal')
% ylabel( "Spearman's rho", 'FontName', 'Arial', 'FontSize', 24, 'FontWeight', 'normal')
xticks(-5:5)
yticks(-.002:.001:1)
ylim([-.002 upper_y_lim])
xlim([-5 5])
xline([0 0]) %,'w')
if strcmp(iroi, "occ")
    iroi_complete = "occipital"
elseif strcmp(iroi, "tem")
    iroi_complete = "temporal"
elseif strcmp(iroi, "par")
    iroi_complete = "parietal"
elseif strcmp(iroi, "fro")
    iroi_complete = "frontal"
elseif strcmp(iroi, "occpar")
    iroi_complete = "occipito-parietal"
elseif strcmp(iroi, "allsens")
    iroi_complete = "all sensors"
end % if strcmp(iroi, "occ"):
title([imod_tit, iroi_complete], 'FontName', 'Arial', 'FontSize', 26, 'FontWeight', 'normal')
ax = gca;
ax.YAxis.FontSize = 24;
ax.XAxis.FontSize = 24;
% ax.XColor = 'w';
% ax.YColor = 'w';
% ax.Color = 'k';  % axes background
% set(gcf, 'InvertHardcopy', 'off')
annotation('textbox',[0.15, 0.7, 0.1, 0.1], 'String',{['peak latency:'], ['rep1 = ' num2str(peakLatency{1})],['rep2 = ', num2str(peakLatency{2})]}, 'FontSize',20, 'EdgeColor', 'none')
hold off
path2save = "/Users/tizianocausin/Downloads"; %/Users/tizianocausin/Library/CloudStorage/OneDrive-SISSA/SIP/figures_SIP/figures_caos_poster";
fig2save = sprintf("%s/ingmar_%s_%s_white.svg", path2save, imod, iroi)
% exportgraphics(gcf, fig2save, 'BackgroundColor', 'white')
print(gcf, '-dsvg', '-r600', '-painters', fig2save) % '-depsc'
% saveas(gcf,fig2save)
end %EOF

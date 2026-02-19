path2sim = "/Volumes/TIZIANO/results/simulations";
parms.modelnames = {"alexnet_conv_layer1", "alexnet_conv_layer4", "alexnet_conv_layer7", "alexnet_conv_layer9", "alexnet_conv_layer11", "alexnet_fc_layer2", "alexnet_fc_layer5", "resnet18_layer1", "resnet18_layer2", "resnet18_layer3", "resnet18_layer4", "resnet18_fc", "real_alexnet_real_conv_layer1", "real_alexnet_real_conv_layer4", "real_alexnet_real_conv_layer7", "real_alexnet_real_conv_layer9", "real_alexnet_real_conv_layer11", "real_alexnet_real_fc_layer2", "real_alexnet_real_fc_layer5"};
freq = 50;
path2save = "/Volumes/TIZIANO/Project1917/figures";
for i = 1:length(parms.modelnames)
plot_simulations(path2sim, parms.modelnames{i}, freq, 1, path2save)
end

function plot_simulations(simulations_dir,imod,freq, save, path2save)
% inputs:
% - results_directory
% - imod
% - parms : .subjects, .fsNew
% - image_dir (where to store the image) ??
if nargin<4
    save = 0;
end

figure
fn2load = sprintf('%s/simulation_%s_%dHz.mat', simulations_dir,imod,freq);
disp(fn2load)
load(fn2load)

sem_mod = standard_err_simulation;

[hl1 hp1] = boundedline(latencytime,dRSA,standard_err_simulation, 'alpha','transparency',.3);
set(hl1,'LineWidth',2);
set(hl1,'color',[0,0.4,.7])
set(hp1,'FaceColor',[0,0.4,.7])
yticks(0.2:.1:1)
ylim([.2 1])
xlim([-5 5])
xline([0 0]) %,'w')
title([imod])
xlabel("lag[sec]")
ylabel("Spearman's Rho")
ax = gca;

hold off
if save==1

fig2save = sprintf("%s/%s_simulation.png", path2save, imod)
exportgraphics(gcf, fig2save, 'BackgroundColor', 'white')
end
end %EOF



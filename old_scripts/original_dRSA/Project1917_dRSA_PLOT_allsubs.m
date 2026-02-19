function Project1917_dRSA_PLOT_allsubs(cfg)

rootdir = '\\cimec-storage5.unitn.it\MORWUR\Projects\INGMAR\Project1917';
addpath(genpath(rootdir));
addpath(genpath('\\cimec-storage5.unitn.it\MORWUR\Projects\INGMAR\ProjectMovie\code\dRSA\boundedline-toolbox'));

if cfg.similarity == 0
    simstring = 'corr';
elseif cfg.similarity == 1
    simstring = ['pcr_' num2str(cfg.nPCRcomps) 'comps'];
end

indirMEG234 = sprintf('%s%sresults%sdRSA%s%s_%dHz_%dstim_%dsec_%diterations_%dMNN_strictICA',rootdir,filesep,filesep,filesep,simstring,cfg.fsNew,90,cfg.stimlen,cfg.iterations,cfg.MNN);
indirMEG56 = sprintf('%s%sresults%sdRSA%s%s_%dHz_%dstim_%dsec_%diterations_%dMNN_strictICA',rootdir,filesep,filesep,filesep,simstring,cfg.fsNew,50,cfg.stimlen,cfg.iterations,cfg.MNN);
indirSIM = sprintf('%s%sresults%sdRSA%ssim_%s_%dHz_%dstim_%dsec_%diterations',rootdir,filesep,filesep,filesep,simstring,cfg.fsNew,50,cfg.stimlen,cfg.iterationsSIM);

tRange=cfg.maxlatency*cfg.fsNew;
binVec2 = round([1  tRange/2+1 tRange+1 tRange*1.5+1 tRange*2+1]);

lineColors = brewermap(6,'YlGnBu');% or 'RdPu'
lineColors(1,:) = [];% first is too bright

%% load data
dRSAsim = zeros(length(cfg.models2test),length(cfg.models2sim),tRange*2+1);
dRSAreal = zeros(length(cfg.subjects),length(cfg.ROI),length(cfg.models2test),length(cfg.conditions),tRange*2+1);
for itest = 1:length(cfg.models2test)
    
    for iROI = 1:length(cfg.ROI)
    
        for isub = 1:length(cfg.subjects)

            if isub < 4% only free viewing

                % load real data
                fn = sprintf('%s%cSUB%02d_%s_%s', indirMEG234, filesep, cfg.subjects(isub), cfg.ROInames{iROI}, cfg.modelnames{itest});
                load(fn,'dRSA','latencytime');
                time = latencytime;
                
                dRSAreal(isub,iROI,itest,1,:) = dRSA;

            else

                for icon = 1:length(cfg.conditions)

                    % load real data
                    fn = sprintf('%s%cSUB%02d_%s_%s_%s', indirMEG56, filesep, cfg.subjects(isub), cfg.ROInames{iROI}, cfg.modelnames{itest}, cfg.conditions{icon});
                    load(fn,'dRSA','latencytime');
             
                    dRSAreal(isub,iROI,itest,icon,:) = dRSA;

                end

            end
            
        end
    
    end
    
    % load simulations
    for isim = 1:length(cfg.models2sim)
        
        fn = sprintf('%s%cSIM_%s_TEST_%s', indirSIM, filesep, cfg.modelnames{isim}, cfg.modelnames{itest});
        load(fn,'dRSA');
        dRSAsim(itest,isim,:) = dRSA;
        
    end

end

dRSAreal(dRSAreal == 0) = nan;

% if you want to plot only a single subject
dRSAreal(2:end,:,:,:,:) = nan;

% random permutations
dRSAperm_all = [];
fn = sprintf('%s%cALLSUB_allsensors_ALLMOD_0200perms', indirMEG234, filesep);
load(fn,'dRSAperm');
% dRSAperm(:,:,:,[1:50 552:end]) = [];
dRSAperm = sort(dRSAperm);
permthresh = round(size(dRSAperm,1)*(cfg.pthresh/cfg.side));
dRSAperm_all(1,:,1:3,:,:) = cat(1,mean(dRSAperm), dRSAperm([permthresh end-permthresh],:,:,:));
fn = sprintf('%s%cALLSUB_allsensors_ALLMOD_0200perms', indirMEG56, filesep);
load(fn,'dRSAperm');
dRSAperm = sort(dRSAperm);
permthresh = round(size(dRSAperm,1)*(cfg.pthresh/cfg.side));
dRSAperm_all(1,:,4:5,:,:) = cat(1,mean(dRSAperm), dRSAperm([permthresh end-permthresh],:,:,:));
fn = sprintf('%s%cALLSUB_occipito-parietal_ALLMOD_0200perms', indirMEG234, filesep);
load(fn,'dRSAperm');
% dRSAperm(:,:,:,[1:50 552:end]) = [];
dRSAperm = sort(dRSAperm);
permthresh = round(size(dRSAperm,1)*(cfg.pthresh/cfg.side));
dRSAperm_all(2,:,1:3,:,:) = cat(1,mean(dRSAperm), dRSAperm([permthresh end-permthresh],:,:,:));
fn = sprintf('%s%cALLSUB_occipito-parietal_ALLMOD_0200perms', indirMEG56, filesep);
load(fn,'dRSAperm');
dRSAperm = sort(dRSAperm);
permthresh = round(size(dRSAperm,1)*(cfg.pthresh/cfg.side));
dRSAperm_all(2,:,4:5,:,:) = cat(1,mean(dRSAperm), dRSAperm([permthresh end-permthresh],:,:,:));

% sem for free view based on 5 subjects, for fixation and combined based on 2 subjects
dRSAsem(:,:,1,:) = squeeze(std(squeeze(dRSAreal(:,:,:,1,:)),'omitnan'))./sqrt(size(dRSAreal,1));
dRSAsem(:,:,2:3,:) = squeeze(std(dRSAreal(:,:,:,2:3,:),'omitnan'))./sqrt(2);

close(figure(1));

figure(1);
set(gcf,'color','w');
set(gcf, 'Units', 'centimeters');
set(gcf, 'Position',  [1 1 40 30]);

ylimit = [0.012 0.012 0.012];
ylimitavg = [0.005 0.005 0.005];
for itest = 1:length(cfg.models2test)% models
    
    % subavg - allsens - free view (5 sub), fixation (2 sub) and combined (2 sub)
    subplot(5,6,itest);
    
    hold on
    h = [];
    for icon = 1:3
        SEM2plot = repmat(squeeze(dRSAsem(1,itest,icon,:)),[1 2]);
        if icon == 1
            boundedline(time,squeeze(mean(dRSAreal(:,1,itest,icon,:),'omitnan')), SEM2plot , 'alpha','cmap', lineColors(icon+2,:));
        end
        h(icon) = plot(time,squeeze(mean(dRSAreal(:,1,itest,icon,:),'omitnan')),'color',lineColors(icon+2,:),'lineWidth',1.5);
    end
    plot([latencytime(1) latencytime(end)],[0 0],'--k');
    plot([0 0],[-1 1],'--k');
    set(gca,'xlim',[latencytime(1) latencytime(end)]);
    set(gca,'ylim',[-ylimitavg(itest) ylimitavg(itest)*1]);
    set(gca,'xtick',latencytime(binVec2));
    set(gca,'Fontsize',cfg.fs,'FontName','Helvetica');
    hold off
    
    if itest == 1
        ylabel('dRSA [Rho]','Fontsize',cfg.fs,'FontName','Helvetica');
        legend(h,'free view (n=5)','fix (n=2)','combi (n=2)'); 
    end
    
    title([cfg.modelnames{itest} ' - all sens - subavg'],'Fontsize',cfg.fs,'FontName','Helvetica','FontWeight','normal');
    
    % subavg - parieto-occipital - free view (5 sub), fixation (2 sub) and combined (2 sub)
    subplot(5,6,itest+3);
    
    hold on
    for icon = 1:3
        SEM2plot = repmat(squeeze(dRSAsem(2,itest,icon,:)),[1 2]);
        if icon == 1
            boundedline(time,squeeze(mean(dRSAreal(:,2,itest,icon,:),'omitnan')), SEM2plot , 'alpha','cmap', lineColors(icon+2,:));
        end
        h(icon) = plot(time,squeeze(mean(dRSAreal(:,2,itest,icon,:),'omitnan')),'color',lineColors(icon+2,:),'lineWidth',1.5);
    end
    plot([latencytime(1) latencytime(end)],[0 0],'--k');
    plot([0 0],[-1 1],'--k');
    set(gca,'xlim',[latencytime(1) latencytime(end)]);
    set(gca,'ylim',[-ylimitavg(itest) ylimitavg(itest)*1]);
    set(gca,'xtick',latencytime(binVec2));
    set(gca,'Fontsize',cfg.fs,'FontName','Helvetica');
    hold off
    
    title([cfg.modelnames{itest} ' - occ par - subavg'],'Fontsize',cfg.fs,'FontName','Helvetica','FontWeight','normal');
    
    % random permutations - avg + 95% CI - all sensors - free view
    subplot(5,6,itest+6);
    
    hold on
    for isub = 1:5
        h(isub) = plot(time,squeeze(dRSAperm_all(1,1,isub,itest,:)),'color',lineColors(isub,:),'lineWidth',1.5);
        plot(time,squeeze(dRSAperm_all(1,2:3,isub,itest,:)),'color',lineColors(isub,:),'lineWidth',1);
    end
    
    plot([latencytime(1) latencytime(end)],[0 0],'--k');
    plot([0 0],[-1 1],'--k');
    set(gca,'xlim',[latencytime(1) latencytime(end)]);
    set(gca,'ylim',[-ylimit(itest) ylimit(itest)]);
    set(gca,'xtick',latencytime(binVec2));
    set(gca,'Fontsize',cfg.fs,'FontName','Helvetica');
    hold off
    
    if itest == 1
        ylabel('dRSA [Rho]','Fontsize',cfg.fs,'FontName','Helvetica');
        legend(h,'sub02','sub03','sub04','sub05','sub06'); 
    end
    
    title('free view - rand perm','Fontsize',cfg.fs,'FontName','Helvetica','FontWeight','normal');
    
    % random permutations - avg + 95% CI - all sensors - free view
    subplot(5,6,itest+9);   
    
    hold on
    for isub = 1:5
        plot(time,squeeze(dRSAperm_all(2,1,isub,itest,:)),'color',lineColors(isub,:),'lineWidth',1.5);
        plot(time,squeeze(dRSAperm_all(2,2:3,isub,itest,:)),'color',lineColors(isub,:),'lineWidth',1);
    end
    
    plot([latencytime(1) latencytime(end)],[0 0],'--k');
    plot([0 0],[-1 1],'--k');
    set(gca,'xlim',[latencytime(1) latencytime(end)]);
    set(gca,'ylim',[-ylimit(itest) ylimit(itest)]);
    set(gca,'xtick',latencytime(binVec2));
    set(gca,'Fontsize',cfg.fs,'FontName','Helvetica');
    hold off
    
    title('free view - rand perm','Fontsize',cfg.fs,'FontName','Helvetica','FontWeight','normal');
         
    % per subject - all sensors - free view
    subplot(5,6,itest+12);
    
    hold on
    for isub = 1:length(cfg.subjects)
        plot(latencytime,squeeze(dRSAreal(isub,1,itest,1,:)),'color',lineColors(isub,:),'lineWidth',1.5);
    end
    plot([latencytime(1) latencytime(end)],[0 0],'--k');
    plot([0 0],[-1 1],'--k');
    set(gca,'xlim',[latencytime(1) latencytime(end)]);
    set(gca,'ylim',[-ylimit(itest) ylimit(itest)]);
    set(gca,'xtick',latencytime(binVec2));
    set(gca,'Fontsize',cfg.fs,'FontName','Helvetica');
    hold off
    
    if itest == 1
        ylabel('dRSA [Rho]','Fontsize',cfg.fs,'FontName','Helvetica');
    end
    
    title('free view','Fontsize',cfg.fs,'FontName','Helvetica','FontWeight','normal');
    
    if itest == 1
       legend('sub02','sub03','sub04','sub05','sub06'); 
    end

    % per subject - occipito-parietal - free view
    subplot(5,6,itest+15);
    
    hold on
    for isub = 1:length(cfg.subjects)
        plot(latencytime,squeeze(dRSAreal(isub,2,itest,1,:)),'color',lineColors(isub,:),'lineWidth',1.5);
    end
    plot([latencytime(1) latencytime(end)],[0 0],'--k');
    plot([0 0],[-1 1],'--k');
    set(gca,'xlim',[latencytime(1) latencytime(end)]);
    set(gca,'ylim',[-ylimit(itest) ylimit(itest)]);
    set(gca,'xtick',latencytime(binVec2));
    set(gca,'Fontsize',cfg.fs,'FontName','Helvetica');
    hold off
    
    title('free view','Fontsize',cfg.fs,'FontName','Helvetica','FontWeight','normal');
    
    % sub 5 and 6 - all sensors - fixation and combined
    subplot(5,6,itest+18);
    
    hold on
    for isub = 4:5
        plot(latencytime,squeeze(dRSAreal(isub,1,itest,2,:)),'color',lineColors(isub,:),'lineWidth',1.5);
    end
    for isub = 4:5
        plot(latencytime,squeeze(dRSAreal(isub,1,itest,3,:)),'--','color',lineColors(isub,:),'lineWidth',1.5);
    end
    plot([latencytime(1) latencytime(end)],[0 0],'--k');
    plot([0 0],[-1 1],'--k');
    set(gca,'xlim',[latencytime(1) latencytime(end)]);
    set(gca,'ylim',[-ylimit(itest) ylimit(itest)]);
    set(gca,'xtick',latencytime(binVec2));
    set(gca,'Fontsize',cfg.fs,'FontName','Helvetica');
    hold off
    
    if itest == 1
        ylabel('dRSA [Rho]','Fontsize',cfg.fs,'FontName','Helvetica');
    end
    
    title('fix / combi','Fontsize',cfg.fs,'FontName','Helvetica','FontWeight','normal');
    
    if itest == 1
       legend('sub05 - fix','sub06 - fix','sub05 - combi','sub06 - combi'); 
    end

    % sub 5 and 6 - occipito-parietal - fixation and combined
    subplot(5,6,itest+21);
    
    hold on
    for isub = 4:5
        plot(latencytime,squeeze(dRSAreal(isub,2,itest,2,:)),'color',lineColors(isub,:),'lineWidth',1.5);
    end
    for isub = 4:5
        plot(latencytime,squeeze(dRSAreal(isub,2,itest,3,:)),'--','color',lineColors(isub,:),'lineWidth',1.5);
    end
    plot([latencytime(1) latencytime(end)],[0 0],'--k');
    plot([0 0],[-1 1],'--k');
    set(gca,'xlim',[latencytime(1) latencytime(end)]);
    set(gca,'ylim',[-ylimit(itest) ylimit(itest)]);
    set(gca,'xtick',latencytime(binVec2));
    set(gca,'Fontsize',cfg.fs,'FontName','Helvetica');
    hold off
    
    title('fix / combi','Fontsize',cfg.fs,'FontName','Helvetica','FontWeight','normal');
    
    % simulations
    subplot(5,6,itest+24)
    
    hold on
    for isim = 1:length(cfg.models2sim)
        plot(time,squeeze(dRSAsim(itest,isim,:)),'color',lineColors(isim+1,:),'lineWidth',1.5);
    end
    plot([latencytime(1) latencytime(end)],[0 0],'--k');
    plot([0 0],[-1 1],'--k');
    set(gca,'xlim',[latencytime(1) latencytime(end)]);
    set(gca,'ylim',[-.1 1]);
    set(gca,'xtick',latencytime(binVec2));
    set(gca,'Fontsize',8,'FontName','Helvetica');
    hold off
    
    if itest == 1
        ylabel('dRSA [Rho]','Fontsize',cfg.fs,'FontName','Helvetica');        
    end
    if itest == 3
       legend(cfg.modelnames); 
    end
    
    xlabel('lag [sec]','Fontsize',cfg.fs,'FontName','Helvetica');
    
end
    


% cd('G:\My Drive\Active projects\Unpredict\figures');
% print -depsc -r600 ActionPrediction_Figure2a_ROIlineplots.eps

% set(gcf,'renderer','Painters')
% print -depsc -tiff -r600 -painters Unpredict_ROIlagplots_normal.eps


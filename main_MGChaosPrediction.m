%% Mackey-Glass Chaotic Time Series Prediction, Experiment 5 in the Patch Learning paper

%% Prof. Dongrui Wu (drwu@hust.edu.cn)
%% Huazhong University of Science and Technology, Wuhan, China

clc; clearvars; close all; rng('default'); warning off all;
nMFs=2; nPatches=2; nEpoches=100;

%% Load the Mackey-Glass Chaotic Time Series data
load mgdata.dat
time = mgdata(:, 1); x = mgdata(:, 2);

trainData = [x(1:600) x(7:606) x(13:612) x(19:618)];
testData = [x(601:1100) x(607:1106) x(613:1112) x(619:1118)];
X=trainData(:,1:end-1); y=trainData(:,end);

RMSEs=zeros(1,nPatches);
for i=1:nPatches+1
    % Patch learning
    YPL=PL_ANFIS(X,y,testData(:,1:end-1),i-1,nMFs,nEpoches);
    error1=testData(:,end)-YPL(:,end);
    RMSEs(1,i)=sqrt(mean(error1.^2));
    % Bagging
    YB2=Bagging_ANFIS(X,y,testData(:,1:end-1),i,nMFs,nEpoches);
    error2=testData(:,end)-YB2; RMSEs(2,i)=sqrt(mean(error2.^2));
    % LSBoost
    mdl=fitrensemble(X,y,'NumLearningCycles',i);
    YB3=predict(mdl,testData(:,1:end-1));
    error3=testData(:,end)-YB3; RMSEs(3,i)=sqrt(mean(error3.^2))
    
    figure;
    set(gcf,'DefaulttextFontName','times new roman','DefaultaxesFontName','times new roman','defaultaxesfontsize',8,...
        'defaulttextfontsize',9,'Position',[200 100 300 250]);
    subplot(211);
    plot(testData(:,end),'k-','linewidth',1); hold on; plot(YPL(:,end),'b--','linewidth',1);
    plot(YB2,'r-.','linewidth',1);    plot(YB3,'g:','linewidth',1);
    xlabel('$t$','interpreter','latex'); ylabel('$x(t)$','interpreter','latex');
    set(gca,'ylim',[.4 1.8]); 
    h=legend('True','PL','Bagging','LSBoost','location','east');
    set(h,'fontsize',8);
    title(['Predictions; $L=' num2str(i-1) '$ in PL'],'interpreter','latex');
    
    subplot(212);
    plot(error1,'b--','linewidth',.8); hold on; plot(error2,'r-.','linewidth',.8); plot(error3,'g:','linewidth',.8);
    xlabel('$t$','interpreter','latex');   ylabel('Error','interpreter','latex');
    set(gca,'ylim',[-.35 .2]);
    title(['Prediction errors; $L=' num2str(i-1) '$ in PL'],'interpreter','latex');
    
    h=legend(['PL, RMSE=' num2str(sqrt(mean(error1.^2)),'%.4f') ', $\ell$=' num2str(i^.25*sqrt(mean(error1.^2)),'%.4f')],...
        ['Bagging, RMSE=' num2str(sqrt(mean(error2.^2)),'%.4f')],...
        ['LSBoost, RMSE=' num2str(sqrt(mean(error3.^2)),'%.4f')],'location','south');
    set(h,'Box','off','position',get(h,'position')+[0 -0.02 0 0],'interpreter','latex');
end

%% PL, 3 patches
YPL=PL_ANFIS(X,y,testData(:,1:end-1),3,nMFs,nEpoches);
error1=testData(:,end)-YPL(:,end);
RMSEs(1,4)=sqrt(mean(error1.^2));

%% The loss
ell=RMSEs(1,:).*(1:size(RMSEs,2)).^(.25)
function [yTest,patches,pFLS]=PL_ANFIS(XTrain,yTrain,XTest,nPatches,numMFs,nEpoches)

%% XTrain: training input matrix, dimensionality [numExamples * numFeatures]
%% yTrain: training output vector, dimensionality [numExamples * 1]
%% XTest: test input matrix; same format as XTrain
%% nPatches: number of patches
%% numMFs: a [nPatches * 1] vector, indicating the number of MFs for each patch (if numFeatures>1, all features have the same number of MFs)
%% nEpoches: number of training epoches in ANFIS

%%  Note: sometimes the 'anfis' function returns errors, and hence this function cannot be finished. 
%% This is due to the limitation of anfis, not PL.
%% Dongrui Wu, drwu@hust.edu.cn

[N,M]=size(XTrain);
if length(numMFs)==1
    numMFs=numMFs*ones(nPatches+1,1);
end

if isempty(XTest); XTest=XTrain; end
yTest=nan(size(XTest,1),nPatches+2);
patches=[min(XTrain); max(XTrain)];

% Train the first ANFIS and obtain its prediction
genOpt = genfisOptions('GridPartition');
genOpt.NumMembershipFunctions = numMFs(1);
genOpt.InputMembershipFunctionType = 'trapmf';
opt = anfisOptions('InitialFIS',genfis(XTrain,yTrain,genOpt),'EpochNumber',nEpoches,'DisplayANFISInformation',0,...
    'DisplayErrorValues',0,'DisplayStepSize',0,'DisplayFinalResults',0);
pFLS(1).FLS=anfis([XTrain,yTrain],opt);
pFLS(1).min=min(XTrain); pFLS(1).max=max(XTrain);
yPred=evalfis(XTrain,pFLS(1).FLS);
yTest(:,1)=evalfis(XTest,pFLS(1).FLS);
idsAllPatch=false(size(yTrain));

% Construct the patches
for k=1:nPatches
    disp(['PL_ANFIS ' num2str(k)]);
    % Identify patches, by checking all possible partitions
    nFLSs=length(pFLS); % Number of fuzzy systems already constructed
    patches=zeros(2,M,nFLSs+1); % patch regions
    CPs=cell(1,M); % the change points for the next FLS
    nIntervals=zeros(1,M);
    for i=1:M
        MFparas=[pFLS(1).FLS.input(i).mf(:).params];
        CPs{i}=unique(MFparas([1:4:end 4:4:end]));
        nIntervals(i)=length(CPs{i})-1;
    end
    totalPatches=prod(nIntervals);
    SSEs=zeros(1,totalPatches);
    for i=1:totalPatches
        idsIntervals=num2ids(i,nIntervals);
        idsSamples=true(N,1);
        for j=1:M
            idsSamples=idsSamples & XTrain(:,j)>=CPs{j}(idsIntervals(j)) & XTrain(:,j)<=CPs{j}(idsIntervals(j)+1);
        end
        SSEs(i)=sum((yTrain(idsSamples)-yPred(idsSamples)).^2);
    end
    disp('SSEs of the patch candidates:');
    SSEs
    [maxSSE,idx]=max(SSEs);
    idsIntervals=num2ids(idx,nIntervals);
    for i=1:M
        patches(:,i,end)=[CPs{i}(idsIntervals(i)); CPs{i}(idsIntervals(i)+1)];
    end
    
    for i=1:nFLSs
        patches(:,:,i)=[max([pFLS(i).min; min(XTrain)]); min([pFLS(i).max; max(XTrain)])];
    end
    
    % Improve ANFIS
    i=nFLSs+1;
    idsPatch=true(N,1);
    for j=1:M % indices of points contained in this patch and having not been considered in previous patches
        idsPatch=idsPatch & (XTrain(:,j)>=patches(1,j,i)) & (XTrain(:,j)<=patches(2,j,i));
    end
    if sum(idsPatch)<=numMFs(i); return; end
    idsAllPatch=idsAllPatch | idsPatch;
    genOpt.NumMembershipFunctions = numMFs(i);
    FIS = genfis(XTrain(idsPatch,:),yTrain(idsPatch),genOpt);
    opt = anfisOptions('InitialFIS',FIS,'EpochNumber',nEpoches,'DisplayANFISInformation',0,...
        'DisplayErrorValues',0,'DisplayStepSize',0,'DisplayFinalResults',0);
    [pFLS(i).FLS,trainRMSE]=anfis([XTrain(idsPatch,:),yTrain(idsPatch)],opt);
    disp(['RMSE of the ' num2str(k) 'th patch (before and after): ' num2str(sqrt(maxSSE/sum(idsPatch))) ' ' num2str(min(trainRMSE))]);
    pFLS(i).min=patches(1,:,i);
    pFLS(i).max=patches(2,:,i);
    yPred(idsPatch)=yTrain(idsPatch);
    yTest(:,k+1)=patchFLS(XTest,pFLS);
end
% Update the default global model
yTest(:,end)=yTest(:,end-1);
if sum(~idsAllPatch)
    i=1;
    genOpt.NumMembershipFunctions = numMFs(i);
    FIS = genfis(XTrain(~idsAllPatch,:),yTrain(~idsAllPatch),genOpt);
    opt = anfisOptions('InitialFIS',FIS,'EpochNumber',nEpoches,'DisplayANFISInformation',0,...
        'DisplayErrorValues',0,'DisplayStepSize',0,'DisplayFinalResults',0);
    pFLS(i).FLS=anfis([XTrain(~idsAllPatch,:),yTrain(~idsAllPatch)],opt);
    yTest(:,end)=patchFLS(XTest,pFLS);
end
end

function ids=num2ids(n,nIntervals)
ids=zeros(1,length(nIntervals));
prods=[1 cumprod(nIntervals(end:-1:1))];
prev=0;
for i=1:length(nIntervals)
    ids(i)=floor((n-1-prev)/prods(end-i))+1;
    prev=prev+(ids(i)-1)*prods(end-i);
end
end

function yPred=patchFLS(X,pFLS)
% Compute the output of a fuzzy system based PL model
% warning off all;
[N,M]=size(X);
yPred=nan(N,1);
for i=1:length(pFLS) % The first FLS has the maximum input range, and the last the smallest. A latter FLS overwrites all previous ones
    ids=true(N,1);
    if i>1
        for j=1:M
            ids=ids & ((X(:,j)-pFLS(i).min(j)).*(X(:,j)-pFLS(i).max(j))<=0);
        end
    end
    if sum(ids)
        yPred(ids)=evalfis(X(ids,:),pFLS(i).FLS);
    end
end
end
function yTest=Bagging_ANFIS(XTrain,yTrain,XTest,nBoots,numMFs,nEpoches)

%% XTrain: training input matrix, dimensionality [numExamples * numFeatures]
%% yTrain: training output vector, dimensionality [numExamples * 1]
%% XTest: test input matrix; same format as XTrain
%% nBoots: number of boostrapes
%% numMFs: a [nPatches * 1] vector, indicating the number of MFs for each patch (if numFeatures>1, all features have the same number of MFs)
%% nEpoches: number of training epoches in ANFIS
%% Dongrui Wu, drwu@hust.edu.cn

[N,M]=size(XTrain);
if length(numMFs)==1
    numMFs=numMFs*ones(nBoots+1,1);
end

if isempty(XTest); XTest=XTrain; end
yTest=zeros(size(XTest,1),1);

%% Train nBoots ANFISs and obtain the prediction
genOpt = genfisOptions('GridPartition');
genOpt.NumMembershipFunctions = numMFs(1);
genOpt.InputMembershipFunctionType = 'trapmf';
for n=1:nBoots
    ids=datasample(1:N,N);
    opt = anfisOptions('InitialFIS',genfis(XTrain(ids,:),yTrain(ids),genOpt),'EpochNumber',nEpoches,...
        'DisplayANFISInformation',0,...
        'DisplayErrorValues',0,'DisplayStepSize',0,'DisplayFinalResults',0);
    FLS=anfis([XTrain(ids,:),yTrain(ids)],opt);
    yTest=yTest+evalfis(XTest,FLS)/nBoots;
end

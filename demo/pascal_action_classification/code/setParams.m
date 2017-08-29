function [param] = setParams(numTrain,dataDir,unaryFile,labelsFile)

param = struct;

% default values
param.temperature = 1;
param.edgeScaling = 1;
param.normNormalizeFeatures = 0;

param.numTrain = numTrain;
param.numNodes  = param.numTrain;
param.dataDir = dataDir;
%param.unaryFeatDir = unaryFeatDir;
param.numClasses = 10;
param.numStates = param.numClasses;

% set/load training sample indices
permutation = 1:param.numTrain;
% permutation = randperm(2800);
newTrainIdx = permutation(1:param.numTrain);
% newTrainIdx = 1:param.numTrain;
% load '../../infoFiles/newTrainIdx_n90.mat';
param.newTrainIdx = newTrainIdx;
% disp(newTrainIdx)
assert(numTrain<=length(newTrainIdx));


% Load labels
load(labelsFile);
load(unaryFile);
param.labels = num2cell(labelsTrain(newTrainIdx));
param.features = unaryFeatures;
param.patterns = num2cell(newTrainIdx);

param.c = 0.1;
param.e = 1e-3;

param.lossFn = @posEst_loss;
param.featureFn = @posEst_featuremap;
param.oracleFn = @posEst_maxOracle;
%param.constraintFn = @posEst_oracle;
param.exptOracleFn = @posEst_oracle;

param.featDim = 2404;
param.unarySize = 2404;
param.dimension = param.unarySize*param.numClasses;
param.numParts = param.numNodes;

end

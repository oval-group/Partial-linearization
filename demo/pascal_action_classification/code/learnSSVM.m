function learnSSVM(outputFile,lambda,method,temperature,rseed,gamma_1Meps)

addpath(genpath(pwd))
addpath '../../../solvers/';
addpath '../../../solvers/helpers';
sbin = 4;
dataDir = '../data/';

labelsFile = [dataDir 'labelsTrain.mat'];
unaryFile =  [dataDir 'unaryFeatures.mat'];
intialParametersDir =  '../infoFiles/initialParameters/';
  
numTrain = 2800;   % number of training samples

%rng(1,'twister');

param = setParams(numTrain,dataDir,unaryFile,labelsFile);
options = setOptions();

param.intialParametersDir = intialParametersDir;
param.intialWfile = [param.intialParametersDir 'W_init.mat'];
param.intialBeliefsDir = [param.intialParametersDir 'beliefs_init/'];
param.intialBeliefsFileformat = [param.intialBeliefsDir 'belPrev_%d.mat'];

param.temperature = temperature;
param.edgeScaling = 1;
param.normNormalizeFeatures = 0;

options.lambda = lambda;
options.outputFile = outputFile;
options.rand_seed = rseed;
options.gamma_1Meps = gamma_1Meps;

param.lambda = options.lambda;
param.debug = options.debug;

options.method = method;
param.method = method;

% param.model = svm_struct_learn(sprintf('- c %f -e %f -v 3 ',param.c,param.e), param);
param.isEG = 0;
if strcmp(method,'fw')
    [param.model, progress] = solverFW(param, options);
elseif strcmp(method,'bcfw')
    [param.model, progress] = solverBCFW(param, options);
elseif strcmp(method,'pl')
    [param.model, progress] = solverPL(param, options);
elseif strcmp(method,'bcpl')
    [param.model, progress] = solverBCPL(param, options);
elseif strcmp(method,'eg')
    param.isEG = 1;
    [param.model, progress] = solverPL(param, options);
elseif strcmp(method,'bceg')
    param.isEG = 1;
    [param.model, progress] = solverBCPL(param, options);
else
    error('Unrecognizable input for method');
end


end

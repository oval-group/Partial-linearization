
clear all; clc;

% addpath '/home/pritish/work/libraries/VLFeat/vlfeat-0.9.16/toolbox';
addpath '../../utils';
addpath '../../solvers/';
addpath '../../solvers/helpers';
addpath '../../utils/svm-struct-matlab-master';
addpath(genpath('../../utils/UGM'));
% vl_setup;
sbin = 4;
featDir = '../../data/parse/padHOG4';

structure = { [13], [14], [9 8 7], [10 11 12], [3 2 1], [4 5 6] };
% [13 -1; 14 13 ; 9 13; 8 9; 7 8; 10 13; 11 10; 12 11; 4 13; 5 4; 6 5; 3 13; 2 3; 1 2]; %Alternate representation.
numTrain = 3;
parm = setParams(numTrain,structure,featDir);
options = setOptions();

parm.temperature = 1e-1;
options.lambda = 100;
options.outputFile = '/home/pritish/work/work/NIPS2015/code/ParLinFW/results/outputFile.mat';

% parm.model = svm_struct_learn(sprintf('-c %f -e %f -v 3 ',parm.c,parm.e), parm);
% [parm.model, progress] = solverPLFW(parm, options);
[parm.model, progress] = solverFW(parm, options);
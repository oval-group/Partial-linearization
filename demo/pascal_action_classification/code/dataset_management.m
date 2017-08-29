%function newTrainIdx = dataset_management(numTrain,memCap)

clear all;clc;

dataDir = '../../data/parse/';
imDir = [dataDir 'people_all/'];

numTrain = 100;
memCap = 1.6;         % Memory cap in GB


imFiles = dir([imDir '*.jpg']);

imSizes = zeros(length(imFiles),2);
imMem = zeros(length(imFiles),1);

for i=1:length(imFiles)
    siz = size(imread([imDir imFiles(i).name]));
    imSizes(i,:) = siz(1:2);
    siz = prod(imSizes(i,:)/4);
    imMem(i) = prod([siz,siz,13])/2^27;     % estimated memory in GB
end

[trainSizDes_val,trainSizDes_idx] = sort(imMem(1:numTrain),'descend');
[testSizAsc_val,testSizAsc_idx] = sort(imMem(numTrain+1:end),'ascend');
testSizAsc_idx = testSizAsc_idx + numTrain;

newTrainIdx = trainSizDes_idx;
if ( nnz(trainSizDes_val>memCap) > nnz(testSizAsc_val>memCap) )
    fprintf('\nToo many images above the specified memory cap. You would need to increase the memory cap.\n');
end

newTrainIdx(trainSizDes_val>memCap) = testSizAsc_idx(1:nnz(trainSizDes_val>memCap));
newTrainIdx = sort(newTrainIdx,'ascend');

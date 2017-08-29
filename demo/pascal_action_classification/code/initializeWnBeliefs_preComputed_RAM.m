function [nodeBelAlphaPrev,edgeBelAlphaPrev,W,WMat] = initializeWnBeliefs_preComputed_RAM(param)

if exist(param.intialWfile,'file')
    load(param.intialWfile);
else
    error('\nFile containing initial parameter W not found.\n');
end

if param.normNormalizeFeatures
    from=1; to=param.numNodes*param.unarySize;
    W(from:to) = W(from:to) / param.maxUnaryFeatNorm;
    WMat(from:to,:) = WMat(from:to,:) / param.maxUnaryFeatNorm;
    
    from=to+1; to=to+param.numEdges*param.binarySize; assert(to==length(W));
    W(from:to) = W(from:to) / param.maxBinaryFeatNorm;
    WMat(from:to,:) = WMat(from:to,:) / param.maxBinaryFeatNorm;
end

nodeBelAlphaPrev = cell(1,param.numTrain);
edgeBelAlphaPrev = cell(1,param.numTrain);

if strcmp(method,'plfw') || strcmp(method,'bcplfw') || strcmp(method,'eg') || strcmp(method,'bceg')
    fprintf('\nCollecting initial beliefs..\n');
    for i=1:param.numTrain
        load(sprintf(param.intialBeliefsFileformat,i));
        nodeBelAlphaPrev{i} = nodeBelAlphaPrev;
        edgeBelAlphaPrev{i} = edgeBelAlphaPrev;
    end
end

end
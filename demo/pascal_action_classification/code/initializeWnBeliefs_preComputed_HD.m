function [W,WMat] = initializeWnBeliefs_preComputed_HD(param)

if exist(param.intialWfile,'file')
    load(param.intialWfile);
else
    error('\nFile containing initial parameter W not found.\n');
end

W = W/(param.numTrain*param.lambda);
WMat = WMat/(param.numTrain*param.lambda);

if param.normNormalizeFeatures
    from=1; to=param.numNodes*param.unarySize;
    W(from:to) = W(from:to) / param.maxUnaryFeatNorm;
    WMat(from:to,:) = WMat(from:to,:) / param.maxUnaryFeatNorm;
    
    from=to+1; to=to+param.numEdges*param.binarySize; assert(to==length(W));
    W(from:to) = W(from:to) / param.maxBinaryFeatNorm;
    WMat(from:to,:) = WMat(from:to,:) / param.maxBinaryFeatNorm;
end

method = param.method;
if strcmp(method,'plfw') || strcmp(method,'bcplfw') || strcmp(method,'eg') || strcmp(method,'bceg')
    fprintf('\nCopying initial beliefs..\n');
    copyfile([param.intialBeliefsDir '/*.mat'], param.bel_dir);
end

end
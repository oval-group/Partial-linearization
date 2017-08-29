function [W,WMat] = initializeWnBeliefs_HD(param)

rng(1,'twister');

fprintf('\nGenerating initial beliefs..\n');
for i=1:param.numTrain
    fprintf('%d,',i);
    nodeBelAlphaPrev = rand(param.numNodes, param.numStates)+eps;
    nodeBelAlphaPrev = nodeBelAlphaPrev ./ repmat( sum(nodeBelAlphaPrev,2), [1,param.numStates] );
    edgeBelAlphaPrev = rand(param.numStates,param.numStates,param.numEdges)/(param.numStates^2) + eps;
    edgeBelAlphaPrev = edgeBelAlphaPrev ./ repmat( sum(sum(edgeBelAlphaPrev,1),2), [param.numStates,param.numStates,1]) ;
    if ~exist(param.intialBeliefsDir,'dir')
        mkdir(param.intialBeliefsDir);
    end
    save( sprintf(param.intialBeliefsFileformat,i),'nodeBelAlphaPrev','edgeBelAlphaPrev');
end

fprintf('\nGenerating initial W..\n');
[W,WMat] = computeW(param);

W = W*(param.numTrain*param.lambda);
WMat = WMat*(param.numTrain*param.lambda);

if param.normNormalizeFeatures
    from=1; to=param.numNodes*param.unarySize;
    W(from:to) = W(from:to) * param.maxUnaryFeatNorm;
    WMat(from:to,:) = WMat(from:to,:) * param.maxUnaryFeatNorm;
    
    from=to+1; to=to+param.numEdges*param.binarySize; assert(to==length(W));
    W(from:to) = W(from:to) * param.maxBinaryFeatNorm;
    WMat(from:to,:) = WMat(from:to,:) * param.maxBinaryFeatNorm;
end

save(param.intialWfile,'W','WMat');

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

fprintf('\nCompleted initializing the parameters.\n');

method = param.method;
if strcmp(method,'plfw') || strcmp(method,'bcplfw') || strcmp(method,'eg') || strcmp(method,'bceg')
    fprintf('\nCopying initial beliefs as previous beliefs..\n');
    copyfile([param.intialBeliefsDir '/*.mat'], param.bel_dir);
end

%W = zeros(param.dimension,1);

end

function [W,WMat] = computeW(param)
W = zeros(param.dimension,1);
WMat = zeros(param.dimension,param.numTrain);
psi = zeros(size(W));
for i=1:param.numTrain
    load( sprintf(sprintf(param.intialBeliefsFileformat,i),i) );
    x = param.patterns{i};
    y = param.labels{i};
    curr = 1;
    % load unary features
    unaryFeatures = loadUnaryFeatures(param,x);
    for nodeIndex = 1:param.numNodes
        unFeats = unaryFeatures{nodeIndex};
        unaryW = sum( unFeats.*repmat(nodeBelAlphaPrev(nodeIndex,:)',[1,size(unFeats,2)]) );
        WMat(curr:curr+param.unarySize-1,i) = unaryW(:);
        W(curr:curr+param.unarySize-1) = W(curr:curr+param.unarySize-1) + unaryW(:);
        curr = curr + param.unarySize;
    end
    % load binary features
    edgeFeatures = loadBinaryFeatures(param,x);
    for edgeIndex = 1:param.numEdges
        binFeats = edgeFeatures{edgeIndex}/param.edgeScaling;
        binaryW = sum(sum( binFeats.*repmat(edgeBelAlphaPrev(:,:,edgeIndex),[1,1,size(binFeats,3)]) ));
        WMat(curr:curr+param.binarySize-1,i) = binaryW(:);
        W(curr:curr+param.binarySize-1) = W(curr:curr+param.binarySize-1) + binaryW(:);
        curr = curr + param.binarySize;
    end
    psi_i = posEst_featuremap(param,x,y,i);
    psi = psi + psi_i;
    WMat(:,i) = psi_i - WMat(:,i);
end
W = psi - W;
W = W/(param.numTrain*param.lambda);
WMat = WMat/(param.numTrain*param.lambda);
assert( norm(W-sum(WMat,2))/sqrt((norm(W)*norm(sum(WMat,2)))) < 1e-5 );
end

function [nodeBelInit,W,WMat] = initializeWnBeliefs_RAM(param)

rng(1,'twister');
nodeBelInit = cell(1, param.numTrain);

W = zeros(param.dimension, 1);
psi = zeros(param.dimension, 1);

WMat = zeros(param.dimension, param.numTrain);
for n = 1:param.numTrain
        
    %nodeB = rand(param.numClasses,1);
    nodeB = zeros(param.numClasses,1)+eps;
    %nodeB = ones(param.numClasses,1);
    nodeB(param.labels{n}) = 1;
    nodeBelInit{n} = nodeB/ sum(nodeB);
    unaryFeatures = loadUnaryFeatures(param,n);
    
    for i = 1:param.numClasses
        from = param.unarySize*(i-1)+1;
        to = param.unarySize*i;
        W(from:to) = W(from:to) + nodeBelInit{n}(i)*unaryFeatures;    
        WMat(from:to, n) = WMat(from:to, n) + nodeBelInit{n}(i)*unaryFeatures;    
    end
    
    psi_n = posEst_featuremap(param, param.patterns{n}, param.labels{n}, n);
    WMat(:, n) = psi_n - WMat(:, n);
    psi = psi + psi_n;
end
W = psi - W;
W  = W/(param.numTrain*param.lambda);
WMat  = WMat/(param.numTrain*param.lambda);
%assert( norm(W-sum(WMat, 2))/ (sqrt(norm(W)*norm(sum(WMat,2)))) < 1E-5 )

end
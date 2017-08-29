function [featMapExptcn, lossExptcn, nodeBel] = posEst_oracle(param,model,x,y,sampleNum, nodeBelAlphaPrev)

% load unary features
unaryFeatures = loadUnaryFeatures(param,x);

% Computing unary potentials
nodePot = full( reshape(model.w,[param.unarySize,param.numClasses])' )*unaryFeatures;

loss =  ones(param.numClasses, 1);
loss(y) = 0;
nodePot = nodePot + loss;

% Compute beliefs
nodePotPrev = log(nodeBelAlphaPrev);
nodePot = nodePot/param.temperature + nodePotPrev;
nodePot = nodePot - max(nodePot) + 50;
nodeBel = exp(nodePot);
nodeBel = nodeBel/sum(nodeBel);

% Feature expectation
% featMapExptcn = repmat(unaryFeatures,[1,param.numClasses]).*repmat(nodeBel',[param.unarySize,1]);

% featMapExptcn = repmat(unaryFeatures,[1,param.numClasses]);
% tmp = repmat(nodeBel',[param.unarySize,1]);
% featMapExptcn = featMapExptcn .*tmp;
% 
% featMapExptcn = featMapExptcn(:);

featMapExptcn = zeros(size(model.w));
idx = find(nodeBel>(0.01/numel(nodeBel)));
for i=1:length(idx)
    from = (idx(i)-1)*length(unaryFeatures)+1;
    to = (idx(i))*length(unaryFeatures);
    featMapExptcn(from:to) = nodeBel(idx(i))*unaryFeatures;
end

lossExptcn = nodeBel'*loss;


end



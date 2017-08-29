function psi = posEst_featuremap(param,x,y,sampleNum)

% load unary features
unaryFeatures = loadUnaryFeatures(param,x);
% load( sprintf(param.unaryFeaturesFileFormat, x) );

% load binary features
%edgeFeatures = loadBinaryFeatures(param,x);
% load( sprintf(param.binaryFeaturesFileFormat, x) );

from = param.unarySize*(y-1)+1;
to = param.unarySize*y;
psi = zeros(param.dimension,1);
psi(from:to) = unaryFeatures;

% curr = 1;
% for nodeIndex = 1:param.numNodes
%     candLocs = param.candLocs{sampleNum}{nodeIndex};
%     node_featIdx = find((candLocs(:,1)==y(nodeIndex,1))&(candLocs(:,2)==y(nodeIndex,2)));
%     node_featIdx = node_featIdx(1);
%     unary = squeeze( unaryFeatures{nodeIndex}(node_featIdx,:) );
%     psi(curr:curr+param.unarySize-1) = unary(:);
%     curr = curr + param.unarySize;
% end
% for edgeIndex = 1:param.numEdges    
%     edgeEnds = param.edgeEnds(edgeIndex,:);
%     
%     candLocs = param.candLocs{sampleNum}{edgeEnds(1)};
%     node1_featIdx = find((candLocs(:,1)==y(edgeEnds(1),1))&(candLocs(:,2)==y(edgeEnds(1),2)));
%     node1_featIdx = node1_featIdx(1);
%     
%     candLocs = param.candLocs{sampleNum}{edgeEnds(2)};
%     node2_featIdx = find((candLocs(:,1)==y(edgeEnds(2),1))&(candLocs(:,2)==y(edgeEnds(2),2)));
%     node2_featIdx = node2_featIdx(1);
%     
%     binary = squeeze( edgeFeatures{edgeIndex}(node1_featIdx,node2_featIdx,:) )/param.edgeScaling;
%     psi(curr:curr+param.binarySize-1) = binary(:);
%     curr = curr + param.binarySize;
% end

psi = sparse(psi);

end

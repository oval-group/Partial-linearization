function ybar = posEst_maxOracle(param,model,x,y,sampleNum)

% load unary features
unaryFeatures = loadUnaryFeatures(param,x);

% Computing unary potentials
nodePot = full( reshape(model.w,[param.unarySize,param.numClasses])' )*unaryFeatures;

if ~isempty(y)
    loss =  ones(param.numClasses, 1);
    loss(y) = 0;
    nodePot = nodePot + loss;
end

[val, optimalDecoding] = max(nodePot);
ybar = optimalDecoding;

if(~isempty(y))
    if param.debug
        psi = posEst_featuremap(param,x,y,sampleNum);
        psibar = posEst_featuremap(param,x,ybar,sampleNum);
        lossbar = posEst_loss(param,ybar,y);
        slack = lossbar + full(dot(model.w,psibar-psi));
        if slack<0,
            fprintf('Negative Slack\n');
            %keyboard;
        end
        
    end
end

end

function delta = posEst_loss(param,y,ybar)

if(y==ybar)
    delta = 0;
else
    delta = 1;
end
%delta = mean(sqrt(sum((y-ybar).^2,2))/param.maxDistance);
end

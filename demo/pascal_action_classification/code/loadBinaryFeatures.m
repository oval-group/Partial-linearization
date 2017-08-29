function edgeFeatures = loadBinaryFeatures(param,sampleName)

load( sprintf(param.binaryFeaturesFileFormat, sampleName) );

if param.normNormalizeFeatures
    edgeFeatures = cellfun(@(x) x/param.maxBinaryFeatNorm, edgeFeatures, 'UniformOutput', false);
end

end
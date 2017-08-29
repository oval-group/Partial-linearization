function [options] = setOptions()
    options = [];
    options.lambda = 1e-2;
    options.gap_threshold = 0.01; % duality gap stopping criterion
    options.num_passes = 10; % max number of passes through data
    options.do_line_search = 1;
    options.debug = 0; % for displaying more info (makes code about 3x slower)
    options.useRAM = 1;
    options.wSaveStep = 10;
    options.singleOPfile = 0;
    options.savePerEpoch = 1;
    options.rand_seed = 5;
    options.sample = 'perm';%'uniform';%
    options.gamma_1Meps = 1;
end

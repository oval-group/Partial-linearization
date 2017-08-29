function [model, progress] = solverBCPL(param, options)
% [model, progress] = solverBCPL(param, options)
%
% Solves the structured support vector machine (SVM) using the block-coordinate
% Partial linearization algorithm, see (Mohapatra, Dokania, Jawahar, Kumar, "Partial Linearization 
% based Optimization for Multi-class SVM". ECCV, 2016) for more details.
% This is Algorithm 2 in the paper, and the code here follows a similar
% notation.
%
% This is a modified version of the function 'solverBCFW' which implements
% the block-coordinate Frank-Wolfe algorithm (Lacoste-Julien, Jaggi, Schmidt, Pletscher; ICML
% 2013).
% 
% Inputs:
%   param: a structure describing the problem with the following fields:
%
%     patterns  -- patterns (x_i)
%         A cell array of patterns (x_i). The entries can have any
%         nature (they can just be indexes of the actual data for
%         example).
%     
%     labels    -- labels (y_i)
%         A cell array of labels (y_i). The entries can have any nature.
%
%     lossFn    -- loss function callback
%         A handle to the loss function L(ytruth, ypredict) defined for 
%         your problem. This function should have a signature of the form:
%           scalar_output = loss(param, ytruth, ypredict) 
%         It will be given an input ytruth, a ground truth label;
%         ypredict, a prediction label; and param, the same structure 
%         passed to solverBCFW.
%
%     exptOracleFn  --  expectation oracle
%         A handle to the 'expectation oracle'  
% 
%     oracleFn  -- loss-augmented decoding callback
%         [Can also be called constraintFn for backward compatibility with
%          code using svm_struct_learn.]
%         A handle to the 'maximization oracle'  
%         which solves the loss-augmented decoding problem. This function
%         should have a signature of the form:
%           ypredict = decode(param, model, x, y)
%         where x is an input pattern, y is its ground truth label,
%         param is the input param structure to solverBCFW and model is the
%         current model structure (the main field is model.w which contains
%         the parameter vector).
%
%     featureFn  feature map callback
%         A handle to the feature map function \phi(x,y). This function
%         should have a signature of the form:
%           phi_vector = feature(param, x, y)
%         where x is an input pattern, y is an input label, and param 
%         is the usual input param structure. The output should be a vector 
%         of *fixed* dimension d which is the same
%         across all calls to the function. The parameter vector w will
%         have the same dimension as this feature map. In our current
%         implementation, w is sparse if phi_vector is sparse.
% 
%  options:    (an optional) structure with some of the following fields to
%              customize the behavior of the optimization algorithm:
% 
%   lambda      The regularization constant (default: 1/n).               
%
%   num_passes  Maximum number of passes through the data before the 
%               algorithm stops (default: 200)
%
%   debug       Boolean flag whether to track the primal objective, dual
%               objective, and training error (makes the code about 3x
%               slower given the extra two passes through data).
%               (default: 0)
%   do_linesearch
%               Boolean flag whether to use line-search. (default: 1)
%   do_weighted_averaging
%               Boolean flag whether to use weighted averaging of the iterates.
%               *Recommended -- it made a big difference in test error in
%               our experiments.*
%               (default: 1)
%   time_budget Number of minutes after which the algorithm should terminate.
%               Useful if the solver is run on a cluster with some runtime
%               limits. (default: inf)
%   rand_seed   Optional seed value for the random number generator.
%               (default: 1)
%   sample      Sampling strategy for example index, either a random permutation
%               ('perm') or uniform sampling ('uniform').
%               [Note that our convergence rate proof only holds for uniform
%               sampling, not for a random permutation.]
%               (default: 'uniform')
%   debug_multiplier
%               If set to 0, the algorithm computes the objective after each full
%               pass trough the data. If in (0,100) logging happens at a
%               geometrically increasing sequence of iterates, thus allowing for
%               within-iteration logging. The smaller the number, the more
%               costly the computations will be!
%               (default: 0)
%   test_data   Struct with two fields: patterns and labels, which are cell
%               arrays of the same form as the training data. If provided the
%               logging will also evaluate the test error.
%               (default: [])
%
% Outputs:
%   model       model.w contains the parameters;
%               model.ell contains b'*alpha which is useful to compute
%               duality gap, etc.
%   progress    Primal objective, duality gap etc as the algorithm progresses,
%               can be used to visualize the convergence.
%
% Authors: P. Mohapatra, P. K. Dokania
% Web: 
%
% Relevant Publication:
%       P. Mohapatra, P. K. Dokania, C. V. Jawahar and M. Pawan Kumar, 
%       Partial Linearization based Optimization for Multi-class SVM. 
%       European Conference on Computer Vision (ECCV), 2016

% == getting the problem description:
phi = param.featureFn; % for \phi(x,y) feature mapping
loss = param.lossFn; % for L(ytruth, ypred) loss function
if isfield(param, 'constraintFn')
    % for backward compatibility with svm-struct-learn
    oracle = param.constraintFn;
else
    oracle = param.exptOracleFn; % loss-augmented decoding function
end

maxOracle = param.oracleFn;

patterns = param.patterns; % {x_i} cell array
labels = param.labels; % {y_i} cell array
n = length(patterns); % number of training examples

% == parse the options
options_default = defaultOptions(n);
if (nargin >= 2)
    options = processOptions(options, options_default);
else
    options = options_default;
end

% general initializations
lambda = options.lambda;
phi1 = phi(param, patterns{1}, labels{1},1); % use first example to determine dimension
d = length(phi1); % dimension of feature mapping
using_sparse_features = issparse(phi1);
progress = [];

assert(options.useRAM==1);  % Run this code only on RAM

if options.useRAM
    % if using RAM, initialize the belief(s) cells
    nodeBelS = cell(1,length(patterns));
    edgeBelS = cell(1,length(patterns));
else
    % if not storing previous beliefs on RAM, then create directories for storing them on HD
    if ~exist('../../temp','dir')
        mkdir('../../temp')
    end
    bel_dir = ['../../temp/' sprintf('%s_l%s_t%s', options.method, num2str(options.lambda), num2str(param.temperature))];
    if ~exist(bel_dir,'dir')
        mkdir(bel_dir);
    end
    param.belPrev_fileFormat = [bel_dir '/belPrev_%d.mat'];
    param.belS_fileFormat = [bel_dir '/belS_%d.mat'];
end

% === Initialization ===
% set w to zero vector
% (corresponds to setting all the mass of each dual variable block \alpha_(i)
% on the true label y_i coordinate [i.e. \alpha_i(y) =
% Kronecker-delta(y,y_i)] using notation from Appendix E of paper).

% w: d x 1: store the current parameter iterate
% wMat: d x n matrix, wMat(:,i) is storing w_i (in Alg. 4 notation) for example i.
%    Using implicit dual variable notation, we would have w_i = A \alpha_[i]
%    -- see section 5, "application to the Structural SVM"
% if using_sparse_features
%     model.w = sparse(d,1);
%     wMat = sparse(d,n); 
% else
%     model.w = zeros(d,1);
%     wMat = zeros(d,n); 
% end

if options.useRAM
    [nodeBelAlphaPrev,model.w,wMat] = initializeWnBeliefs_RAM(param);
else
    [model.w,wMat] = initializeWnBeliefs_HD(param);
end
if using_sparse_features
    model.w = sparse(model.w);
end

ell = 0; % this is \ell in the paper. Here it is assumed that at the true label, the loss is zero
         % Implicitly, we have ell = b' \alpha
ellMat = zeros(n,1); % ellMat(i) is \ell_i in the paper (implicitly, ell_i = b' \alpha_[i])

if (options.do_weighted_averaging)
    wAvg = model.w; % called \bar w in the paper -- contains weighted average of iterates
    lAvg = 0;
end

% logging
if (options.debug_multiplier == 0)
    debug_iter = n;
    options.debug_multiplier = 100;
else
    debug_iter = 1;
end
progress.primal = [];
progress.dual = [];
progress.eff_pass = [];
progress.train_error = [];
if (isstruct(options.test_data) && isfield(options.test_data, 'patterns'))
    progress.test_error = [];
end
progress.model = [];

progress.dual_iter = [];
progress.time = [];

fprintf('running BCFW on %d examples. The options are as follows:\n', length(patterns));
options

rand('state',options.rand_seed);
randn('state',options.rand_seed);
tic();


% === Main loop ====
k=0; % same k as in paper
for p=1:options.num_passes

    perm = [];
    if (isequal(options.sample, 'perm'))
        perm = randperm(n);
    end
    
    %param.temperature = param.temperature/10;

    for dummy = 1:n
        tIterID = tic();
        % (each numbered comment correspond to a line in algorithm 4)
        % 1) Picking random example:
        if (isequal(options.sample, 'uniform'))
            i = randi(n); % uniform sampling
        else
            i = perm(dummy); % random permutation
        end
    
        % 2) solve the loss-augmented inference for point i
        if options.useRAM
            % call the expectation oracle
            [featMapExptcn_i, loss_i, nodeBelS{i}] = oracle(param, model, patterns{i}, labels{i}, i, nodeBelAlphaPrev{i});
        else
            % load previous beliefs
            load( sprintf(param.belPrev_fileFormat,i ) );
            % call the expectation oracle
            [featMapExptcn_i, loss_i, nodeBelS , edgeBelS] = oracle(param, model, patterns{i}, labels{i}, i, nodeBelAlphaPrev, edgeBelAlphaPrev);
        end
        
                
        % 3) define the update quantities:
        % [note that lambda*w_s is subgradient of 1/n*H_i(w) ]
        % psi_i(y) := phi(x_i,y_i) - phi(x_i, y)
        psi_i =   phi(param, patterns{i}, labels{i}, i)   -   featMapExptcn_i;
        w_s = 1/(n*lambda) * psi_i;
        ell_s = 1/n*loss_i;

        % sanity check, if this assertion fails, probably there is a bug in the
        % maxOracle or in the featuremap
        %assert((loss_i - model.w'*psi_i) >= -1e-12);
        
        % 4) get the step-size gamma:
        if (options.do_line_search)
            % analytic line-search for the best stepsize [by default]
            % formula from Alg. 4:
            % (lambda * (w_i-w_s)'*w - ell_i + ell_s)/(lambda ||w_i-w||^2)
            % equivalently, we get the following:
            gamma_opt = (model.w'*(wMat(:,i) - w_s) - 1/lambda*(ellMat(i) - ell_s))...
                              / ( (wMat(:,i) - w_s)'*(wMat(:,i) - w_s) +eps);
            % +eps is to avoid division by zero...            
            if options.gamma_1Meps
                gamma = max(0,min(1-eps,gamma_opt)); % truncate on [0,1]
            else
                gamma = max(0,min(1,gamma_opt)); % truncate on [0,1]
            end
        else
            % we use the fixed step-size schedule (as in Algorithm 3)
            gamma = 2*n/(k+2*n);
        end
        % Set gamma to 1 to run exponentiated gradient
        if param.isEG
            gamma = 1;
        end
        
        % 5-6-7-8) finally update the weights and ell variables
        model.w = model.w - wMat(:,i); % this is w^(k)-w_i^(k)
        wMat(:,i) = (1-gamma)*wMat(:,i) + gamma*w_s;
        model.w = model.w + wMat(:,i); % this is w^(k+1) = w^(k)-w_i^(k)+w_i^(k+1)
        
        ell = ell - ellMat(i); % this is ell^(k)-ell_i^(k)
        ellMat(i) = (1-gamma)*ellMat(i) + gamma*ell_s;
        ell = ell + ellMat(i); % this is ell^(k+1) = ell^(k)-ell_i^(k)+ell_i^(k+1)
    
        % 9) Optionally, update the weighted average:
        if (options.do_weighted_averaging)
            rho = 2/(k+2); % resuls in each iterate w^(k) weighted proportional to k
            wAvg = (1-rho)*wAvg + rho*model.w;
            lAvg = (1-rho)*lAvg + rho*ell; % this is required to compute statistics on wAvg -- such as the dual objective
        end
        
        k=k+1;
        
        % update Beliefs of Alphas
        if options.useRAM
            nodeBelAlphaPrev{i} = (1-gamma)*nodeBelAlphaPrev{i} + gamma*nodeBelS{i};
        else
            % load previous and S beliefs
            nodeBelAlphaPrev = (1-gamma)*nodeBelAlphaPrev + gamma*nodeBelS;
            edgeBelAlphaPrev = (1-gamma)*edgeBelAlphaPrev + gamma*edgeBelS;
            save( sprintf(param.belPrev_fileFormat,i ), 'nodeBelAlphaPrev', 'edgeBelAlphaPrev' );
            clear nodeBelAlphaPrev edgeBelAlphaPrev nodeBelS edgeBelS;
        end
        
        % debug: compute objective and duality gap. do not use this flag for
        % timing the optimization, since it is very costly!
        % (makes the code about 3x slower given the additional 2 passes
        % through the data).
        if (options.debug && k >= debug_iter)
            if (options.do_weighted_averaging)
                model_debug.w = wAvg;
                model_debug.ell = lAvg;
            else
                model_debug.w = model.w;
                model_debug.ell = ell;
            end
            f = -objective_function(model_debug.w, model_debug.ell, lambda); % dual value -equation (4)
            primal = primal_objective(param, maxOracle, model, lambda);            
            gap = primal-f;
            train_error = average_loss(param, maxOracle, model_debug);
            fprintf('pass %d (iteration %d), SVM primal = %f, SVM dual = %f, duality gap = %f, train_error = %f \n', ...
                             p, k, primal, f, gap, train_error);

            progress.primal = [progress.primal; primal];
            progress.dual = [progress.dual; f];
            progress.eff_pass = [progress.eff_pass; k/n];
            progress.train_error = [progress.train_error; train_error];
            if (isstruct(options.test_data) && isfield(options.test_data, 'patterns'))
                param_debug = param;
                param_debug.patterns = options.test_data.patterns;
                param_debug.labels = options.test_data.labels;
                test_error = average_loss(param_debug, maxOracle, model_debug);
                progress.test_error = [progress.test_error; test_error];
            end
            
            save(options.outputFile,'progress');

            debug_iter = min(debug_iter+n,ceil(debug_iter*(1+options.debug_multiplier/100))); 
        end

        % time-budget exceeded?
        t_elapsed = toc();
        if (t_elapsed/60 > options.time_budget)
            fprintf('time budget exceeded.\n');
            if (options.do_weighted_averaging)
                model.w = wAvg; % return the averaged version
                model.ell = lAvg;
            else
                model.ell = ell;
            end
            return
        end
        
        if isempty(progress.time)
            prevTime = 0;
        else
            prevTime = progress.time(end);
        end
        curTime = prevTime+toc(tIterID);
        
        if (options.do_weighted_averaging)
            model_debug.w = wAvg;
            model_debug.ell = lAvg;
        else
            model_debug.w = model.w;
            model_debug.ell = ell;
        end
        f = -objective_function(model_debug.w, model_debug.ell, lambda);
        
        if options.savePerEpoch
            progress.time = [progress.time; curTime];
            progress.dual_iter = [progress.dual_iter; f];
        else
            if options.singleOPfile
                progress.time = [progress.time; curTime];
                progress.dual_iter = [progress.dual_iter; f];
                if mod(k,options.wSaveStep)==0
                    progress.model = [progress.model; model];
                end
                %save(options.outputFile,'progress');
            else
                progress.time = curTime;
                progress.dual_iter = f;
                if mod(k,options.wSaveStep)==0
                    progress.model = model;
                end
                %save([options.outputFile(1:end-4) '_i' num2str(k) '.mat'],'progress');
            end
        end
        
        if ~options.debug
            fprintf('pass %d (iteration %d), SVM dual = %.16f\n',  p, k, f);
        end
       
    end
    
    if options.savePerEpoch
        if p==1
            if ~exist(options.outputFile(1:end-4),'dir')
                %mkdir(options.outputFile(1:end-4))
            end
        else
            progress.time = progress.time(2:end);
        end
        progress.model = model;
        progress.nodeBelAlphaPrev = nodeBelAlphaPrev;
        progress.p = p;
        %save([options.outputFile(1:end-4) '/p' num2str(p) '.mat'],'progress');
        progress.dual_iter = [];
        progress.time = progress.time(end);
        progress.model = [];
    end
    
end % end of main loop

if (options.do_weighted_averaging)
    model.w = wAvg; % return the averaged version
    model.ell = lAvg;
else
    model.ell = ell;
end

save([options.outputFile],'model');   % model saved in mat file

end % solverBCFW


function options = defaultOptions(n)

options = [];
options.num_passes = 200;
options.do_line_search = 1;
options.do_weighted_averaging = 0;
options.time_budget = inf;
options.debug = 0;
options.rand_seed = 1;
options.sample = 'uniform'; % sampling strategy in {'uniform', 'perm'}
options.debug_multiplier = 0; % 0 corresponds to logging after each full pass
options.lambda = 1/n;
options.test_data = [];
options.gap_threshold = 0.1;
options.gap_check = 0;%10; % this makes the code about 10% slower than if use gap_check = 0

end % defaultOptions

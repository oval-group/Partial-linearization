function adj = getAdjacencyMatrixFrontal

numStates = 68;

J = [34 33 32 35 36 ... % nose
    31 30 29 28 ... % nose
    40 41 42 39 38 37 ... % left eye
    18:22 ... % left eyebrow
    43 48 47 44 45 46 ... % right eye
    27:-1:23 ... % right eyebrow
    52 51 50 49 61 62 63 53 54 55 65 64 ... % upper lip
    56 66 57 67 59 68 60 58 ... % lower lip
    9:-1:1 10:17]; % jaw

%opts.mixture(4).anno2treeorder = full(sparse(I,J,S,68,68)); % label transformation
parents = [0 1 2 1 4 ... % nose
    1 6 7 8 ... % nose
    9 10 11 10 13 14 ... % left eye
    15:19 ... % left eyebrow
    9 21 22 21 24 25 ... % right eye
    26:30 ... % right eyebrow
    1 32 33 34 34 33 32 32 39 40 40 39 ... % upper lip
    41 44 45 46 47 48 49 ... % lower lip
    47 ... % ren zhong
    51:59 52 61:67]; % jaw

adj = zeros(68,68);

for i = 1:numStates
    if(parents(i))
        adj(J(i), J(parents(i)) ) = 1;
    end
end
adj = adj+adj';
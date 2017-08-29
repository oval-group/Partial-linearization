function o = intersection_over_union(mat1, mat2, filterSize) %% ASSUMED CORRECT.
    y1 = max(mat1(:,1), mat2(:,1));
    x1 = max(mat1(:,2), mat2(:,2));
    y2 = min(mat1(:,1), mat2(:,1)) + filterSize(1)-1;
    x2 = min(mat1(:,2), mat2(:,2)) + filterSize(2)-1;
    w = x2-x1+1;
    h = y2-y1+1;
    inter = w .* h;
    o = inter ./ (2*prod(filterSize)-inter);
    o(w<0) = 0;
    o(h<0) = 0;

end
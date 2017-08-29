function edgeEnds = create_edge_ends(adj)

edges  = [];
for i=1:size(adj,1)
    for j = i+1:size(adj,1)
        if(adj(i,j))
            edges = [edges; [i,j] ];
        end
    end
end

[v,id]=sort(edges*[max(max(edges))+1;1]);
edgeEnds = edges(id,:);

end
function compareConvergenceTime

resultsDir = '../../results/';

%ipLambda = 0.01;

% wrtTime = 1;

files = dir([resultsDir '*.mat']);

splits = cell(length(files),3);
for i=1:length(files)
    split = regexp(files(i).name(1:end-4),'_','split');
    if numel(split)>3
        split = split(1:3);   % assuming the filename is in method_lambda_temp_<everythingelse> format
    elseif numel(split)<3
        split{end+1} = {split, ''};
    end
    splits(i,:) = split;
    %splits(i,:) = regexp(files(i).name(1:end-4),'_','split');
end

lambdas = unique(splits(:,2));

%clr = {'b','g','r','k','m','c','y'};
%marker = {'.','o','+'};
clrM = hsv(8); clr = cell(size(clrM,1),1);
% clrM = [0,0,1;0,1,0;1,0,0];
for i=1:size(clrM,1)
    clr{i} = clrM(i,:);
end
linTyp = {'-.','-','--'};

colIdx = zeros(3,1);

resultsCell = cell(length(lambdas),1);

for i=1:length(lambdas)
    %assert(length(lambdas)==1);
    
    %if strcmp(lambdas{i},['l' num2str(ipLambda)])
        idx = find(strcmp(splits(:,2),lambdas{i}));
        legendCell = cell(1,length(idx));
        
        finalObjVal = inf;
        
        for j=1:length(idx)
            load([resultsDir files(idx(j)).name]);
            
            splt = regexp(files(idx(j)).name(1:end-4),'_','split');
            results(j).method = splt{1};
            results(j).progress = progress;
            
%             if strcmp(splt{1},'bcfw') || strcmp(splt{1},'fw')
%                 finalObjVal = progress.dual_iter(end);
%             end
            if finalObjVal > progress.dual_iter(end)
                finalObjVal = progress.dual_iter(end);
            end
        end
        
        for j=1:length(idx)
            finalIdx = find(results(j).progress.dual_iter >= finalObjVal, 1);
            results(j).finalTime = results(j).progress.time(finalIdx);
        end
    %end
    resultsCell{i} = results;
end

figure;
meanTime = zeros(1,length(resultsCell{1}));
for k=1:length(resultsCell)
    subplot(1,length(resultsCell)+1,k);    
    results = resultsCell{k};
    x = 1:length(results);
    y = [results(:).finalTime];
    meanTime = meanTime + y;
    bar(x,y);
    set(gca,'XTickLabel',{results(:).method});
    set(gca,'YTickLabel',[]);
    for i=1:length(results)
        text(x(i),y(i),num2str(y(i),'%0.2f'),'HorizontalAlignment','center','VerticalAlignment','bottom');
        %text(y(i),x(i),num2str(y(i),'%0.2f'),'HorizontalAlignment','center','VerticalAlignment','bottom');
    end
    title(['l=' lambdas{k}(2:end)]);
    if k==1
        ylabel('Training time (sec)');
    end
end

meanTime = meanTime/length(resultsCell);

subplot(1,length(resultsCell)+1,length(resultsCell)+1);
x = 1:length(resultsCell{1});
y = meanTime;
bar(x,y);
set(gca,'XTickLabel',{resultsCell{1}(:).method});
set(gca,'YTickLabel',[]);
for i=1:length(y)
    text(x(i),y(i),num2str(y(i),'%0.2f'),'HorizontalAlignment','center','VerticalAlignment','bottom');
    %text(y(i),x(i),num2str(y(i),'%0.2f'),'HorizontalAlignment','center','VerticalAlignment','bottom');
end
title('Mean');

figure
x = 1:length(resultsCell{1});
y = meanTime;
bar(x,y,0.5);
set(gca,'XTickLabel',{resultsCell{1}(:).method});
set(gca,'YTickLabel',[]);
set(gca,'FontSize',20)
for i=1:length(y)
    text(x(i),y(i),num2str(y(i),'%0.2f'),'HorizontalAlignment','center','VerticalAlignment','bottom','fontsize',20);
    %text(y(i),x(i),num2str(y(i),'%0.2f'),'HorizontalAlignment','center','VerticalAlignment','bottom');
end
ylabel('Mean Training time (sec)');

end




function result = string_join(L,delimiter)
append_delimiter = @(in) [in delimiter];
tmp = cellfun(append_delimiter, L(1:end-1), 'UniformOutput', false);
result = horzcat(tmp{:}, L{end});
end

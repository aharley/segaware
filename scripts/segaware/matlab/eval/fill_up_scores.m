clear;
% load('crossval_partial_results.mat');
% load('crossval_partial_results2.mat');

nScores = sum(scores~=0);

for i=1:nScores
    score = scores(i);
    code = deploys(i).name(1:end-9);
    save(['scores/' code '.mat'],'score','code');
end
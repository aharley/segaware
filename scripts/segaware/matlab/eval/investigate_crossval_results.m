clc;
clear; clc;
score_mats = dir('scores/*.mat');
nScores = numel(score_mats);

superMat = zeros(nScores,10);

scores_bin1 =0;
scores_bin2 =0;
for i=1:nScores
    % here i need to parse the result dir name
    code = score_mats(i).name(1:end-4);
    s = strsplit(code,'_');
    k1=str2double(s{1}(2:end));
    d1=str2double(s{2}(2:end));
    rC1=strcmp(s{3}(2),'T');
    rB1=strcmp(s{4}(2),'T');
    s1=str2double(['0.' s{5}(2:end)]);
    
    k2=str2double(s{6}(2:end));
    d2=str2double(s{7}(2:end));
    rC2=strcmp(s{8}(2),'T');
    rB2=strcmp(s{9}(2),'T');
    s2=str2double(['0.' s{10}(2:end)]);
    
    superMat(i,1) = k1;
    superMat(i,2) = d1;
    superMat(i,3) = rC1;
    superMat(i,4) = rB1;
    superMat(i,5) = s1;
    superMat(i,6) = k2;
    superMat(i,7) = d2;
    superMat(i,8) = rC2;
    superMat(i,9) = rB2;
    superMat(i,10) = s2;
    load(['scores/' score_mats(i).name]);
    superMat(i,11) = score;
end

superMat = unique(superMat,'rows');

k1s = superMat(:,1);
d1s = superMat(:,2);
s1s = superMat(:,5);
k2s = superMat(:,6);
d2s = superMat(:,7);
s2s = superMat(:,10);

% range = 5:2:11;
% range = 0.01:0.05:0.11;
%range = [0.01,0.06,0.11,0.16,0.21];
range = [0.001,0.003,0.005];
nVals = length(range);
vals = zeros(nVals,1);

%k09_d09_rT_rT_s0030__k09_d01_rF_rT_s3000: 71.449

k1=9;
d1=9;
%s1=0.001;
k2=9;
d2=1;

for i=1:nVals
    %s2 = range(i);
s1 = range(i);    
s2 = 0.3
    
    k1_eq = k1s==k1;
    d1_eq = d1s==d1;    
    s1_eq = s1s==s1;
    k2_eq = k2s==k2;
    d2_eq = d2s==d2;
    s2_eq = s2s==s2;
    
%     include = logical(k1_eq.*d1_eq.*s1_eq.*s2_eq);
    include = logical(k1_eq.*d1_eq.*s1_eq.*k2_eq.*d2_eq.*s2_eq);
    nIncludes = sum(include>0);
%     fprintf('k1=%d, s1=%.4f, s2=%.4f, nIncludes=%d\n',k1,s1,s2,nIncludes);
    fprintf('k1=%d, d1=%d, s1=%.4f, k2=%d, d2=%d, s2=%.4f, nIncludes=%d\n',k1,d1,s1,k2,d2,s2,nIncludes);

    %     fprintf('for k=%d, s2=%.4f, nIncludes = %d\n',k1,s2,nIncludes);
    superMat(include,:)
    if nIncludes > 0
        vals(i) = mean(superMat(include,11));
    end
% whos s2
end
%plot(range,vals);

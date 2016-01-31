function [MissRate] = GetAvgMiss(target, targetout)
target = target';
targetout = targetout';
NPosT = max(1, sum(target(:, 1)>0));
NNegT = sum(target(:, 1)<1);
NegScore = targetout(NPosT+1:end, 1);
PosScore = targetout(1:NPosT);
NegScoreSorted = sort(NegScore, 'descend');
% ref=[10.^(-2.5:.25:0)];
ref=[10.^(-3:.25:-2) 10.^(-2.05:.1:0)];
% switch validationset
%     case 0
%         NNeg = 1200;
%     case 1
%         NNeg = 2000;
%     case 4
%         NNeg = 4025;
% end;
% NNeg = NNeg * 10;
NNeg = NNegT;
refN = round(ref .* NNeg);
refN(refN<1) = [];
refN(refN>NNeg) = [];
ScoreN = NegScoreSorted(refN);
% ScoreN2 = unique(ScoreN);
% ScoreN2 = sort(ScoreN2, 'descend');
PCount = zeros(length(ScoreN), 1);
for loop = 1:length(ScoreN)
    score = ScoreN(loop);
    PCount(loop) = length(find(PosScore<score))/NPosT;
end;
MissRate = mean(PCount);
end


function [alg] = GetSelWeight(alg)
if alg.selectweight
    idxSentence1 = [
        1 1 0 0 0 0; %1 2
        1 0 1 0 0 0; %1 3
        0 1 0 1 0 0; %2 4
        0 0 1 1 0 0; %3 4
        0 0 1 0 1 0; %3 5
        0 0 0 1 0 1; %4 6
        0 0 0 0 1 1; %5 6
        ];
    SentenceToPar1 = [
        1 0 0 0 0 0 0; %7
        1 1 1 1 0 0 0; %7 8 9 10
        0 1 0 0 1 0 0; %8 11
        0 0 1 0 0 1 0; %9 12
        1 1 1 1 1 1 1; %7 8 9 10 11 12 13
        1 1 0 0 1 0 0; %7 8 11
        1 0 1 0 0 1 0; %7 9 12
        ];
    alg.SentenceToPar = SentenceToPar1';
    alg.idxSentence = idxSentence1';
else
    alg.SentenceToPar = 1;
    alg.idxSentence = 1;
%     alg.selectweight = true;
%     if DxysLayer==0
%         alg.selectweight = false;
%         idxSentence = []; SentenceToPar = [];
%     end;
end
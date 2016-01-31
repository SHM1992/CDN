function YUVI = RGB2YUV(I)
%this function is used to transform RGB imgs to YUV imgs and save them

% clear all; clc;

%for training samples:
%load('D:\lab\theory & papers\PD\my PD for surveillance videos\New Joint Deep Learning\data\JDN data\CaltechTrain\w_1.400000_h_1.400000\CaltechTestAllimBoxesBeforeNmsRsz3.mat');
%load('D:\lab\theory & papers\PD\my PD for surveillance videos\New Joint Deep Learning\data\JDN data\CaltechTrain\w_1.400000_h_1.400000\CNNDLTData3Color63_4.mat');

%AllYUVimBoxesBeforeNmsRsz = AllimBoxesBeforeNmsRsz;
%trainBoxesNum = 1;
%for i = 1:length(AllimBoxesBeforeNmsRsz)
%  for j = 1:length(AllimBoxesBeforeNmsRsz{i})
%   I = double(AllimBoxesBeforeNmsRsz{i}{j}.im);
    I = double(I);
    R = I(:,:,1);
    G = I(:,:,2);
    B = I(:,:,3);
    Y = 0.299*R + 0.587*G + 0.114*B;
    U = -0.147*R - 0.289*G + 0.436*B;
    V = 0.615*R - 0.515*G - 0.100*B;
    Y = round(Y);
    U = round(U);
    V = round(V);
    %AllYUVimBoxesBeforeNmsRsz{i}{j}.im = cat(3,Y,U,V);
    YUVI = cat(3,Y,U,V);
%   trainBoxesNum = trainBoxesNum + 1;
%  end
%end

%save('D:\lab\theory & papers\PD\my PD for surveillance videos\New Joint Deep Learning\data\JDN data\CaltechTrain\w_1.400000_h_1.400000\CaltechTestAllYUVimBoxesBeforeNmsRsz.mat',AllYUVimBoxesBeforeNmsRsz);


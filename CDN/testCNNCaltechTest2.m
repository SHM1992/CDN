function [rs] = testCNNCaltechTest2(out, Test_Boxes, Test_Frame)
ImagSourcepath = '..\data\JDN data\CaltechTest\data-USA\';
dstbasepath = '..\dbEval\data-USA\res\Test\';
load('..\data\JDN data\CaltechTest\Params\Caltech_TestFilename');

model.Testing = true;
VNeg = [];
model.interval = 10;
model.testPos = false;
filnum = length(SrcNames);
scale = 2;
Scale_1 = 1/scale;
PCNum = 1;
%     partbox=cell(filnum,1);
for i = 1:PCNum:filnum
    jend = min(PCNum, filnum-i+1);
    for j=1:jend
        srcNames{j} = ImNames{i+j-1};
        dstName3{j} = [dstbasepath srcNames{j}(9:end-9) 'I' srcNames{j}(end-8:end)];
        dstName3{j}(end-2:end) = 'txt';
        dstmatNames{j}=dstName3{j};
        dstmatNames{j}(end-2:end)='mat';
    end;
    if i==1
        j = 1;
        srcName = srcNames{j};
        srcName2 = [ImagSourcepath srcName(1:end)];
    end
    j = 1;
    srcName = srcNames{j};
    srcName2 = [ImagSourcepath srcName(1:end)];
    box{j} = [Test_Boxes(Test_Frame.S(i):Test_Frame.E(i), 1:4) out(1, Test_Frame.S(i):Test_Frame.E(i))'];
    srcName = srcNames{j};
    srcName2 = [ImagSourcepath srcName(1:end)];
    box{j} = [Test_Boxes(Test_Frame.S(i):Test_Frame.E(i), 1:4) out(1, Test_Frame.S(i):Test_Frame.E(i))'];
    srcName = srcNames{j};
    srcName2 = [ImagSourcepath srcName(1:end)];
    box{j} = [Test_Boxes(Test_Frame.S(i):Test_Frame.E(i), 1:4) out(1, Test_Frame.S(i):Test_Frame.E(i))'];
    if mod(i+j-1, 1000) == 0
        fprintf([': test: %d/%d %d found\n'],i+j-1,filnum, size(box{j}, 1));
    end;
    box{j}(:, 3:4) = box{j}(:, 3:4) - box{j}(:, 1:2) + 1;
    box2 = bbNms2(box{j}, 'type', 'max');
    if ~exist(dstName3{j}(1:end-15),'dir') mkdir(dstName3{j}(1:end-15));end
    if ~exist(dstName3{j}(1:end-10),'dir') mkdir(dstName3{j}(1:end-10));end
    fid2 = fopen(dstName3{j}, 'w');
    if ~isempty(box2)
        boxsize = size(box2, 1);
        box2 = box2 .* Scale_1;
        if (boxsize>0)
            fprintf(fid2, '%f, %f, %f, %f, %f\n', box2');
        end;
    end;
    fclose(fid2);
end
if nargin < 3
    suffix = [];
end
rs = dbEval_new_test(0, 61);
end

function [net,cnn_best_index] = cnntrain(net, cnn_index, cnn_best_index, AfterNMS, x, y, opts, test_x, test_y, Test_Boxes, Test_Frame, Train_Boxes)%, tx, ty
y = single(y);
net.batchMaxl  = 10;

TrainIDx = 1:size(y,2);
PosIdx = TrainIDx(y(1, :)>0);
NegIdx = TrainIDx(y(1, :)<1);
if AfterNMS
    PosBatchSize = round(opts.batchsize/5);
else
    PosBatchSize = round(opts.batchsize*0.6);
end
NegBatchSize = opts.batchsize;
fprintf('No.%d cnn is training:\n',cnn_index);
fprintf('PosBatchSize: %d, NegBatchSize: %d\n', PosBatchSize, NegBatchSize);
m = length(NegIdx);
numbatches = floor(m / opts.batchsize);
opts.batchsize = PosBatchSize+NegBatchSize;
PosRepNum = ceil(PosBatchSize*numbatches/length(PosIdx));
PosIdx = repmat(PosIdx, [1 PosRepNum]);
mp = length(PosIdx);

if rem(numbatches, 1) ~= 0
    error('numbatches not integer');
end
net.rL = [];
targetout = zeros(2, numbatches*opts.batchsize);
target = zeros(2, numbatches*opts.batchsize);
TestBatch = 100;
TestBatchNum = floor(size(test_x{1}, 3)/TestBatch);
TestBatchEnd = floor(size(test_x{1}, 3)/TestBatch)*TestBatch;
TestNum = size(test_x{1}, 3);
out = zeros(size(y,1), TestNum);
rs_min = 10000;

for i = 1 : opts.numepochs
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
    OutsIdx = 0;
    tic;
    kk = randperm(m);
    pkk = randperm(mp);
    for l = 1 : numbatches
        trainIdx =  [PosIdx(pkk((l - 1) * PosBatchSize + 1 : l * PosBatchSize)) NegIdx(kk((l - 1) * NegBatchSize + 1 : l * NegBatchSize))];
        for cellidx = 1:length(x)
            batch_x{cellidx} = x{cellidx}(:, :, trainIdx);
        end
        batch_y = y(:,    trainIdx);
        
        net.batchl = l;
        net = cnnff(net, batch_x);
        net = cnnbp(net, batch_y);
        net = cnnapplygrads(net, opts);
        targetout(:, OutsIdx+1:OutsIdx+opts.batchsize) = net.o;
        target(:, OutsIdx+1:OutsIdx+opts.batchsize) = batch_y;
        OutsIdx = OutsIdx + opts.batchsize;
        if isempty(net.rL)
            net.rL(1) = net.L;
        end
        net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
        if mod(l, 32) == 0
            EvalOut = targetout(:, 1:OutsIdx); %net.o;
            EvalGnd = target(:, 1:OutsIdx); %batch_y;
            LogO = log(EvalOut);
            Error = EvalGnd.* LogO;
            Error = max(Error, -20000);
            Error(isnan(Error)) = 0;
            Error = -sum(sum(Error));
            %                 f =
            MissRate = GetAvgMiss(EvalGnd, EvalOut);
            fprintf('Batch: %d(%d), Epoch: %d, Error: %.5f, MissRate: %.5f\n', l, numbatches, i, Error/OutsIdx, MissRate);
        end
    end
    toc;
    for l = 1:TestBatchNum
        for cellidx = 1:length(test_x)
            Tbatch_x{cellidx} = test_x{cellidx}(:, :, (l-1)*TestBatch+1:l*TestBatch);
        end
        net = cnnff(net, Tbatch_x);
        if mod(l, 50) == 0
            fprintf('test l: %d(%d)\n', l,TestBatchNum);
        end;
        out(:, (l-1)*TestBatch+1:l*TestBatch) = net.o;
    end
    for cellidx = 1:length(test_x)
        Tbatch_x{cellidx} = test_x{cellidx}(:, :, TestBatchEnd+1:end);
    end
    net = cnnff(net, Tbatch_x);
    out(:, TestBatchEnd+1:TestNum) = net.o;
    [rs] = testCNNCaltechTest2(out, Test_Boxes, Test_Frame);
    if rs<rs_min 
        rs_min = rs;
        cnn_best_index = i;
    end
    fprintf('Epoch %d, rs: %.2f\n', i, rs);
    net.testrs(i) = rs;
    
    ModelFname = ['.\New_CNNModel\CNN_Model_iter' num2str(cnn_index) '.' num2str(i) '.mat'];
    cnn_model = copycnnmodel(net);
    save(ModelFname, 'cnn_model');
end
end

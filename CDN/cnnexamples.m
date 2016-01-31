clear all;
clc;
% close all;
Reload_readear_Data = true; % True: get features again using the bounding boxes detected by the HOG+CSS+SVM
Init_cnn = true;
AfterNMS = true;
train = true;%true:train all the para in the model
cnn_train = true; %true: train all the deep networks
nn_train = true; %true: train the deep neural network for collaboration
test = true;

if ~exist('Pathadd', 'var')
%    addpath .\gabor
    addpath ..\util
    addpath ..\tmptoolbox\matlab
    addpath ..\tmptoolbox\classify
    addpath ..\tmptoolbox
    addpath ..\tmptoolbox\images
    addpath ..\dbEval
    Pathadd = 1;
end;
wRatio=1.4;
hRatio=1.4;
Crop = [12+1 12+84; 5 5+28-1];
CropSize = Crop(:,2)-Crop(:,1)+1;

TrainCropImagepath=['../data/JDN data/CaltechTrain/' sprintf('w_%f_h_%f/',wRatio,hRatio)];


TrainCropImagesFName = [TrainCropImagepath 'CaltechTestAllimBoxesBeforeNmsRsz3'];
TrainCropLabelsFName = [TrainCropImagepath 'CaltechTestAllimBoxesBeforeNmsRszLabel3'];
TrainCropBoxesFName = [TrainCropImagepath 'CaltechTestAllimBoxesBeforeNmsRszBox3'];
ReaderDataFName = [TrainCropImagepath 'CNNDLTData3Color63_4.mat'];

dstCropImagepath=['../data/JDN data/CaltechTest/' sprintf('w_%f_h_%f/',wRatio,hRatio)];
TestCropImagesFName = [dstCropImagepath 'CaltechTestAllimBoxesBeforeNmsRsz2'];
TestCropLabelsFName = [dstCropImagepath 'CaltechTestAllimBoxesBeforeNmsRszLabel2'];
TestCropBoxesFName = [dstCropImagepath 'CaltechTestAllimBoxesBeforeNmsRszBox2'];

if Reload_readear_Data
    load(TrainCropImagesFName,'AllimBoxesBeforeNmsRsz');
    load(TrainCropLabelsFName, 'Labels');
    load(TrainCropBoxesFName, 'Allpartboxes');
    [train_x, train_y, Train_Boxes, Train_Frame] = GetData_datareader(AllimBoxesBeforeNmsRsz, Labels, Allpartboxes, 1, Crop); %
    
    
    load(TestCropImagesFName, 'AllimBoxesBeforeNmsRsz');
    load(TestCropLabelsFName, 'Labels');
    load(TestCropBoxesFName, 'Allpartboxes');
    [test_x, test_y, Test_Boxes, Test_Frame] = GetData_datareader(AllimBoxesBeforeNmsRsz, Labels, Allpartboxes, 0, Crop);
    save(ReaderDataFName, '-v7.3', 'train_x', 'train_y', 'test_x', 'test_y', 'Test_Boxes', 'Test_Frame', 'Train_Boxes');
    clear AllimBoxesBeforeNmsRsz Labels Allpartboxes;
else
    if ~exist('train_x', 'var') || ~exist('test_x', 'var')
        load(ReaderDataFName, 'train_x', 'train_y', 'test_x', 'test_y', 'Test_Boxes', 'Test_Frame', 'Train_Boxes');
    end
end
%% ex1
if train

    %train the component deep networks
    if cnn_train
        %clustering training samples
        fprintf('training CNN...\n');
        fprintf('sampling...\n');
        [clusterRes1,clusterRes2,clusterRes3] = clustering(train_x,train_y,Train_Boxes,2,2,2);
        cnn = cell(1,7);
        cnn_best_index = [1,1,1,1,1,1,1];%select the trained model with minimum rs 
        % training UDNs
        fprintf('training CNN No.1 to No. 2...\n');
        for i = 1:2
            if Init_cnn || ~isfield(cnn{i}, 'testrs')
                cnn{i}.layers = {
                    struct('type', 'i') %input layer
                    struct('type', 'c', 'outputmaps', 64, 'kernelsize', 9) %convolution layer
                    struct('type', 's', 'scale', 4) %sub sampling layer
                    struct('type', 'c', 'outputmaps', 20, 'kernelsize', 9) %convolution layer
                    };
                load('./CNNModel_init.mat'); %initialization
                cnn{i} = cnnsetup3(cnn{i}, cnn_model, clusterRes1{i}.x, clusterRes1{i}.y, CropSize);
            end
            cnn{i}.CropSize = CropSize;
            opts.alpha = 0.025;
            opts.batchsize = 50;
            opts.numepochs = 5;


            [cnn{i},cnn_best_index(i)] = cnntrain(cnn{i},i, cnn_best_index(i), AfterNMS, clusterRes1{i}.x, clusterRes1{i}.y, opts, test_x, test_y, Test_Boxes, Test_Frame, clusterRes1{i}.boxes);

        end
        fprintf('done!\n');
        fprintf('training CNN No.3-No.4...\n');
        for i = 3:4
            if Init_cnn || ~isfield(cnn{i}, 'testrs')
                cnn{i}.layers = {
                    struct('type', 'i') %input layer
                    struct('type', 'c', 'outputmaps', 64, 'kernelsize', 9) %convolution layer
                    struct('type', 's', 'scale', 4) %sub sampling layer
                    struct('type', 'c', 'outputmaps', 20, 'kernelsize', 9) %convolution layer
                    };
                load('./CNNModel_init.mat');
                cnn{i} = cnnsetup3(cnn{i}, cnn_model, clusterRes2{i-2}.x, clusterRes2{i-2}.y, CropSize);
            end
            cnn{i}.CropSize = CropSize;
            opts.alpha = 0.025;
            opts.batchsize = 50;
            opts.numepochs = 5;


            [cnn{i},cnn_best_index(i)] = cnntrain(cnn{i},i, cnn_best_index(i), AfterNMS, clusterRes2{i-2}.x, clusterRes2{i-2}.y, opts, test_x, test_y, Test_Boxes, Test_Frame, clusterRes2{i-2}.boxes);
			
        end
        fprintf('done!\n');
        fprintf('training CNN No.5-No.6...\n');
        for i = 5:6
            if Init_cnn || ~isfield(cnn{i}, 'testrs')
                cnn{i}.layers = {
                    struct('type', 'i') %input layer
                    struct('type', 'c', 'outputmaps', 64, 'kernelsize', 9) %convolution layer
                    struct('type', 's', 'scale', 4) %sub sampling layer
                    struct('type', 'c', 'outputmaps', 20, 'kernelsize', 9) %convolution layer
                    };
                load('./CNNModel_init.mat');
                cnn{i} = cnnsetup3(cnn{i}, cnn_model, clusterRes3{i-4}.x, clusterRes3{i-4}.y, CropSize);
            end
            cnn{i}.CropSize = CropSize;
            opts.alpha = 0.025;
            opts.batchsize = 50;
            opts.numepochs = 5;


            [cnn{i},cnn_best_index(i)] = cnntrain(cnn{i},i, cnn_best_index(i), AfterNMS, clusterRes3{i-4}.x, clusterRes3{i-4}.y, opts, test_x, test_y, Test_Boxes, Test_Frame, clusterRes3{i-4}.boxes);

        end
        fprintf('done!\n');
        fprintf('training CNN No.7...\n');
        % training one original UDN
        i = 7;
        if Init_cnn || ~isfield(cnn{i}, 'testrs')
            cnn{i}.layers = {
                struct('type', 'i') %input layer
                struct('type', 'c', 'outputmaps', 64, 'kernelsize', 9) %convolution layer
                struct('type', 's', 'scale', 4) %sub sampling layer
                struct('type', 'c', 'outputmaps', 20, 'kernelsize', 9) %convolution layer
                };
            load('./CNNModel_init.mat');
            cnn{i} = cnnsetup3(cnn{i}, cnn_model, train_x, train_y, CropSize);
        end
        cnn{i}.CropSize = CropSize;
        opts.alpha = 0.025;
        opts.batchsize = 50;
        opts.numepochs = 5;


        [cnn{i},cnn_best_index(i)] = cnntrain(cnn{i},i, cnn_best_index(i), AfterNMS, train_x, train_y, opts, test_x, test_y, Test_Boxes, Test_Frame, Train_Boxes);
        save('.\bestindex.mat','cnn_best_index');
    end
    

    %training deep neural network 
    if nn_train
        fprintf('training NN...\n');
        out_input = zeros(size(train_x{1},3),7);
        out = cell(1,7);
        load('.\bestindex.mat');
        for i = 1:7
            fprintf(strcat('loading cnn_model No.',num2str(i)));
            fprintf('\n');      
            modelFname = ['.\New_CNNModel\CNN_Model_iter' num2str(i) '.' num2str(cnn_best_index(i)) '.mat'];
            if exist(modelFname,'file') == 0
                fprintf('error');
                return;
            else
                load(modelFname);
                fprintf('testing CNN...\n');
                out{i} = cnntest(cnn_model,train_x,train_y);
                out_input(:,i) = out{i}(1,:)';
                fprintf('done!\n');
            end 
        end
        %%  Using 50 hidden units
        fprintf('start training NN...\n');
        nn = nnsetup([7 2]);

        nn.lambda = 1e-5;       %  L2 weight decay
        nn.alpha  = 1e-0;       %  Learning rate
        opts.numepochs =  10;   %  Number of full sweeps through data
        opts.batchsize = 142;   %  Take a mean gradient step over this many samples
        nn = nntrain(nn, out_input, train_y', opts);

        out_input = [];
        out = [];
        fprintf('done!\n');
    end
end

%test 
if test
    %test UDNs
    fprintf('start testing...\n');
    out_input = zeros(size(test_x{1},3),7);
    out = cell(1,7);
    load('.\bestindex.mat');
    for i = 1:7
         fprintf(strcat('loading cnn_model No.',num2str(i)));
         fprintf('\n');        
         modelFname = ['.\New_CNNModel\CNN_Model_iter' num2str(i) '.' num2str(cnn_best_index(i)) '.mat'];
         if exist(modelFname,'file') == 0
            fprintf('error');
            return;
         else
            load(modelFname);
            out{i} = cnntest(cnn_model,test_x,test_y);
            out_input(:,i) = out{i}(1,:)';
         end 
    end
    %test DNN
    modelFname = '.\New_NNModel\NN_Model_iter10.mat';
    if exist(modelFname,'file') == 0
        fprintf('error');
        return;
    else
        fprintf('loading nn_model\n');
        load(modelFname);
%         load('.\New_CNNModel\CNN_Model_iter5.mat');
        outputs = nntest(nn,out_input,test_y');
        [rs] = testCNNCaltechTest2(outputs', Test_Boxes, Test_Frame);
        fprintf('the result is %f\n',rs);
    end
    fprintf('done!\n');
    save('.\CDN2.mat','outputs');
end




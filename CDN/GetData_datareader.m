function [Data, Label, Boxes, Frame] = GetData_datareader(data_file, label_file, Allpartboxes, TrainingData, Crop)
DataNum = 0;
datanum = 0;
for i = 1:length(data_file) % for each frame
    datanum = datanum+sum(abs(label_file{i}) > 0.1);
end;
Label = zeros(2, datanum+10000);
Boxes = zeros(datanum+10000, 5);
Frame.S = zeros(length(data_file), 1);
Frame.E = zeros(length(data_file), 1);
fprintf('datanum: %d\n', datanum);
for i = 1:length(data_file) % for each frame
    Frame.S(i) = DataNum+1;
    for j = 1:length(label_file{i})
        if abs(label_file{i}(j)) > 0.1
            DataNum = DataNum + 1;
            im2 = data_file{1, i}{j}.im;
            im2 = im2(Crop(1,1):Crop(1,2), Crop(2,1):Crop(2,2), 1:3);
            im4 = rgb2hsv(im2);
            im5 = single(imresize(im4, 0.5));
            filters = [-1 0 1; -2 0 2; -1 0 1]; %sobel operator
            im71 = zeros(size(im5));
            im72 = zeros(size(im5));
            for chn = 1:3
                im71(:,:,chn) = conv2(im5(:,:,chn), filters, 'same');
                im72(:,:,chn) = conv2(im5(:,:,chn), filters', 'same');
            end
            im73 = sqrt(im71.^2 + im72.^2);
            im73 = single(im73);
            im74 = max(im73, [], 3);
            im7 = [im73(:,:,1) im73(:,:,2); im73(:,:,3) im74];
            im4 = im4(:,:,3);
            im6 = [im5(:,:,1) im5(:,:,2); im5(:,:,3) zeros(size(im5, 1), size(im5, 2))];
            
            im4 = preproc_data(im4);
            im6 = preproc_data(im6);
            im7 = preproc_data(im7);
            im1{1} = squeeze(im4);
            im1{2} = squeeze(im6);
            im1{3} = squeeze(im7);
%             YUVI = RGB2YUV(data_file{i}{j}.im);
            if DataNum==1
                Data{1} = zeros(size(im1{1}, 1), size(im1{1}, 2), datanum+3000, 'single');
                Data{2} = zeros(size(im1{1}, 1), size(im1{1}, 2), datanum+3000, 'single');
                Data{3} = zeros(size(im1{1}, 1), size(im1{1}, 2), datanum+3000, 'single');
              
%                 Data{4} = zeros(size(im1{1}, 1), size(im1{1}, 2),
%                 datanum+3000, 'single');                 %add HOG of Y
%                 Data{5} = zeros(size(im1{1}, 1), size(im1{1}, 2),
%                 datanum+3000, 'single');                 %add HOG of U
%                 Data{4} = zeros(size(im1{1}, 1), size(im1{1},
%                 2),datanum+3000, 'single');              %add similarity or LBP
%                 Data{4} = zeros(size(im1{1}, 1), size(im1{1}, 2),datanum+3000, 'single');
            end
            Data{1}(:,:,DataNum) = im1{1};
            Data{2}(:,:,DataNum) = im1{2};
            Data{3}(:,:,DataNum) = im1{3};

%             Data{4}(:,:,DataNum) = addHOG(YUVI(1));
%             Data{5}(:,:,DataNum) = addHOG(YUVI{2});
%             Data{4}(:,:,DataNum) = addsimilarity(YUVI{1}); %simple one
%             Data{4}(:,:,DataNum) = addLBP(YUVI);
%               Data{4}(:,:,DataNum) = addLBPsimilarity(YUVI(:,:,1));
             
            Boxes(DataNum, :) = Allpartboxes{i}(j, [1:4 end]);
            if label_file{i}(j) > 0
                Label(:, DataNum) = [1; 0];
                if TrainingData
                    %flip positive examples
                    DataNum = DataNum + 1;
                    Boxes(DataNum, :) = Allpartboxes{i}(j, [1:4 end]);
                    im6 = [im5(:,end:-1:1,1) im5(:,end:-1:1,2); im5(:,end:-1:1,3) zeros(size(im5, 1), size(im5, 2))];
                    im7 = [im73(:,end:-1:1,1) im73(:,end:-1:1,2); im73(:,end:-1:1,3) im74(:, end:-1:1)];
                    im7 = preproc_data(im7);
                    im6 = preproc_data(im6);
                    im1{1} = squeeze(im4(:, end:-1:1));
                    im1{2} = squeeze(im6);
%                     YUVI = RGB2YUV(data_file{i}{j}.im(:,end:-1:1,:));
                    Data{1}(:,:,DataNum) = im1{1};
                    Data{2}(:,:,DataNum) = im1{2};
                    Data{3}(:,:,DataNum) = squeeze(im7);

%                     Data{4}(:,:,DataNum) = addHOG(YUVI(1));
%                     Data{5}(:,:,DataNum) = addHOG(YUVI{2});
%                     Data{4}(:,:,DataNum) = addsimilarity(YUVI{1});
%                     Data{4}(:,:,DataNum) = addLBP(YUVI);
%                       Data{4}(:,:,DataNum) = addLBPsimilarity(YUVI(:,:,1));
                    Label(:, DataNum) = [1; 0];
                end
            else
                Label(:, DataNum) = [0; 1];
            end
        end;
    end;
    Frame.E(i) = DataNum;
end;

fprintf('DataNum: %d\n', DataNum);
Data{1}(:,:,DataNum+1:end) = [];
Data{2}(:,:,DataNum+1:end) = [];
Data{3}(:,:,DataNum+1:end) = [];
% Data{4}(:,:,DataNum+1:end) = [];
% Data{5}(:,:,DataNum+1:end) = [];
Label(:,DataNum+1:end) = [];
Boxes(DataNum+1:end, :) = [];
PosL = zeros(2,1);
for i = 1:length(label_file)
    PosL(1) = PosL(1) + sum(label_file{i}>0);
    PosL(2) = PosL(2) + sum(label_file{i}<0);
end;
fprintf('PosL: %d\n', PosL);
PosL2 = sum(Label, 2);
fprintf('PosL2: %d\n', PosL2);
end


function [out_data] = preproc_data(out_data)
out_data = (out_data - mean(out_data(:)))/(std(out_data(:))+0.00001);
out_data = single(out_data);
end

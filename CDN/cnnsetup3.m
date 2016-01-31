function net = cnnsetup3(net, net2, x, y, CropSize)
load('G.mat');
inputmaps = length(x);
mapsize = CropSize';%size(squeeze(x(:, :, 1)));
alg.selectweight = true;
alg = GetSelWeight(alg);
net.alg = alg;
load('CDBNModel.mat');
RandomVal = 0.5;
ModelWeight = 0;
net.w2 = ModelWeight*w2(1:6, 1:7)+RandomVal*rand(6, 7);
net.w2(7, 1:7) = ModelWeight*w2(7, 1:7)+RandomVal*randn(1, 7);
net.w2(:, 8:14) = 0.1*randn(7, 7);
net.w3 = ModelWeight*w3(1:7, 1:7)+RandomVal*rand(7, 7);
net.w3(8:14, :) = 0.1*randn(7, 7);
net.w3(15, :) = ModelWeight*w3(14, 1:7)+RandomVal*rand(1, 7);
net.w3(:, 8:14) = 0.1*randn(15, 7);
net.w_class(1:7, :) = ModelWeight*w_class(1:7, :)+RandomVal*rand(7, 2);
net.w_class(8:14, :) = 0.1*randn(7, 2);
net.w_class(15, :) = ModelWeight*w_class(8, :)+RandomVal*rand(1, 2);
net.w_class(:, 2) = -net.w_class(:, 1);
net.c2 = c2(1:7);
net.c2(8:14) = 0;
net.c3 = c3;
net.c3(8:14) = 0;

for l = 1 : numel(net.layers)   %  layer
    if strcmp(net.layers{l}.type, 's')
        if l<4
            mapsize = floor(mapsize / net.layers{l}.scale);
        else
            fvnum = 0;
            net.layers{l}.mapsize = floor(net.layers{l-1}.mapsize ./ net.layers{l}.scale);
            for j = 1 : inputmaps
                fvnum = fvnum + prod(net.layers{l}.mapsize(j,1:2));
                net.layers{l}.b{j} = net.layers{l}.b{j};
            end
        end
        
    end
    if strcmp(net.layers{l}.type, 'c')
        if l < 3
            mapsize = mapsize - net.layers{l}.kernelsize + 1;
            for j = 1 : net.layers{l}.outputmaps  %  output map
                for i = 1 : inputmaps  %  input map
                    net.layers{l}.k{i}{j} = net2.layers{l}.k{i}{j};
                end
                net.layers{l}.b{j} = net2.layers{l}.b{j};
            end
            inputmaps = net.layers{l}.outputmaps;
        else
            fvnum = 0;
            StartRow = [1 1 4  4  9  9  1 1 1 4 4  4  9  1 1 1   1 -1  1  1       ];
            EndRow =   [3 3 9  9 15 15  3 9 9 9 15 15 15 3 9 15 15 17 15 15      ];
            StartCol = [1 3 1  3  2  4  1 1 4 1 1  4  1  1 1 1   4 1  1  1       ];
            EndCol =   [3 5 3  5  3  5  5 2 5 5 2  5  5  5 5 2   5 5  5  5       ];
            mapsizePrv = mapsize;
            FilterSize = prod(mapsizePrv);
            fltNum = 0;
            for i = 1 : inputmaps
                %                     Rootk{i} = reshape(net2.ffW(1,fltNum+1:fltNum+FilterSize), [mapsize(1) mapsize(2)]);
                fltNum = fltNum + FilterSize;
            end
            for p = 1:net.layers{l}.outputmaps
                startrs = StartRow(p);
                startcs = StartCol(p);
                endrs = EndRow(p);
                endcs = EndCol(p);
                rows = endrs - startrs + 1;
                cols = endcs - startcs + 1;
                net.layers{l}.ksize{p} = [rows cols];
                defp = [startrs+2 startcs];
                defs(p, :) = defp;
                net.layers{l}.Ppos{p} = [startrs+2 startcs];
                mapsize(p,:) = mapsizePrv - [rows cols] + 1;
                fvnum = fvnum + prod(mapsize(p,:));
                fan_out = net.layers{l}.outputmaps * prod(net.layers{l}.ksize{p});
                fan_in = inputmaps * net.layers{l}.kernelsize ^ 2;
                for i = 1 : inputmaps  %  input map
                    net.layers{l}.k{i}{p} = net2.layers{l}.k{i}{p};
                end
                net.layers{l}.b{p} = net2.layers{l}.b{p};
                if p~=18 %not root
                    net.layers{l}.Defw(p, :) = [0.05 0 0.05 0];
                else
                    net.layers{l}.Defw(p, :) = [1000 0 1000 0];
                end
                
            end
            fvnum = net.layers{l}.outputmaps;
            net.layers{l}.mapsize = mapsize;
            inputmaps = net.layers{l}.outputmaps;
        end
    end
end
onum = size(y, 1);

net.ffb = zeros(onum, 1, 'single');
net.ffW = single((rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum)));
if onum == 2
    net.ffW(2, :) = -net.ffW(1, :);
end
end
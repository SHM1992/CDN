function net = cnnbp(net, y)
    n = numel(net.layers);

    %% Sqr loss
%     net.e = net.o - y;
%     net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);
%     net.od = net.e .* (net.o .* (1 - net.o));   %  output delta
%     net.fvd = single(net.ffW' * net.od);              %  feature vector delta
   
    %% LR loss
%     C = 1;
%     C2 = C * 2;
    net.e = net.o - y;
    LogO = log(net.o);
%     LogO(net.o<-10) = net.o(net.o<-10);
    Error = y.*LogO;
    Error(isnan(Error)) = 0;
    Error = max(Error, -20000);
    Error = -sum(sum(Error));
%     Error = min(Error, 20000);    
    net.L = Error / size(net.e, 2);
    net.od = net.e;   %  output delta

alg = net.alg;
IO = net.e;
Ix_class=IO';
% C2 = C * 2;
dw_class =  (net.ph3)'*Ix_class;%+C2*w_class;
idxSentence = alg.idxSentence';
SentenceToPar = alg.SentenceToPar';
Ix3 = (Ix_class*net.w_class').*net.ph3.*(1-net.ph3);
Ix3 = Ix3(:,1:end-1);
c3 = 1; %suppose c3 is always 1
IX_feat3 = (Ix3*c3);

dw3 =  net.ph2'*Ix3;
if alg.selectweight
    dw3(1:7, 1:7) = dw3(1:7, 1:7) .* SentenceToPar';
end;
dw3_biasrow = zeros(1, size(dw3, 2)); %suppose c3 is always 1
net.dc3 = dw3_biasrow;
% dw3 = [dw3; dw3_biasrow];

Ix2 = (Ix3*net.w3').*net.ph2.*(1-net.ph2); 
Ix2 = Ix2(:,1:end-1);
c2 = 1; %suppose c3 is always 1
IX_feat2 = (Ix2*c2);
dw2 =  net.ph1'*Ix2;
if alg.selectweight
    dw2(1:6, 1:7) = dw2(1:6, 1:7) .* idxSentence';
end;
dw2_biasrow = zeros(1, size(dw2, 2));
net.dc2 = dw2_biasrow;

Ix1 = (Ix2*net.w2').*net.ph1.*(1-net.ph1); 
Ix1 = Ix1(:,1:end-1);
c1 = 1;
IX_feat1 = Ix1*c1;

% dw2 = [dw2; dw2_biasrow];
% dw1 = zeros(size(net.w1));
% net.dw1 = dw1;
net.dw2 = dw2;
net.dw3 = dw3;
net.dw_class = dw_class;

net.fvd = [IX_feat1'; IX_feat2'; IX_feat3'];

% fix me!!
% net.fvd = single(net.ffW' * net.od);              %  feature vector delta

%     net.fvd = single(net.ffW' * net.od);              %  feature vector delta

    
%             fprintf('cnbp m');
%             tic;
    if strcmp(net.layers{n}.type, 'c')         %  only conv layers has tanh function
%         net.fvd = net.fvd .* ( (1 - net.fv .* net.fv));
    end
%             toc;
% if net.batchl > net.batchMaxl
    %  reshape feature vector deltas into output map style
%     sa = size(net.layers{n}.a{1});
%     fvnum = sa(1) * sa(2);
    for j = 1 : numel(net.layers{n}.a)
        sa = net.fvsa(j, :);
        fvnum = sa(1) * sa(2);
        def = net.layers{n}.Ppos{j};
        net.layers{n}.d{j} = zeros(net.layers{n}.scrSize{j}, 'single');
        net.layers{n}.d{j}(def(1), def(2), :) = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
        net.layers{n}.d2{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
    end
    
    for l = (n - 1) : -1 : 1
        
        if strcmp(net.layers{l}.type, 'c')
            %             fprintf('cnbp c l');
            %             tic;
            if l < 3
                Scale_one_div = 1/net.layers{l + 1}.scale ^ 2;
            else
                Scale_one_div = 1;
            end
            for j = 1 : numel(net.layers{l}.a)
%                 Sign = net.layers{l}.Sign{j};
%                 Sign = zeros(size(net.layers{l}.a{j}), 'single');
%                 Sign(net.layers{l}.az{j} > 0.00001) = 1;
%                 Sign(net.layers{l}.az{j} < -0.00001)= -1;
%                 Sign = net.layers{l}.az{1}{j} > 0;
                
%                 net.layers{l}.d{j} = single((expand(net.layers{l + 1}.d{j}, [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]) * Scale_one_div) );
                Expanded = zeros(size(net.layers{l}.a{j}), 'single');
                Expanded2 = expand(net.layers{l + 1}.d{j}, [net.layers{l + 1}.scale net.layers{l + 1}.scale 1]);
                Expanded(1:size(Expanded2,1), 1:size(Expanded2,2), :) = Expanded2;
                net.layers{l}.d{j} = single( (1 - net.layers{l}.a{j} .*net.layers{l}.a{j}) .*  ( Expanded * Scale_one_div) );
                net.layers{l}.d{j} = net.layers{l}.d{j} .* net.layers{l}.Sign{j};
            end
            %             toc;
        elseif strcmp(net.layers{l}.type, 's')
            %             fprintf('cnbp s l');
            %             tic;
            for i = 1 : numel(net.layers{l}.a)
                z = zeros(size(net.layers{l}.a{i}));
                for j = 1 : numel(net.layers{l + 1}.a)
                    z = z + convn(net.layers{l + 1}.d{j}, rot180(net.layers{l + 1}.k{i}{j}), 'full');
                end
                net.layers{l}.d{i} = z;
            end
            %             toc;
        end
    end
    
    %%  calc gradients
    %             fprintf('cnbp c gradients');
    %             tic;
    for l = 2 : n
        if strcmp(net.layers{l}.type, 'c')
            for j = 1 : numel(net.layers{l}.a)
                sizedj_1_div(j) =  1 / size(net.layers{l}.d{j}, 3);
            end
            for i = 1 : numel(net.layers{l - 1}.a)
                if l < 3
                    for j = 1 : numel(net.layers{l}.a)
%                         net.layers{l}.dk{i}{j} = fconvn(flipall(net.layers{l - 1}.a{i}), flipall(net.layers{l}.d{j})) * sizedj_1_div(j);
                        dk2 = fconvn(net.layers{l - 1}.a{i}, net.layers{l}.d{j}) * sizedj_1_div(j);
                        dk2 = flipall(dk2);
                        net.layers{l}.dk{i}{j} = dk2;
                    end
                else
                    % Debug me
                    for j = 1 : numel(net.layers{l}.a)
%                         def = net.layers{l}.Ppos{j};                        
%                         net.layers{l}.dk{i}{j} = zeros(size(net.layers{l}.k{i}{j}));

                        def = net.layers{n}.Ppos{j};
                        ksize = size(net.layers{l}.k{i}{j});
%                         a1 = flipall(net.layers{l - 1}.a{i});
%                         d1 = flipall(net.layers{l}.d{j});
%                         dsize = size(net.layers{l}.d{j});
%                         def2 = dsize(1:2) - def(1:2) + 1;
%                         a2 = a1(def2(1):def2(1)+ksize(1)-1, def2(2):def2(2)+ksize(2)-1, :);
%                         d2 = repmat(flipall(net.layers{l}.d2{j}), [ksize(1) ksize(2) 1]);
%                         dk2 = sum(a2.*d2, 3) .* sizedj_1_div(j);
                        
%                         a3 = net.layers{l - 1}.a{i}(def(1):def(1)+ksize(1)-1, def(2):def(2)+ksize(2)-1, :);
                        d3 = repmat((net.layers{l}.d2{j}), [ksize(1) ksize(2) 1]);
                        dk3 = sum(net.layers{l - 1}.a{i}(def(1):def(1)+ksize(1)-1, def(2):def(2)+ksize(2)-1, :).*d3, 3) .* sizedj_1_div(j);
%                         dk3 = sum(a3.*d3, 3) .* sizedj_1_div(j);
                        dk3 = flipall(dk3);
                        net.layers{l}.dk{i}{j} = dk3; %fconvn(flipall(net.layers{l - 1}.a{i}), flipall(net.layers{l}.d{j})) * sizedj_1_div(j);
%                         if sum(sum(abs(dk3-net.layers{l}.dk{i}{j}))) > 0.001
%                             keyboard;
%                         end
%                         net.layers{l}.dk{i}{j}(def(1), def(2)) = net.layers{l - 1}.a{i} .* net.layers{l}.d{j} .* sizedj_1_div(j);
                    end
                    
                end
            end
            for j = 1 : numel(net.layers{l}.a)
                %                 sizedj_1_div =  1 / size(net.layers{l}.d{j}, 3);
                if l >=3
                    d2 = squeeze(net.layers{l}.d2{j});
                    d2 = repmat(d2', [4 1]);
                    ddef = sum(d2.* net.layers{l}.def{j}, 2);
                    net.layers{l}.ddef{j} = ddef * sizedj_1_div(j);
%                     net.layers{l}.ddef{j} = sum(net.layers{l}.d{j}.* net.layers{l}.def{j}) * sizedj_1_div(j);
                end
                net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) * sizedj_1_div(j);
            end
        end
    end
% end
%              toc;
   net.dffW = (net.od * (net.fv)') / size(net.od, 2);
    net.dffb = mean(net.od, 2);

    function X = rot180(X)
        X = flipdim(flipdim(X, 1), 2);
    end
end

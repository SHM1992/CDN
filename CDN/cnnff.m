function net = cnnff(net, x)
    n = numel(net.layers);
    inputmaps = length(x);
    for i = 1:inputmaps
        net.layers{1}.a{i} = x{i};
    end

    for l = 2 : n   %  for each layer
        if strcmp(net.layers{l}.type, 'c')
            if l < 3
                %  !!below can probably be handled by insane matrix operations
                z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.kernelsize - 1 net.layers{l}.kernelsize - 1 0], 'single');
                for j = 1 : net.layers{l}.outputmaps   %  for each output map
                    %  create temp output map
                    z(:) = 0;
                    for i = 1 : inputmaps   %  for each input map
                        %  convolve with corresponding kernel and add to temp output map
                        z1 = fconvn(net.layers{l - 1}.a{i}, flipall(net.layers{l}.k{i}{j}));
                        z = z + (z1);
                    end
                    %  add bias, pass through nonlinearity
                    z1 = tanh(z + net.layers{l}.b{j});
                    Sign = zeros(size(z1), 'single');
                    Sign(z1 > 0.00001) = 1;
                    Sign(z1 < -0.00001)= -1;
                    net.layers{l}.Sign{j} = Sign;
                    net.layers{l}.a{j} = abs(z1);
                end
            else
                for j = 1 : net.layers{l}.outputmaps   %  for each output map
                    z = zeros(size(net.layers{l - 1}.a{1}) - [net.layers{l}.ksize{j}(1) - 1 net.layers{l}.ksize{j}(2) - 1 0], 'single');
                    %  create temp output map
%                     z(:) = 0;
                    for i = 1 : inputmaps   %  for each input map
                        %  convolve with corresponding kernel and add to temp output map
                        z1 = fconvn(net.layers{l - 1}.a{i}, flipall(net.layers{l}.k{i}{j}));
                        z = z + (z1);
                    end
                    net.layers{l}.scrSize{j} = size(z);
                    defw = net.layers{l}.Defw(j, :);
                    ix = zeros(size(z, 3), 1);
                    iy = zeros(size(z, 3), 1);
                    PartScore = zeros(1, 1, size(z, 3), 1);
                    def = net.layers{l}.Ppos{j};
                    for sampIdx = 1:size(z, 3)
                        [score_tmp,Ix_tmp,Iy_tmp] = dtAccS(z(:,:,sampIdx), defw(1), defw(2), defw(3), defw(4));
                        ix(sampIdx) = Ix_tmp(def(1), def(2));
                        iy(sampIdx) = Iy_tmp(def(1), def(2));                        
                        PartScore(sampIdx) = score_tmp(def(1), def(2));
                    end
                    dx  = def(2) - ix;
                    dy  = def(1) - iy;
                    defvector = -[dx.^2 dx dy.^2 dy]';  %Am I right about the sign??
                    net.layers{l}.Sign{j} = 1;
                    %  add bias, pass through nonlinearity
%                     net.layers{l}.az{j} = 1;
                    net.layers{l}.a{j} = (PartScore+net.layers{l}.b{j});
%                     net.layers{l}.a{j} = tanh(PartScore+net.layers{l}.b{j});
                    net.layers{l}.def{j} = defvector;
%                     net.layers{l}.az{j} = tanh(z + net.layers{l}.b{j});
%                     net.layers{l}.a{j} = abs(net.layers{l}.az{j});
                end
            end
            %  set number of input maps to this layers number of outputmaps
            inputmaps = net.layers{l}.outputmaps;
        elseif strcmp(net.layers{l}.type, 's')
            %  downsample
%             fprintf('cnff s');
%             tic
            filter = ones(net.layers{l}.scale, 'single') / (net.layers{l}.scale ^ 2);
            for j = 1 : inputmaps
                  z = fconvn(net.layers{l - 1}.a{j}, flipall(filter));
%                 z = convn(net.layers{l - 1}.a{j}, ones(net.layers{l}.scale) / (net.layers{l}.scale ^ 2), 'valid');   %  !! replace with variable
                net.layers{l}.a{j} = z(1 : net.layers{l}.scale : end, 1 : net.layers{l}.scale : end, :);
            end
%             toc;
        end
    end

    %  concatenate all end layer feature maps into vector
    net.fv = [];
    for j = 1 : numel(net.layers{n}.a)
        sa = size(net.layers{n}.a{j});
        net.fvsa(j, :) = sa;
        net.fv = [net.fv; reshape(net.layers{n}.a{j}, sa(1) * sa(2), sa(3))];
    end
    
    
    w2 = net.w2; w3 = net.w3; c2 = net.c2; c3 = net.c3; w_class = net.w_class;
    S = net.fv';
%     S1 = S(:, 1:6);
%     S2 = S(:, 7:13);
%     S3 = S(:, 14:20);
    N = size(S, 1);
%     d1 = sigm(S(:, 1:6)); 
%     d2 = sigm(S(:, 7:13)); 
%     d3 = sigm(S(:, 14:20));
    d1 = (S(:, 1:6)); 
    d2 = (S(:, 7:13)); 
    d3 = (S(:, 14:20));
%     d1 = [d1 ones(N, 1)];
    d2 = [d2 zeros(N, 7)];
    d3 = [d3 zeros(N, 7)];
    
    ph1 = [sigm(d1) ones(N, 1)];
    ph2 = 1./(1+exp(-(ph1*w2+d2.*repmat(c2, N, 1))));  
    % ph2(:, 15:end) = d2(:, 15:end);
    ph2 = [ph2  ones(N,1)];
    ph3 = 1./(1+exp(-(ph2*w3+d3.*repmat(c3, N, 1))));
    ph3 = [ph3  ones(N,1)];
    net.ph1 = ph1; net.ph2 = ph2; net.ph3 = ph3;
    targetout = exp(ph3*w_class);
    targetout = targetout';
    
    net.o = targetout./repmat(sum(targetout,1),2,1);
    

    %  feedforward into output perceptrons
%     targetout = exp(net.ffW * net.fv+ repmat(net.ffb, 1, size(net.fv, 2)));
%     targetout = min(targetout, 1000);
%     net.o = targetout./repmat(sum(targetout,1),2,1);
%     net.o = sigm(net.ffW * net.fv + repmat(net.ffb, 1, size(net.fv, 2)));
end

function net = cnnapplygrads(net, opts)

if net.batchl>net.batchMaxl
    LayerStart = 2;
else
    LayerStart = 4;
end
for l = LayerStart : numel(net.layers)
    %     for l = 2 : numel(net.layers)
    if strcmp(net.layers{l}.type, 'c')
        for j = 1 : numel(net.layers{l}.a)
            for ii = 1 : numel(net.layers{l - 1}.a)
                net.layers{l}.k{ii}{j} = net.layers{l}.k{ii}{j} - opts.alpha * net.layers{l}.dk{ii}{j};
                if j == 19 && l >= 3
                    net.layers{l}.k{ii}{j}(4:end, 4:5) = 0;
                end
                if j == 20 && l >= 3
                    net.layers{l}.k{ii}{j}(4:end, 1:2) = 0;
                end
            end
            if l >=3
                net.layers{l}.Defw(j, :) = net.layers{l}.Defw(j, :) - opts.alpha * net.layers{l}.ddef{j}';
                %Constrained to be positive
                net.layers{l}.Defw(:, 1) = max(net.layers{l}.Defw(:, 1), 0.01);
                net.layers{l}.Defw(:, 3) = max(net.layers{l}.Defw(:, 3), 0.01);
                % root does not move
                net.layers{l}.Defw(18, 1) = 1000;
                net.layers{l}.Defw(18, 3) = 1000;
            end
            net.layers{l}.b{j} = net.layers{l}.b{j} - opts.alpha * net.layers{l}.db{j};
        end
    end
end

% net.w1 = net.w1 - opts.alpha * net.dw1;
net.w2 = net.w2 - opts.alpha * net.dw2;
net.w3 = net.w3 - opts.alpha * net.dw3;
net.c2 = net.c2 - opts.alpha * net.dc2;
net.c3 = net.c3 - opts.alpha * net.dc3;
net.w_class = net.w_class - opts.alpha * net.dw_class;
regval = 0.01;

[net.w2, net.w3, net.w_class] = GetRegularizedW(net.w2, net.w3, net.w_class, regval);

alg = net.alg;
if alg.selectweight
    net.w2(1:6, 1:7) = net.w2(1:6, 1:7).*alg.idxSentence;
    net.w3(1:7, 1:7) = net.w3(1:7, 1:7).*alg.SentenceToPar;
end
net.c2(net.c2<regval) = regval;
net.c3(net.c3<regval) = regval;

% net.ffW = net.ffW - opts.alpha * net.dffW;
% net.ffb = net.ffb - opts.alpha * net.dffb;

end

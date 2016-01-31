function outputs = nntest(net, x, y)
%     e = 0;
%     bad = [];
    outputs = zeros(size(x,1),size(y,2));
    for i = 1 : size(x, 1)
        %  feedforward
        net = nnff(net, x(i, :), y(i, :));
        outputs(i,:) = net.a{net.n};
%         [~, g] = max(net.a{net.n});
%         if g == 1
%             outputs(:,i) = [1;0];
%         else
%             if g == 2
%                 outputs(:,i) = [0;1];
%             end
%         end
%         if g ~= find(y(i, :))
%             e = e + 1;
%             bad = [bad; i];
%         end
    end
%     er = e / size(x, 1);
    end

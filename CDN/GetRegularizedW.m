function [w2, w3, w_class] = GetRegularizedW(w2, w3, w_class,regval)
        w2_1 = w2(1:6, 1:7);
        w3_1 = w3(1:7, 1:7);

        w2_1(w2_1<regval) = regval;%0.000001 
        w3_1(w3_1<regval) = regval;


        w2(1:6, 1:7) = w2_1;
        w3(1:7, 1:7) = w3_1;                
       
        w_cIdx = false(size(w_class, 1), 1);
        w_cIdx(1:7) = w_class(1:7, 1)<regval;
        w_class(w_cIdx, 1) = regval;

        w2(end, w2(end, :)>-0.1) = -0.1;
        w3(end, w3(end, :)>-0.1) = -0.1;        
        
        w_class(:, 2) = -w_class(:, 1);

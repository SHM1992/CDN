function cnn_model = copycnnmodel(net)
            if net.batchl > net.batchMaxl
                for layer = 1 : numel(net.layers)
                    cnn_model.layers{layer}.type = net.layers{layer}.type;
                    if strcmp(net.layers{layer}.type, 'c')
                        cnn_model.layers{layer}.outputmaps = net.layers{layer}.outputmaps;
                        cnn_model.layers{layer}.kernelsize = net.layers{layer}.kernelsize;
                    end
                    if strcmp(net.layers{layer}.type, 's')
                        cnn_model.layers{layer}.scale = net.layers{layer}.scale;
                    end
                end
                for layer = 2 : numel(net.layers)
                    if strcmp(net.layers{layer}.type, 'c')
                        for il = 1 : length(net.layers{layer}.k)
                            for jl = 1 : (net.layers{layer}.outputmaps)
                                cnn_model.layers{layer}.k{il}{jl} = net.layers{layer}.k{il}{jl};
                            end
                        end
                        if layer>=3
                            cnn_model.layers{layer}.ksize = net.layers{layer}.ksize;
                            cnn_model.layers{layer}.Defw = net.layers{layer}.Defw;
                            cnn_model.layers{layer}.Ppos = net.layers{layer}.Ppos;                        end
                        for jl = 1 : (net.layers{layer}.outputmaps)
                            cnn_model.layers{layer}.b{jl} = net.layers{layer}.b{jl};
                        end
                        %             net.layers{layer}.b{jl} = net.layers{layer}.b{jl} - opts.alpha * net.layers{layer}.db{jl};
                    end
                end
            end
            
cnn_model.w2 = net.w2;
cnn_model.w3 = net.w3;
cnn_model.c2 = net.c2;
cnn_model.c3 = net.c3;
cnn_model.alg = net.alg;
cnn_model.w_class = net.w_class;

           cnn_model.ffW = net.ffW;
           cnn_model.ffb = net.ffb;
           cnn_model.testrs = net.testrs;

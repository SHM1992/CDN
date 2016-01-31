This package contains the Matlab codes implementing the CDN algorithm 
described in ICIP2016 paper "Collaborative deep networks for pedestrian 
detection".

This code is based on the JDN code that is mentioned in [a]. 
We only modified the algorithm to implement our method, but directly used the same preprocessed data to train our model 
and the same evaluation method to compare the JDN and our CDN. 

Before  running our method, you need to download the JDN code first at 
http://www.ee.cuhk.edu.hk/~wlouyang/projects/ouyangWiccv13Joint/index.html.
Then, please delete the CNN folder and copy our CDN folder to the same path (root directory of your project).
In the complete project, you can find CDN folder, data folder, dbEval folder, tmptoolbox folder and util folder. 
Next, pay attention to the path settings in our code and adjust them to fit your environment.
Finally, run .\CDN\cnnexamples.m to implement pedestrian detection and see how the functions are called. 

This code will take several days to finish the training and testing. We are working on acceleration and will release a faster version soon.

For any questions, feel free to email me at songhongmeng4edu@pku.edu.cn.

[a]W. Ouyang and X. Wang. Joint Deep Learning for Pedestrian Detection. In ICCV, 2013.

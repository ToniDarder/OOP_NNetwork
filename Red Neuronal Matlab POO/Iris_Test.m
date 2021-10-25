clear;
clc;
load('IrisTest.mat');
NN = Network([3],0);
data1 = Data('iris.mat',0);
[NN.ThetaOpt,NN.Num_Features] = NN.Train(data1);
Error = abs(NN.ThetaOpt - Mytest.ThetaOpt);
if Error < 5*10^(-2)
    disp('Test Passed');
else
    disp('Test Failed');
end

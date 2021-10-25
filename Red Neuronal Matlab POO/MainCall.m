clear;
clc;
Iris_Test;
Mytest = Network([3],0);
data1 = Data('iris.mat',0);
[Mytest.ThetaOpt,Mytest.Num_Features] = Mytest.Train(data1);
PlotBoundary(data1,Mytest);
clear;
clc;
Mynet = Network([2,3],0);
data1 = Data('iris.mat',0);
[Mynet.ThetaOpt,Mynet.Num_Features] = Mynet.Train(data1);
PlotBoundary(data1,Mynet);
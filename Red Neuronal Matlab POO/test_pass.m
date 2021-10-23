clear;
clc;
Mynet = Network([2,3],0);
data1 = Data('iris.mat',0);
Mynet.ComputeCost(data1.Xdata,data1.Ydata,Mynet.Theta0);

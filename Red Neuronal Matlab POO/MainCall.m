clear;
clc;
%% Testing with a sample nn
Iris_Test;

%% Create Network, data and trainer
lambda = 0;
testratio = 0;
d = 1;
mynet_iris = Network([3,3,3],'linear');
data1 = Data('iris.mat',testratio,d);
linear_trainer = Trainer(lambda,'linear');

%% Train the networks and plot boundaries
showgraph = true;
[mynet_iris.thetaOpt,mynet_iris.thetaOpt_m] = linear_trainer.train(mynet_iris,data1,showgraph);
PlotBoundary(data1,mynet_iris);


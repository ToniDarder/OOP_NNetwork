clear;
clc;
%% Testing with a sample nn
Iris_Test;

%% Create Network, data and trainer
lambda = 0;
testratio = 0;
d = 1;
mynet = Network([3,3,3],'linear');
data1 = Data('iris.mat',testratio,d);
linear_trainer = Trainer(lambda,'linear');

%% Train the networks and plot boundaries
showgraph = true;
[mynet.thetaOpt,mynet.thetaOpt_m] = linear_trainer.train(mynet,data1,showgraph);
PlotBoundary(data1,mynet);


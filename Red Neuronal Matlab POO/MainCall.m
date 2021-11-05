clear;
clc;
%% Testing with a sample nn
Iris_Test;

%% Create Network, data and trainer
mynet = Network([3,3,3]);
data1 = Data('iris.mat',0);
linear_trainer = Trainer(mynet,data1,0);

%% Train the networks and plot boundaries
showgraph = true;
linear_trainer.train(showgraph);
mynet.thetaOpt = linear_trainer.thetaOpt_vec;
PlotBoundary(data1,mynet);


%% Testing with a sample nn
clear; 
close all; 
clc;
%Iris_Test;

%% Initialization of parameters
%lambda = 3.3*10^-4; % [3,3]
lambda = 3.3*10^-4;

datasets = ["iris.csv", "SteelPlateFaults_datax27y1n1941.csv","SteelPlateFaults_datax27y1n700_2345.csv"];   
file = datasets(1);                                                
testratio = 0;                                                      
pol_deg = 1;                                                          

%% Initialize the objects (Data, Network, Trainer)

%data1 = Data(file,testratio,pol_deg);
load("iris_33_goodmin_10e-4.mat");
net_structure = [3,data1.nLabels];      
s.lambda      = lambda;
s.data        = data1;
s.isDisplayed = true;
s.Net_Structure = net_structure;
network = Network(s);
s.network     = network;
s.type = 'SGD';
sgd_opt = Trainer.create(s);
fmin_opt = Fminunc_Optimizer(s);


%% Train the networks and plot boundaries showgraph = true;
% data1.var_corrmatrix();
sgd_opt.train();
nFigure = 100;
network.plotBoundary(nFigure);


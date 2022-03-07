%% Testing with a sample nn
clear; 
close all; 
clc;
%Iris_Test;

%% Initialization of parameters
%lambda = 3.3*10^-4; % [3,3]
lambda = 3.3*10^-4;

datasets = load("datasets.mat").datasets; 
disp('Datsets available:')
disp(datasets)
file = datasets(1);                                                
testratio = 0;                                                      
pol_deg = 1;                                                          

%% Initialize the objects (Data, Network, Trainer)

data1 = Data(file,testratio,pol_deg);
%load("iris_33_goodmin_10e-4.mat");
net_structure = [data1.nFeatures,3,2,3,data1.nLabels];      
s.lambda      = lambda;
s.data        = data1;
s.isDisplayed = true;
s.Net_Structure = net_structure;
network = Network(s);
s.network     = network;
s.type = 'SGD';
sgd_opt = Trainer.create(s);
s2 = s;
s2.type = 'fmin';
fmin_opt = Trainer.create(s2);
s3 = s;
s3.type = 'SGD_mom';
sgd_mom_opt = Trainer.create(s3);

%% Train the networks and plot boundaries showgraph = true;
% data1.var_corrmatrix();
sgd_opt.train();
nFigure = 100;
network.plotBoundary(nFigure);


% setSolverOptions in each Optimizer
% try different lambdas for SGD

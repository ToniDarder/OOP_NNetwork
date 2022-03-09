%% Testing with a sample nn
clc;
clear; 
%close all; 
%Iris_Test;

%% Initialization of hyperparameters
lambda          = 3.3*10^-4;
learningRate    = 0.01;
pol_deg         = 1;  
testratio       = 0;     

%% Loading of files/datasets
datasets = load("datasets.mat").datasets; 
disp('Datsets available:')
disp(datasets)
file = datasets(1);                                                
                                                 
                                                        

%% Initialize the objects (Data, Network, Trainer)
% Create de data and network objects
%data1 = Data(file,testratio,pol_deg);
load("iris_33_goodmin_10e-4.mat");
net_structure       = [data1.nFeatures,3,data1.nLabels];      
n.lambda            = lambda;
n.Net_Structure     = net_structure;
n.data              = data1;
network = Network(n);

% Create a trainer from each type of solver
t               = n;
t.isDisplayed   = true;
t.network       = network;
t2 = t; t3 = t;
t.type  = 'SGD';
t2.type = 'fmin';
t3.type = 'SGD_mom';
sgd_opt     = Trainer.create(t);
fmin_opt    = Trainer.create(t2);
sgd_mom_opt = Trainer.create(t3);

%% Possible functions
% data.var_corrmatrix();
% trainer.train();
% nFigure = 100;
% network.plotBoundary(nFigure);
% network.plotConnections();

%% Suggestions
% setSolverOptions in each Optimizer
% refactoring storeValue, plot and no more print

% try different lambdas for SGD
% try large dataSet for SGD (Gradient) with different lambdas
% try one layer
% try more feautures in iris and see if gradient is still converging slow
% study regularization related with overfitting
% give tolerance line search
% Plot cost,lineSearch (hbar),OptimalityCritera in subPlot for SGD and fmincon

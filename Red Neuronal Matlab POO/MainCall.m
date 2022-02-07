%% Testing with a sample nn
clear; 
close all; 
clc;
%Iris_Test;

%% Initialization of parameters
lambdavec = linspace(1*10^-3,5*10^-2,10);
%lambda = lambdavec(6);
%lambda = 3.3*10^-4; % [3,3]
lambda = 3.3*10^-4;

datasets = ["iris.csv", "SteelPlateFaults_datax27y1n1941.csv","SteelPlateFaults_datax27y1n700_2345.csv"];     % List of Datasets 
file = datasets(1);                                                 % Selected dataset
testratio = 0;                                                      % Percentage of data destined for testing
pol_deg = 1;                                                        % Polinomial degree for feature combinations (1 = given features)   

% Initialize the objects (Data, Network, Trainer)

data1 = Data(file,testratio,pol_deg);
net_structure = [3,data1.numLabels];      
network = Network(net_structure);
s.lambda      = lambda;
s.network     = network;
s.data        = data1;
s.isDisplayed = true;
linearTrainer = Trainer(s);
% data1.var_corrmatrix();

%% Train the networks and plot boundaries
showgraph = true;
linearTrainer.train();
nFigure = 100;
network.plotBoundary(data1,network.thetaOpt_m,nFigure);


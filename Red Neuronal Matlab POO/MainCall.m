%% Testing with a sample nn
clc;
clear; 
close all; 
%Iris_Test;

%% Initialization of hyperparameters
% Data
pol_deg         = 1;  
testratio       = 0; 

% Network
lambda          = 0;

%Trainer
learningRate    = 0.01;
    

%% Loading of files/datasets
datasets = load("datasets.mat").datasets; 
disp('Datsets available:')
disp(datasets)
file = datasets(input('Choose: '));                                                                                                     

%% Initialize the objects (Data, Network, Trainer)
% Create de data and network objects
data1 = Data(file,testratio,pol_deg);
%load("iris_33_goodmin_10e-4.mat");
hiddenlayers = [4,8];
net_structure       = [data1.nFeatures,hiddenlayers,data1.nLabels];      
n.lambda            = lambda;
n.Net_Structure     = net_structure;
n.data              = data1;
n.prop              = 'backprop';
network = Network(n);

% Create a trainer from each type of solver
t               = n;
t.isDisplayed   = true;
t.network       = network;
t.lr            = learningRate;
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
% a.class = ''; a.type = ''; a.min = ; a.max = ;
% analyzeHyperparameter(a,network,data1,sgd_opt)
% nFigure = 100;
% network.plotBoundary(nFigure);
% network.plotConnections();

%% Suggestions

% 1 - Plot boundary in corrmatrix (investigar)
% 2 - PLotconections add colors
% 3 - Try minibatch SGD for iris & larger datasets
% 4 - Plot cost,lineSearch (hbar),OptimalityCritera in subPlot for SGD and fmincon
% 5 - Change plotBoundary to colormapped

% for i = 1:length(y)
%     for j = 1:size(y,2)
%         if y(i,j) ~= 0
%             y2(i) = j;
%         end
%     end
% end



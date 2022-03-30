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

%% Create the data object
data1 = Data(file,testratio,pol_deg);
%load("iris_33_goodmin_10e-4.mat");
%load('Test1iris.mat');
%load('Test4circles1.mat')

%% Create Network Object
hiddenlayers = [4,8];
net_structure           = [data1.nFeatures,hiddenlayers,data1.nLabels];      
n.lambda                = lambda;
n.Net_Structure         = net_structure;
n.data                  = data1;
n.prop                  = 'backprop';
n.costFunction          = '-loglikelihood';
n.activationFunction    = 'sigmoid';
network = Network(n);

%% Create a trainer object
t               = n;
t.isDisplayed   = true;
t.network       = network;
t.lr            = learningRate;
t.type          = 'SGD';
optimizer       = Trainer.create(t);

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
% 2 -c PLotconections add colors
% 3 - Try minibatch SGD for iris & larger datasets
% 4 -c Plot cost,lineSearch (hbar),OptimalityCritera in subPlot for SGD and fmincon
% 5 -c Change plotBoundary to colormapped

% for i = 1:length(y)
%     for j = 1:size(y,2)
%         if y(i,j) ~= 0
%             y2(i) = j;
%         end
%     end
% end



%% Testing with a sample nn
clc;
clear; 
close all; 
%Iris_Test;

%% Initialization of hyperparameters
% Data
pol_deg         = 1;  
testratio       = 20;  %

% Network
lambda          = 5*10^-2;

%Trainer
learningRate    = 0.01;
    

%% Loading of files/datasets
datasets = load("datasets.mat").datasets; 
disp('Datsets available:')
for i = 1:length(datasets)
    fprintf('%d - %s \n',i,datasets(i))
end
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
n.costFunction          = '-loglikelihoodZ';
n.activationFunction    = 'sigmoid';
network = Network(n);

%% Create a trainer object
t               = n;
t.isDisplayed   = true;
t.network       = network;
t.lr            = learningRate;
t.type          = 'fmin';
t.batchsize     = 100;
t.optTolerance  = 10^-5;
t.maxevals      = 5000;
t.learningType  = 'dynamic';
optimizer       = Trainer.create(t);

%% Possible functions
% data.draw_corrmatrix();
% trainer.train();
% a.class = ''; a.type = ''; a.min = ; a.max = ;
% analyzeHyperparameter(a,network,data1,sgd_opt)
% nFigure = 100;
% network.plotBoundary(nFigure);
% network.plotConnections();
% network.plotConfusionMatrix();


%% Suggestions

% 1 -n Plot boundary in corrmatrix (investigar)
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



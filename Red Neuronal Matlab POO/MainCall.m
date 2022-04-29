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
lambda          = 5*10^-4;

%Trainer
learningRate    = 1;

%% Loading of files/datasets
datasets = load("datasets.mat").datasets;
disp('Datsets available:')
for i = 1:length(datasets)
    fprintf('%d - %s \n',i,datasets(i))
end
file = datasets(input('Choose: '));

%% Create the data object
%data1 = Data(file,testratio,pol_deg);
data1 = load('data1BATCHANALY.mat').data1;

%% Create Network Object
hiddenlayers = [500,300];
net_structure           = [data1.nFeatures,hiddenlayers,data1.nLabels];
n.lambda                = lambda;
n.Net_Structure         = net_structure;
n.data                  = data1;
n.prop                  = 'backprop';
n.costFunction          = '-loglikelihood-softmax';
n.activationFunction    = 'tanh';
network = Network(n);

%% Create a trainer object
t               = n;
t.isDisplayed   = false;
t.network       = network;
t.lr            = learningRate;
t.type          = 'SGD';
t.batchsize     = 500;
t.optTolerance  = 1*10^-3;
t.maxevals      = 20000;
t.learningType  = 'static';
t.nPlot         = 100;
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

% for i = 1:length(y)
%     for j = 1:size(y,2)
%         if y(i,j) ~= 0
%             y2(i) = j;
%         end
%     end
% end

% Kernel trick
% Analysis batch size ,lambda size
% Overfitting, data size
% AutoEncoder vs PCA

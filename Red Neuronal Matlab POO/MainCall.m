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
lambda          = 0.0;

%Trainer
learningRate    = 0.1;
alpha           = 0.9;

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
t.network       = network;
t.lr            = learningRate;
t.alpha         = alpha;
t.type          = 'SGD';
t.batchsize     = 200;
t.optTolerance  = 1*10^-3;
t.maxevals      = 20000;
t.learningType  = 'static';
t.isDisplayed   = false;
t.nPlot         = 20;
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

% xGD   = [5.8 8.5 18.9 41.8 86.7 198.1 904.9];
% xNest = [0 24.4 15.4 9.37 18.3 34.2 135.2];
% y     = [10 100 200 500 1000 2000 8000];
% loglog(y,xGD,'b-o',y,xNest,'r-o')
% legend('SGD','Nesterov')
% xlabel('Batch Size')
% ylabel('Computation time [s]')
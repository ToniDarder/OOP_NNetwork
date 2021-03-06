%% Testing with a sample nn
clc;
clear;
close all;

%% Initialization of hyperparameters
pol_deg         = 1;
testratio       = 20;  
lambda          = 0.;
learningRate    = 0.01;
alpha           = 0.5;

%% Loading of files/datasets
datasets = load("datasets.mat").datasets;
disp('Datsets available:')
for i = 1:length(datasets)
    fprintf('%d - %s \n',i,datasets(i))
end
file = datasets(input('Choose: '));

%% Create the data object
data1 = Data(file,testratio,pol_deg);
%data1 = load('data1BATCHANALY.mat').data1;

%% Create Network Object
hiddenlayers = [4,8];
net_structure           = [data1.nFeatures,hiddenlayers,data1.nLabels];
n.lambda                = lambda;
n.Net_Structure         = net_structure;
n.data                  = data1;
n.costFunction          = '-loglikelihood-softmax';
n.activationFunction    = 'tanh';
n.outputFunction        = 'softmax';
network = Network(n);

%% Create a trainer object
t               = n;
t.network       = network;
t.lr            = learningRate;
t.alpha         = alpha;
t.type          = 'Nesterov';
t.batchsize     = 200;
t.optTolerance  = 1*10^-6;
t.maxevals      = 20000;
t.maxepochs     = 1000;
t.earlyStop     = 200;
t.learningType  = 'static';
t.isDisplayed   = false;
t.nPlot         = 40;
optimizer       = Trainer.create(t);

%% Possible functions
% data.draw_corrmatrix();
% trainer.train();
% a.class = ''; a.type = ''; a.min = ; a.max = ;
% analyzeHyperparameter(a,network,data1,optimizer)
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
%     end12
% end

% Kernel trick
% Analysis batch size ,lambda size
% Overfitting, data size
% AutoEncoder vs PCA

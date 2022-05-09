clc;clear;close all;

pol_deg         = 1;
testratio       = 20;  %
lambda          = 0.0;
learningRate    = 0.1;
alpha           = 0.9;

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

%% Create a trainer object
t               = n;
t.lr            = learningRate;
t.alpha         = alpha;
t.type          = 'SGD_mom';
t.optTolerance  = 1*10^-3;
t.maxevals      = 20000;
t.learningType  = 'static';
t.isDisplayed   = false;
t.nPlot         = 20;


y     = [1 5 10 50 100 200 500 1000 2000 4000 8000];
for i = 1:length(y)
    t.batchsize     = y(i);
    network = Network(n);
    t.network       = network;
    optimizer       = Trainer.create(t);
    optimizer.train();
    fvSGD_mom(i) = network.cost;
    [~,y_pred] = max(network.getOutput(data1.Xtest),[],2);
    [~,y_target] = max(data1.Ytest,[],2);
    testError = mean(1-(y_pred ~= y_target));
    teSGD_mom(i) = testError;
    disp(y(i));
end




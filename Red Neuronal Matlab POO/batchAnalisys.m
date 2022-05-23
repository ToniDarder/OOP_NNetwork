clear;clc;

data = load('data1BATCHANALY.mat').data1;
hiddenlayers = [500,300];
net_structure           = [data.nFeatures,hiddenlayers,data.nLabels];
n.lambda                = 0;
n.Net_Structure         = net_structure;
n.data                  = data;
n.prop                  = 'backprop';
n.costFunction          = '-loglikelihood-softmax';
n.activationFunction    = 'ReLU';
network = Network(n);
t               = n;
t.lr            = 0.01;
t.alpha         = 0.5;
t.type          = 'SGD_mom';
t.optTolerance  = 1*10^-10;
t.maxevals      = 20000;
t.maxepochs     = 5000;
t.learningType  = 'static';
t.isDisplayed   = false;
t.nPlot         = 40;   

batch       = [1 5 10 50 100 200 500 1000 2000 4000 8000];
historyFV   = zeros([10,length(batch)]);
historyTE   = zeros([10,length(batch)]);

historyTIME = zeros([10,length(batch)]);

for i = 1:length(batch)
    t.batchsize = batch(i);
    for j = 1:10
    network.computeInitialTheta();
    t.network = network;
    optimizer = Trainer.create(t);
    optimizer.train();
    historyTIME(j,i) = toc;

    [~,y_pred] = max(network.getOutput(data.Xtest),[],2);
    [~,y_target] = max(data.Ytest,[],2);
    testError = mean(y_pred ~= y_target);
    historyFV(j,i) = network.cost;
    historyTE(j,i) = testError;
    fprintf('%d%d',i,j)
   end
end
FV      = round(mean(historyFV,1),4)
FV_sd   = std(historyFV,0,1)
TE      = round(mean(historyTE,1),4)
TE_sd   = std(historyTE,0,1)
TIME    = round(mean(historyTIME,1),4)
TIME_sd = std(historyTIME,0,1)





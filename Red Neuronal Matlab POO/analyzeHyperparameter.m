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
t.type          = 'SGD';
t.optTolerance  = 1*10^-10;
t.maxevals      = 20000;
t.maxepochs     = 5000;
t.learningType  = 'static';
t.isDisplayed   = false;
t.nPlot         = 40;   
t.batchsize     = 200;

object = 'network';
hp.type = 'lambda';


v = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1];
n = length(v);
Tcost = zeros([5,n]);
VtestError = zeros([5,n]);
trainer.isDisplayed = false;
for i = 1:n
    hp.value = v(i);
    for j = 1:5
        network.computeInitialTheta();
        t.network = network;
        optimizer = Trainer.create(t);
        switch object
            case 'data'
                data.updateHyperparameter(hp);
            case 'network'
                network.updateHyperparameter(hp)
            case 'trainer'
                trainer.updateHyperparameter(hp)
        end
        optimizer.train();
        Tcost(j,i) = network.cost;
        [~,y_pred] = max(network.getOutput(data.Xtest),[],2);
        [~,y_target] = max(data.Ytest,[],2);
        VtestError(j,i) = mean(y_pred ~= y_target);
        disp(string(i))
        fprintf('%d %d %f %f',i,j,VtestError(j,i),network.lambda)
    end
end
cost = round(mean(Tcost,1),4)
cost_std = std(Tcost,0,1)
Verror = round(mean(VtestError,1),4)
Verror_std = std(VtestError,0,1)

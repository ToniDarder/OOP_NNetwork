clear;
clc;
load('IrisTest.mat');
irisData = Data('iris.csv',0,1);
s.lambda      = 0;
s.data        = irisData;
s.isDisplayed = false;
s.Net_Structure = [3];
nn = Network(s);
s.network     = nn;
showgraph = false;
linear_trainer = Trainer(s);
linear_trainer.train();
Error = abs(nn.theta - network.theta);
if Error < 5*10^(-2)
    disp('Test Passed');
else
    disp('Test Failed');
end
clear;
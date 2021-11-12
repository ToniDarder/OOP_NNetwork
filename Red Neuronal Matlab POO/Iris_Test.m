clear;
clc;
load('IrisTest.mat');
nn = Network([3],'linear');
data1 = Data('iris.mat',0,1);
showgraph = false;
linear_trainer = Trainer(0,'linear');
[nn.thetaOpt,nn.thetaOpt_m] = linear_trainer.train(nn,data1,showgraph);
Error = abs(nn.thetaOpt - mytest.thetaOpt);
if Error < 5*10^(-2)
    disp('Test Passed');
else
    disp('Test Failed');
end

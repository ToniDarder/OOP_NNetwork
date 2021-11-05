clear;
clc;
load('IrisTest.mat');
nn = Network([3]);
data1 = Data('iris.mat',0);
showgraph = false;
linear_trainer = Trainer(nn,data1,0);
linear_trainer.train(showgraph);
nn.thetaOpt = linear_trainer.thetaOpt_vec;
Error = abs(nn.thetaOpt - mytest.thetaOpt);
if Error < 5*10^(-2)
    disp('Test Passed');
else
    disp('Test Failed');
end

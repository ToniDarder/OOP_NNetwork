%% Testing with a sample nn
Iris_Test;

%% Initialization of parameters
lambdavec = linspace(1*10^-3,5*10^-2,10);
%lambda = lambdavec(6);
%lambda = 3.3*10^-4; % [3,3]
lambda = 0;

datasets = ["iris.csv", "SteelPlateFaults_datax27y1n1941.csv"];     % List of Datasets 
file = datasets(1);                                                 % Selected dataset
testratio = 0;                                                      % Percentage of data destined for testing
pol_deg = 1;                                                        % Polinomial degree for feature combinations (1 = given features)
net_structure = [3,3];                                              % Distribution of layers and neurons in the NN. last layer should be equal to the the number of groups 

% Initialize the objects (Data, Network, Trainer)
mynet_iris = Network(net_structure,'linear');
data1 = Data(file,testratio,pol_deg);
linear_trainer = Trainer(lambda,'linear');

%% Train the networks and plot boundaries
showgraph = true;
[mynet_iris.thetaOpt,mynet_iris.thetaOpt_m] = linear_trainer.train(mynet_iris,data1,showgraph);
PlotBoundary(data1,mynet_iris);


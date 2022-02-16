classdef Trainer < handle

    properties (Access = private) 
       network
       isDisplayed
       %theta0
    end

    methods (Access = public)

        function obj = Trainer(s)
           obj.init(s);
        end
        
        function train(obj)
           nn = obj.network;
           nn.theta = obj.solveTheta(nn);
        end

    end    

    methods (Access = private)

        function init(obj,s)
            obj.network     = s.network;
            obj.isDisplayed = s.isDisplayed;
        end

        function theta = solveTheta(obj,nn)
           opt = obj.setSolverOptions();
           nn.theta = obj.computeinitialtheta(); 
           theta0 = nn.theta;
           F = @(theta) nn.computeCost(theta); 
           theta = fminunc(F,theta0,opt); 
        end

        function stop = myoutput(obj,theta,optimvalues,state,args)
            persistent optfig bound_ev hist_Cost Thistory
            stop = false;  
            switch state
                case 'init'
                    Thistory = [];
                    hist_Cost = [0;0;0];
                    optfig = figure;
                    bound_ev = figure;
                    
                case 'iter'
                    obj.network.theta = theta;
                    Thistory = [Thistory, theta];
                    iter = optimvalues.iteration;
                    f = optimvalues.fval;
                    [c,~] = obj.network.computeLossFunction(theta);
                    r = obj.network.computeRegularizationTerm(theta)*obj.network.lambda; 
                    nIter = 1;
                    if mod(iter,nIter) == 0                       
                        v = 0:nIter:iter;
                        hist_Cost = [hist_Cost(1,:), f;
                                     hist_Cost(2,:), c;
                                     hist_Cost(3,:), r];
                        figure(optfig)
                        plot(v,hist_Cost(1,2:end),'+-r',v,hist_Cost(2,2:end),'+-b',v,hist_Cost(3,2:end),'+-k')
                        legend('Fval','Loss','Regularization')
                        xlabel('Iterations')
                        ylabel('Function Values')
                        drawnow
                    end
                    if mod(iter,25) == 0
                        nFigure = bound_ev;
                        figure(nFigure);
                        obj.network.plotBoundary(nFigure)
                        %PlotBoundary  
                    end
                case 'done'
            end
        end
       
       function theta0 = computeinitialtheta(obj)
           nPL = obj.network.neuronsPerLayer;
           nF = obj.network.data.nFeatures;
           nLayer = obj.network.nLayers;
           nTheta = nF*nPL(1);
           for i = 2:nLayer
                nTheta = nTheta + nPL(i-1)*nPL(i);
           end
           theta0 = zeros(1,nTheta);
       end

       function opt = setSolverOptions(obj)
           opt = optimoptions(@fminunc);
           opt.SpecifyObjectiveGradient = true;
           opt.Algorithm = 'quasi-newton';
           opt.StepTolerance = 10^-6;
           opt.MaxFunctionEvaluations = 3000;              
           if obj.isDisplayed == true
                args = [];
                opt.Display = 'iter';
                opt.CheckGradients = true;
                opt.OutputFcn = @(theta,optimvalues,state)obj.myoutput(theta,optimvalues,state,args);
           end
       end            
    end
end
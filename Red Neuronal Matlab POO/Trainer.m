classdef Trainer < handle

    properties (Access = protected) 
       network
       isDisplayed   
       cost
       figureBoundary
       figureCost
       xIter
       delta
    end

    methods (Access = public, Static)

        function self = create(s)
           switch s.type
               case 'SGD'
                   self = SGD_Optimizer(s);
           end
        end

    end

   methods (Access = public)
        
        function train(self)
           opt = self.setSolverOptions();
           x0  = self.network.theta0; 
           F = @(theta) self.costFunction(theta); 
           %fminunc(F,x0,opt); 
           %obj.StochasticGradientDescent(F,x0,opt);
        end      
    end    

    methods (Access = protected)

        function init(obj,s)
            obj.network     = s.network;
            obj.isDisplayed = s.isDisplayed;
            obj.delta = 10^-4;
        end

        function stop = myoutput(obj,x,optimvalues,state,args)
            stop = false;
            switch state
                case 'init'
                    obj.cost = [0;0;0];
                    obj.figureCost = figure;
                    obj.figureBoundary = figure;                    
                case 'iter'
                    obj.xIter = [obj.xIter, x];
                 %   obj.network.plot(iter);
                    iter = optimvalues.iteration;
                    f    = optimvalues.fval;
                    r = obj.network.regularization;
                    c = obj.network.loss;
                    nIter = 100;
                    if mod(iter,nIter) == 0                       
                        v = 0:nIter:iter;
                        obj.cost = [obj.cost(1,:), f;
                                    obj.cost(2,:), c;
                                    obj.cost(3,:), r];
                        figure(obj.figureCost)
                        plot(v,obj.cost(1,2:end),'+-r',v,obj.cost(2,2:end),'+-b',v,obj.cost(3,2:end),'+-k')
                        legend('Fval','Loss','Regularization')
                        xlabel('Iterations')
                        ylabel('Function Values')
                        drawnow
                    end
                    if mod(iter,nIter*5) == 0
                        figure(obj.figureBoundary);
                        obj.network.plotBoundary(obj.figureBoundary)
                    end
                case 'done'
            end
        end
    end
    
    methods (Access = protected)

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

        function [J,g] = costFunction(obj,x)
            theta = x;
            net   = obj.network;
            net.computeCost(theta)
            J = net.cost;
            g = net.gradient; 
        end       
    end
end
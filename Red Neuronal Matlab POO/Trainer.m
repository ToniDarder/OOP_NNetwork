classdef Trainer < handle

    properties (Access = public) 
       network
       isDisplayed   
       cost
       figureBoundary
       figureCost
       xIter
       delta
    end

    methods (Access = public)

        function self = Trainer(s)
           switch s.type
               case 'SGD'
                   self = SGD_Optimizer(s);
               case 'fmin'
                   self = Fminunc_Optimizer(s);
           end
        end
        
        function train(self)
           opt = self.setSolverOptions();
           x0  = self.network.theta0; 
           F = @(theta) self.costFunction(theta); 
           %fminunc(F,x0,opt); 
           %obj.StochasticGradientDescent(F,x0,opt);
        end      
    end    

    methods (Access = protected)

        function init(self,s)
            self.network     = s.network;
            self.isDisplayed = s.isDisplayed;
            self.delta = 10^-4;
        end

        function stop = myoutput(self,x,optimvalues,state,args)
            stop = false;
            switch state
                case 'init'
                    self.cost = [0;0;0];
                    self.figureCost = figure;
                    self.figureBoundary = figure;                    
                case 'iter'
                    self.xIter = [self.xIter, x];
                 %   obj.network.plot(iter);
                    iter = optimvalues.iteration;
                    f    = optimvalues.fval;
                    r = self.network.regularization;
                    c = self.network.loss;
                    nIter = 100;
                    if mod(iter,nIter) == 0                       
                        v = 0:nIter:iter;
                        self.cost = [self.cost(1,:), f;
                                    self.cost(2,:), c;
                                    self.cost(3,:), r];
                        figure(self.figureCost)
                        plot(v,self.cost(1,2:end),'+-r',v,self.cost(2,2:end),'+-b',v,self.cost(3,2:end),'+-k')
                        legend('Fval','Loss','Regularization')
                        xlabel('Iterations')
                        ylabel('Function Values')
                        drawnow
                    end
                    if mod(iter,nIter*5) == 0
                        figure(self.figureBoundary);
                        self.network.plotBoundary(self.figureBoundary)
                    end
                case 'done'
            end
        end

        function opt = setSolverOptions(self)
           opt = optimoptions(@fminunc);
           opt.SpecifyObjectiveGradient = true;
           opt.Algorithm = 'quasi-newton';
           opt.StepTolerance = 10^-6;
           opt.MaxFunctionEvaluations = 3000;              
           if self.isDisplayed == true
                args = [];
                opt.Display = 'iter';
                opt.CheckGradients = true;
                opt.OutputFcn = @(theta,optimvalues,state)self.myoutput(theta,optimvalues,state,args);
           end
        end 

        function [J,g] = costFunction(self,x)
            theta = x;
            net   = self.network;
            net.computeCost(theta)
            J = net.cost;
            g = net.gradient; 
        end  
    end
end
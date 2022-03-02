classdef SGD_Optimizer < Trainer
    properties (Access = public)

    end

    properties (Access = private)

    end

    methods(Access = public)
        function self = SGD_Optimizer(s)
            self@Trainer(s);  
        end
        
        function train(self)
           opt = self.setSolverOptions();
           x0  = self.network.theta0; 
           F = @(theta) self.costFunction(theta); 
           self.StochasticGradientDescent(F,x0,opt);
        end     
    end
    
    methods(Access = private)
        function StochasticGradientDescent(self,F,x0,opt)
            d   = self.delta;    
            iter = -1; 
            funcount = 0;
            [~,grad] = F(x0);            
            epsilon = 1e-1;
            gnorm = norm(grad,2);
            while gnorm > d                
                if iter == -1
                    x = x0;
                    state = 'init';           
                else
                    state = 'iter';
                end
                [f,grad] = F(x);
                funcount = funcount + 1;
                fOld = f;
               % epsilon = obj.lineSearchLR(x,grad,F);
                while f >= fOld                                    
                    xnew = x - epsilon*grad;
                    [f,~] = F(xnew);
                    epsilon = epsilon/2;
                    funcount = funcount + 1;
                end
                x = xnew;
                epsilon = 40*epsilon;
                gnorm = norm(grad,2);
                self.printValues(iter,funcount,x,f,epsilon,gnorm,state,opt);
                iter = iter + 1;
            end 
        end

        function printValues(self,iter,funcount,x,f,epsilon,gnorm,state,opt)
            formatstr = ' %5.0f       %5.0f    %13.6g  %13.6g   %12.3g\n';
            if mod(iter,20) == 0
                fprintf(['                                                        First-order \n', ...
                    ' Iteration  Func-count       f(x)        Step-size       optimality\n']);
            end
            fprintf(formatstr,iter,funcount,f,epsilon,gnorm);
            if self.isDisplayed == true
                optimvalues.iteration = iter;
                optimvalues.fval = f;
                stop = opt.OutputFcn(x,optimvalues,state);
            end
        end    
        
    end
    
    methods (Access = protected)
        function opt = setSolverOptions(self)
            opt = setSolverOptions@Trainer(self);
        end

        function [J,g] = costFunction(self,theta)
            [J,g] = costFunction@Trainer(self,theta);
        end
    end
end
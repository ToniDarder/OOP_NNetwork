classdef SGD_Optimizer < Trainer

    properties (Access = public)

    end

    properties (Access = private)
       batchSize
       MaxFunctionEvaluations
       lSearchtype
       learningRate
       optTolerance
    end

    methods(Access = public)

        function self = SGD_Optimizer(s)
            self.init(s)
            self.learningRate = s.lr;
            self.batchSize = s.batchsize;
            self.optTolerance = s.optTolerance;
            self.MaxFunctionEvaluations = s.maxevals;
            self.lSearchtype = s.learningType;
        end
        
        function train(self)
           x0  = self.network.theta0; 
           F = @(theta) self.costFunction(theta,self.batchSize); 
           self.optimize(F,x0);
        end 
    end
    
    methods(Access = private)  
        
        function optimize(self,F,x0)
            d        = self.optTolerance; 
            epsilon  = self.learningRate;
            iter     = -1; 
            funcount = 0;
            [~,grad] = F(x0);            
            gnorm = norm(grad,2);
            while gnorm > d && funcount <= self.MaxFunctionEvaluations 
                if iter == -1
                    x = x0;
                    state = 'init';           
                else
                    state = 'iter';
                end
                [f,grad] = F(x);  
                [epsilon,x,funcount] = self.lineSearch(x,grad,F,f,epsilon,funcount);                
                opt.epsilon = epsilon; opt.gnorm = gnorm;
                self.displayIter(iter,funcount,x,f,opt,state);
                gnorm = norm(grad,2);
                funcount = funcount + 1;
                iter = iter + 1;
            end 
        end

        function [e,x,funcount] = lineSearch(self,x,grad,F,fOld,e,funcount)
            switch self.lSearchtype
                case 'static'
                    xnew = x - e*grad;
                case 'dynamic'
                    f = fOld;
                    xnew = x;
                    while f >= 1*(fOld - e*(grad*grad'))
                        xnew = x - e*grad;
                        [f,~] = F(xnew);
                        e = e/2;
                        funcount = funcount + 1;
                    end
                    e = 5*e; 
                case 'fminbnd'
                    xnew = @(e1) x - e1*grad;
                    f = @(e1) F(xnew(e1));
                    [e,~] = fminbnd(f,e/10,e*10);
                    xnew = xnew(e);
            end
            x = xnew;
        end     
        
        function displayIter(self,iter,funcount,x,f,opt,state)
            if self.isDisplayed == true
                self.printValues(funcount,opt,f,iter)
                self.storeValues(x,f,state,opt);
                self.plotMinimization(iter);
            end
        end  

        function printValues(self,funcount,opt,f,iter)
            formatstr = ' %5.0f       %5.0f    %13.6g  %13.6g   %12.3g\n';
            if mod(iter,20) == 0
                fprintf(['                                                        First-order \n', ...
                    ' Iteration  Func-count       f(x)        Step-size       optimality\n']);
            end
            fprintf(formatstr,iter,funcount,f,opt.epsilon,opt.gnorm);
        end
    end    
end
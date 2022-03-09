classdef SGD_Optimizer < Trainer

    properties (Access = public)

    end

    properties (Access = private)
       batchSize
    end

    methods(Access = public)

        function self = SGD_Optimizer(s)
            self.init(s)
            self.batchSize = 150;
        end
        
        function train(self)
           opt = self.setSolverOptions();
           x0  = self.network.theta0; 
           F = @(theta) self.costFunction(theta,self.batchSize); 
           self.optimize(F,x0,opt);
        end     
    end
    
    methods(Access = private)

        function optimize(self,F,x0,opt)
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
                [epsilon,x,funcount] = self.lineSearch(x,grad,F,fOld,epsilon,funcount);      
                %[epsilon,x] = self.lineSearchFm(x,grad,F,epsilon);                
                epsilon = 10*epsilon;
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

        function [e,x,funcount] = lineSearch(self,x,grad,F,fOld,e,funcount)
            f = fOld;
            xnew = x;
            while f >= 1.1*(fOld - e*(grad*grad'))
                xnew = x - e*grad;
                [f,~] = F(xnew);
                e = e/2;
                funcount = funcount + 1;         
            end
            x = xnew;
        end     

        function [e,x] = lineSearchFm(self,x,grad,F,e)
            xnew = @(e1) x - e1*grad;
            f = @(e1) F(xnew(e1));
            [e,c] = fminbnd(f,0.01,e);
            x = xnew(e);
        end

        function opt = setSolverOptions(self)
           opt.MaxFunctionEvaluations = 3000;              
           if self.isDisplayed == true
                args = [];
                opt.Display = 'iter';
                opt.OutputFcn = @(theta,optimvalues,state)self.myoutput(theta,optimvalues,state,args);
           end
        end 
    end
    
end
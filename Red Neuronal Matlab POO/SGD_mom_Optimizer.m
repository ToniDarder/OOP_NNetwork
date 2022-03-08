classdef SGD_mom_Optimizer < Trainer
    properties (Access = public)

    end

    properties (Access = private)
        batchSize
    end

    methods(Access = public)

        function self = SGD_mom_Optimizer(s)
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
            epsilon = 10;
            alpha = 0.5;
            v = 0;
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
                [epsilon,x,funcount,v] = self.lineSearch(x,grad,F,fOld,epsilon,v,alpha,funcount);      
                %[epsilon,x] = self.linSearchFm(x,grad,F,epsilon);                
                epsilon = 10*epsilon;
                if alpha < 0.99
                    if mod(iter,5) == 0
                        alpha = alpha + 0.09;
                    end
                end
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

        function [e,x,funcount,v] = lineSearch(self,x,grad,F,fOld,e,v,a,funcount)
            f = fOld;
            deltaFC = 0;
            while f >= fOld && deltaFC < 10
                v = a*v - e*grad;
                xnew = x + v;
                [f,~] = F(xnew);
                e = e/2;
                deltaFC = deltaFC + 1;         
            end
            funcount = funcount + deltaFC;
            x = xnew;
        end     

        function [e,x] = linSearchFm(self,x,grad,F,e)
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
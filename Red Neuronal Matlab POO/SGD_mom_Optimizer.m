classdef SGD_mom_Optimizer < Trainer
    properties (Access = public)

    end

    properties (Access = private)
       batchSize
       MaxFunctionEvaluations
       lSearchtype
       learningRate
       alpha
       optTolerance
    end

    methods(Access = public)

        function self = SGD_mom_Optimizer(s)
            self.init(s)
            self.learningRate = s.lr;
            self.alpha        = s.alpha;
            self.optTolerance = s.optTolerance;
            self.MaxFunctionEvaluations = s.maxevals;
            self.lSearchtype = s.learningType;
            self.nPlot = s.nPlot;
            self.data = s.data;
            if s.batchsize > length(s.data.Xtrain)
                self.batchSize = length(s.data.Xtrain);
            else
                self.batchSize = s.batchsize;
            end
        end
        
        function train(self)
           tic
           x0  = self.network.thetavec; 
           F = @(theta,X,Y) self.costFunction(theta,X,Y); 
           self.optimize(F,x0);
           toc
        end 
    end
    
    methods(Access = private)  
        
        function optimize(self,F,th0)
            nD = length(self.data.Xtrain);
            epsilon0  = self.learningRate;
            epoch = 1;
            iter     = -1; 
            funcount = 0;
            nB = fix(nD/self.batchSize);
            alarm = 0;
            [~,y_pred] = max(self.network.getOutput(self.data.Xtest),[],2);
            [~,y_target] = max(self.data.Ytest,[],2);
            min_testError = mean(y_pred ~= y_target);
            while epoch <= 500
                order = randperm(nD,nD);
                for i = 1:nB
                    [Xb,Yb] = self.createMinibatch(order,i);
                    if iter == -1
                        th = th0;
                        epsilon = epsilon0;
                        v = 0;
                        state = 'init';   
                    else
                        state = 'iter';
                    end
                    [f,~] = F(th,Xb,Yb);  
                    [th,grad,v] = self.updateX(th,v,epsilon,self.alpha,F,Xb,Yb);                
                    gnorm = norm(grad,2);
                    opt.epsilon = epsilon*gnorm; opt.gnorm = gnorm;
                    %self.displayIter(epoch,iter,funcount,th,f,opt,state);
                    funcount = funcount + 1;
                    iter = iter + 1;
                    if toc > 120
                        break
                    end
                end
                [~,y_pred] = max(self.network.getOutput(self.data.Xtest),[],2);
                [~,y_target] = max(self.data.Ytest,[],2);
                testError = mean(y_pred ~= y_target);
                if testError < min_testError
                    min_testError = testError;
                    alarm = 0;
                elseif testError == min_testError
                    alarm = alarm + 0.25;
                else
                    alarm = alarm + 1;
                end
                epoch = epoch + 1;
                if toc > 120
                    break
                end
            end 
        end

        function [x,grad,v] = updateX(self,x,v,e,alpha,F,Xb,Yb)
            x_hat    = x + alpha*v;
            [~,grad] = F(x_hat,Xb,Yb);
            v        = alpha*v - e*grad;
            x        = x + v;     
        end     
        
        function displayIter(self,epoch,iter,funcount,x,f,opt,state)
            self.printValues(epoch,funcount,opt,f,iter)
            if self.isDisplayed == true
                if mod(iter,self.nPlot) == 0 || iter == -1
                    self.storeValues(x,f,state,opt);
                    self.plotMinimization(iter);
                end
            end
        end  

        function printValues(self,epoch,funcount,opt,f,iter)
            formatstr = '%5.0f    %5.0f       %5.0f    %13.6g  %13.6g   %12.3g\n';
            if mod(iter,20) == 0
                fprintf(['                                                        First-order \n', ...
                    'Epoch Iteration  Func-count       f(x)        Step-size       optimality\n']);
            end
            fprintf(formatstr,epoch,iter,funcount,f,opt.epsilon,opt.gnorm);
        end

        function [x,y] = createMinibatch(self,order,i)
               I = self.batchSize;            
               X = self.data.Xtrain;
               Y =  self.data.Ytrain;
               cont = 1;
               if i == fix(length(X)/I)
                   plus = mod(length(X),I);
                   x = zeros([I+plus,size(X,2)]);
                   y = zeros([I+plus,size(Y,2)]);
               else
                   plus = 0;
                   x = zeros([I,size(X,2)]);
                   y = zeros([I,size(Y,2)]);
               end
               for j = (i-1)*I+1:i*I+plus
                   x(cont,:) = X(order(j),:);
                   y(cont,:) = Y(order(j),:);
                   cont = cont+1;
               end
           end
    end    
end
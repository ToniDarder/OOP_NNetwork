classdef SGD < Trainer

    properties (Access = protected)
       lSearchtype
       learningRate
       MaxEpochs
       MaxFunctionEvaluations
       optTolerance
       earlyStop
       batchSize
    end

    methods(Access = public)

        function self = SGD(s)
            self.init(s)
            self.data         = s.data;
            self.lSearchtype  = s.learningType;
            self.learningRate = s.lr;
            self.optTolerance = s.optTolerance;
            self.earlyStop    = s.earlyStop;
            self.nPlot        = s.nPlot;
            self.MaxEpochs    = s.maxepochs;
            self.MaxFunctionEvaluations = s.maxevals;
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
            nD            = length(self.data.Xtrain);
            epsilon0      = self.learningRate;
            epoch         =  1;
            iter          = -1; 
            funcount      =  0;
            alarm         =  0;
            gnorm         =  1;
            min_testError =  1;
            nB            = fix(nD/self.batchSize);
            criteria(1)   = epoch <= self.MaxEpochs; 
            criteria(2)   = alarm < self.earlyStop; 
            criteria(3)   = gnorm > self.optTolerance;
            while criteria(1) == 1 && criteria(2) == 1 && criteria(3) == 1
                if nB == 1
                    order = 1:nD;
                else
                    order = randperm(nD,nD);
                end
                for i = 1:nB
                    [Xb,Yb] = self.createMinibatch(order,i);
                    if iter == -1
                        th      = th0;
                        epsilon = epsilon0;
                        state   = 'init';   
                    else
                        state   = 'iter';
                    end
                    [f,grad]              = F(th,Xb,Yb);  
                    [epsilon,th,funcount] = self.lineSearch(th,grad,F,f,epsilon,epsilon0,funcount,Xb,Yb);                
                    gnorm                 = norm(grad,2);
                    opt.epsilon           = epsilon*gnorm; 
                    opt.gnorm             = gnorm;
                    funcount              = funcount + 1;
                    iter                  = iter + 1;
                    self.displayIter(epoch,iter,funcount,th,f,opt,state);
%                     if toc > 120
%                         break
%                     end
                end
                [alarm,min_testError] = self.validateES(alarm,min_testError);
                epoch = epoch + 1;
%                 if toc > 120
%                     break
%                 end
                criteria(1)   = epoch <= self.MaxEpochs; 
                criteria(2)   = alarm < self.earlyStop; 
                criteria(3)   = gnorm > self.optTolerance;
            end
            if criteria(1) == 0
                fprintf('Minimization terminated, maximum number of epochs reached %d\n',epoch)
            elseif criteria(2) == 0
                fprintf('Minimization terminated, validation set did not decrease in %d iterations\n',self.earlyStop)
            else
                fprintf('Minimization terminated, reached the optimalty tolerance of %f\n',self.optTolerance)
            end
        end

        function [e,x,funcount] = lineSearch(self,x,grad,F,fOld,e,e0,funcount,Xb,Yb)
            type = self.lSearchtype;
            switch type
                case 'static'
                    xnew = self.step(x,e,grad);
                case 'decay'
                    tau = 50;
                    xnew = self.step(x,e,grad);
                    e = e - 0.99*e0*30/tau;
                case 'dynamic'
                    f = fOld;
                    xnew = x;
                    while f >= 1.001*(fOld - e*(grad*grad'))
                        xnew = self.step(x,e,grad);
                        [f,~] = F(xnew,Xb,Yb);
                        e = e/2;
                        funcount = funcount + 1;
                    end
                    e = 10*e;                
                case 'fminbnd'
                    xnew = @(e1) self.step(x,e1,grad);
                    f = @(e1) F(xnew(e1),Xb,Yb);
                    [e,~] = fminbnd(f,e/10,e*10);
                    xnew = xnew(e);
            end
            x = xnew;
        end 

        function x = step(self,x,e,grad,F,Xb,Yb)
            x = x - e*grad;
        end

        function [alarm,min_testError] = validateES(self,alarm,min_testError)
            [~,y_pred]   = max(self.network.getOutput(self.data.Xtest),[],2);
            [~,y_target] = max(self.data.Ytest,[],2);
            testError    = mean(y_pred ~= y_target);
            if testError < min_testError
                min_testError = testError;
                alarm = 0;
            elseif testError == min_testError
                alarm = alarm + 0.5;
            else
                alarm = alarm + 1;
            end
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
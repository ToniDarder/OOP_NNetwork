classdef Trainer < handle

    properties (Access = private) 
       network
       isDisplayed   
       cost
       figureBoundary
       figureCost
       xIter
       delta
    end

    methods (Access = public)

        function obj = Trainer(s)
           obj.init(s);
        end
        
        function train(obj)
           opt = obj.setSolverOptions();
           x0  = obj.network.theta0; 
           F = @(theta) obj.costFunction(theta); 
           fminunc(F,x0,opt); 
           obj.StochasticGradientDescent(F,x0,opt);
        end    
    end    

    methods (Access = private)

        function init(obj,s)
            obj.network     = s.network;
            obj.isDisplayed = s.isDisplayed;
            obj.delta = 10^-4;
        end

        function StochasticGradientDescent(obj,F,x0,opt)
            d   = obj.delta;    
            iter = -1; 
            [f,grad] = F(x0);
            epsilon = [];
            gnorm = norm(grad,2);
            while gnorm > d
                if iter == -1
                    x = x0;
                    state = 'init';           
                else
                    state = 'iter';
                end
                fOld = f; gradOld = grad; epsilonOld = epsilon;
                epsilon = obj.lineSearchLR(x,grad,F);
                deltaX = epsilon*grad;
                x = x - deltaX;
                [f,grad] = F(x);
%                 if f >= fOld
%                     f = fOld; epsilon = epsilonOld; grad = gradOld;
%                 end
                gnorm = norm(grad,2);
                if obj.isDisplayed == true                    
                    formatstr = ' %5.0f       %5.0f    %13.6g  %13.6g   %12.3g\n';
                    if mod(iter,20) == 0
                        fprintf(['                                                        First-order \n', ...
                            ' Iteration  Func-count       f(x)        Step-size       optimality\n']);
                    end
                    fprintf(formatstr,iter,iter,f,epsilon,gnorm);
                    optimvalues.iteration = iter;
                    optimvalues.fval = f;
                    stop = opt.OutputFcn(x,optimvalues,state);
                end
                iter = iter + 1;
            end 
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
                    nIter = 1;
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
                    if mod(iter,25) == 0
                        figure(obj.figureBoundary);
                        obj.network.plotBoundary(obj.figureBoundary)
                    end
                case 'done'
            end
        end

        function [J,g] = costFunction(obj,x)
            theta = x;
            net   = obj.network;
            net.computeCost(theta)
            J = net.cost;
            g = net.gradient; 
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

    methods (Access = private,Static)
        function e = lineSearchLR(x,g,F)
            e = [linspace(0.01,1,10),linspace(2,10,9),linspace(20,100,9)];    % Learning Rate, set to small cte o do a linesearch
            newg = zeros([length(e),length(x)]);
            newx = x - e'.*g; 
            j = zeros([1,length(e)]);
                for i = 1:length(e)
                    [j(i),newg(i,:)] = F(newx(i,:));
                end
                [M,I] = min(j);
                e = e(I);
                f = M;
       end
    end
end
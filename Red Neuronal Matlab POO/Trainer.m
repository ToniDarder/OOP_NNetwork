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
               case 'fmin'
                   self = Fminunc_Optimizer(s);
               case 'SGD_mom'
                   self = SGD_mom_Optimizer(s);
           end
        end
    end

    methods (Access = protected)

        function init(self,s)
            self.network     = s.network;
            self.isDisplayed = s.isDisplayed;
            self.delta = 10^-4;
        end

        function [J,g] = costFunction(self,x,I)
            theta = x;
            Ibatch = I;
            net   = self.network;
            net.computeCost(theta,Ibatch)
            J = net.cost;
            g = net.gradient;
        end

        %function storeValue
        % function plot()

        % No more print

        function stop = myoutput(self,x,optimvalues,state,args)
            stop = false;
            switch state
                case 'init'
                    self.cost = [0;0;0];
                    self.figureCost = figure;
                    self.figureBoundary = figure;                    
                case 'iter'
                    %obj.storeValue()
                    %obj.plot();

                    self.xIter = [self.xIter, x];
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

    end
end
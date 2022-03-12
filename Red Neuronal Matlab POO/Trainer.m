classdef Trainer < handle

    properties (Access = public)
        isDisplayed
    end
    
    properties (Access = protected) 
       network
          
       costHist
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
            self.network      = s.network;
            self.isDisplayed  = s.isDisplayed;
        end

        function [J,g] = costFunction(self,x,I)
            theta   = x;
            Ibatch  = I;
            net     = self.network;
            net.computeCost(theta,Ibatch)
            J = net.cost;
            g = net.gradient;
        end

        function storeValues(self,x,f,state)
            switch state
                case 'init'
                    self.costHist = [0,0,0];
                    self.figureCost = figure;
                    self.figureBoundary = figure;
                case 'iter'
                    cV = zeros(1,3);
                    cV(1) = f;
                    cV(2) = self.network.regularization;
                    cV(3) = self.network.loss;
                    self.xIter = [self.xIter, x];
                    self.costHist = [self.costHist;cV];
            end
        end

        function plotMinimization(self,iter)
            nIter = 10;
            if mod(iter,nIter) == 0 && iter ~= 0
                v = 0:nIter:iter;
                figure(self.figureCost)
                plot(v,self.costHist(2:nIter:end,1),'+-r')
                plot(v,self.costHist(2:nIter:end,2),'+-b')
                plot(v,self.costHist(2:nIter:end,3),'+-k')
                legend('Fval','Loss','Regularization')
                xlabel('Iterations')
                ylabel('Function Values')
                drawnow
                if mod(iter,nIter*5) == 0
                    figure(self.figureBoundary);
                    self.network.plotBoundary()
                end
            end
        end    
    end
end
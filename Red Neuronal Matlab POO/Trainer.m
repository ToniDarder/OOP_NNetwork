classdef Trainer < handle

    properties (Access = public)
        
        isDisplayed
        costHist
        optHist
    end
    
    properties (Access = protected) 
       data
       network
       figureCost
       figureOpt
       xIter
       nPlot
    end

    methods (Access = public, Static)

        function self = create(s)
           switch s.type
               case 'SGD'
                   self = SGD(s);
               case 'Fminunc'
                   self = Fminunc(s);
               case 'Nesterov'
                   self = Nesterov(s);
           end
        end
    end

    methods (Access = protected)

        function init(self,s)
            self.network      = s.network;
            self.isDisplayed  = s.isDisplayed;
        end

        function [J,g] = costFunction(self,x,Xb,Yb)
            theta   = x;
            self.network.computeCost(theta,Xb,Yb)
            J = self.network.cost;
            g = self.network.gradient;
        end

        function storeValues(self,x,f,state,opt)
            switch state
                case 'init'
                    self.costHist = [0,0,0];
                    self.optHist = [0,0];      
                    self.figureCost = figure;
                    self.figureOpt = figure;
                case 'iter'
                    cV = zeros(1,3);
                    cV(1) = f;
                    cV(2) = self.network.regularization;
                    cV(3) = self.network.loss;
                    self.xIter = [self.xIter, x];
                    self.costHist = [self.costHist;cV];
                    oV = zeros(1,2);
                    oV(1) = opt.gnorm;
                    oV(2) = opt.epsilon;
                    self.optHist = [self.optHist;oV];
            end
        end

        function plotMinimization(self,iter)
            nIter = self.nPlot;
            v = 0:nIter:iter;
            if iter > 0
            self.plotCostRegErr(v);
            self.plotEpsOpt(v)
            end
            if self.network.data.nFeatures <= 2
                self.network.plotBoundary('contour')
            end
        end  

        function plotCostRegErr (self,v)
            figure(self.figureCost)
            semilogy(v,self.costHist(2:end,1),'+-r',v,self.costHist(2:end,3),'+-b',v,self.costHist(2:end,2),'+-k')
            legend('Fval','Loss','Regularization')
            xlabel('Iterations')
            ylabel('Function Values')
            title('Cost minimization')
            xlim([10,inf])
            drawnow
            hold off
        end

        function plotEpsOpt(self,v)
            figure(self.figureOpt)
            subplot(2,1,1)
            plot(v,self.optHist(2:end,1),'+-k')
            yline(1,'-','Gtol Criteria')
            xlabel('Iterations')
            ylabel('Optimalty criteria')
            title('Gradient norm vs iter')
            xlim([10,inf])
            subplot(2,1,2)
            plot(v,self.optHist(2:end,2),'+-g')
            xlabel('Iterations')
            ylabel('Learning Rate')
            title('Step Size vs Iter')
            xlim([10,inf])
        end
    end
end
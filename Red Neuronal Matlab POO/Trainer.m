classdef Trainer < handle

    properties (Access = public)
        isDisplayed
    end
    
    properties (Access = protected) 
       network          
       costHist
       optHist
       figureCost
       figureOpt
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
            nIter = 10;
            if mod(iter,nIter) == 0 && iter ~= 0
                v = 0:nIter:iter;
                self.plotCostRegErr(v,nIter);
                self.plotEpsOpt(v,nIter)
            end
            if mod(iter,nIter*5) == 0
                self.network.plotBoundary('contour')
            end
        end  

        function plotCostRegErr (self,v,nIter)
            figure(self.figureCost)
            hold on
            plot(v,self.costHist(2:nIter:end,1),'+-r')
            plot(v,self.costHist(2:nIter:end,2),'+-b')
            plot(v,self.costHist(2:nIter:end,3),'+-k')
            legend('Fval','Loss','Regularization')
            xlabel('Iterations')
            ylabel('Function Values')
            title('Cost minimization')
            xlim([10,inf])
            drawnow
            hold off
        end

        function plotEpsOpt(self,v,nIter)
            figure(self.figureOpt)
            subplot(2,1,1)
            plot(v,self.optHist(2:nIter:end,1),'+-k')
            yline(1,'-','Gtol Criteria')
            xlabel('Iterations')
            ylabel('Optimalty criteria')
            title('Gradient norm vs iter')
            xlim([10,inf])
            subplot(2,1,2)
            plot(v,self.optHist(2:nIter:end,2),'+-g')
            xlabel('Iterations')
            ylabel('Learning Rate')
            title('Step Size vs Iter')
            xlim([10,inf])
        end
    end
end
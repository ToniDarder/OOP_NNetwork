classdef Network < handle
 
    properties (GetAccess = public, SetAccess = private)
       W
       b
       theta0
       theta
       neuronsPerLayer
       nLayers
       lambda
       data
       cost
       gradient
       regularization
       loss
    end
    
   properties (Access = private)
       propagator
       plotter
   end

   methods (Access = public)

       function self = Network(s)
           self.init(s);
           self.computeInitialTheta();
       end

       function computeInitialTheta(self)
           nPL    = self.neuronsPerLayer;
           nLayer = self.nLayers;
           nW     = 0;
           nb     = 0;
           for i = 2:nLayer
                nW = nW + nPL(i-1)*nPL(i);
                nb = nb + nPL(i);
           end
           self.W = zeros(1,nW);
           self.b = zeros(1,nb);
           th0 = [self.W,self.b];
           self.theta0 = (th0+rand([1,nW+nb])+10^-2)*10^(0);
       end

       function h = getOutput(self,X)
            h = self.propagator.compute_last_H(X,self.theta_m);
       end
       
       function computeCost(self,theta,I)
           self.theta = theta;
           [self.W,self.b] = self.propagator.theta_to_Wb(self.theta);
           [J,grad] = self.propagator.propagate(theta,I); 
           self.computeLoss();
           self.computeRegularization();
           self.cost = J; 
           self.gradient = grad;
       end 

       function computeLoss(self)
           l = self.propagator.loss;
           self.loss = l;
        end

        function computeRegularization(self)
            r = self.propagator.regularization;
            l = self.lambda;
            self.regularization = r*l;
        end

        function plotBoundary(self,type) 
           self.plotter.plotBoundary(self.W,self.b,type);
       end

       function plotConections(self)
           self.plotter.plotNetworkStatus(self.W);
       end

       function plotConfusionMatrix(self)
           self.plotter.drawConfusionMat(self.W,self.b);
       end

       function updateHyperparameter(self,h)
           switch h.type
               case 'lambda'
                    self.lambda = h.value;
                    self.propagator.lambda = self.lambda;
           end
       end
   end

   methods (Access = private)

       function init(self,s)
           self.neuronsPerLayer = s.Net_Structure;
           self.nLayers = length(s.Net_Structure);
           self.data = s.data;
           self.lambda = s.lambda;
           s.self = self;
           self.propagator = Propagator(s);
           self.plotter = Plotter(s,self.propagator);
       end  
   end   
end

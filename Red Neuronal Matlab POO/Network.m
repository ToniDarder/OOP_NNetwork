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

   properties (Dependent)
       theta_m
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
           self.theta0 = th0+rand([1,nW+nb])+10^-4;
       end

       function h = getOutput(self,X)
            h = self.propagator.compute_last_H(X,self.theta_m);
       end
       
       function computeCost(self,theta,I) 
           self.theta = theta;
           [self.W,self.b] = self.propagator.thetavec_to_thetamat(self.theta);
           [J,grad] = self.propagator.propagate(theta,I); 
           self.computeLoss();
           self.computeRegularization();
           self.cost = J; 
           self.gradient = grad;
       end 

       function computeLoss(self)
           l = self.propagator.loss;
           self.loss = extractdata(l);
        end

        function computeRegularization(self)
            r = self.propagator.regularization;
            l = self.lambda;
            self.regularization = extractdata(r*l);
        end

        function plotBoundary(self) 
           self.plotter.plotBoundary(self.W,self.b);
       end

       function plotConections(self)
           self.plotter.plotNetworkStatus(self.W);
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

   methods
       function value = get.theta_m(self)
           value = self.propagator.thetavec_to_thetamat(self.theta);
       end
   end
   
end

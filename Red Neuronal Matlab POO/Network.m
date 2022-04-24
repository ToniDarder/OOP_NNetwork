classdef Network < handle
 
    properties (GetAccess = public, SetAccess = private)
       thetavec
       neuronsPerLayer
       nLayers
       lambda
       data
       cost
       gradient
       regularization
       loss
    end

    properties (Dependent)
        layer
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
           self.thetavec = (rand([1,nW+nb])*2 - 1 +10^-2);
       end

       function h = getOutput(self,X)
            h = self.propagator.compute_last_H(X,self.theta_m);
       end
       
       function computeCost(self,theta,I)
           self.thetavec = theta;
           [J,grad] = self.propagator.propagate(self.layer,I); 
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
           self.plotter.plotBoundary(self.layer,type);
       end

       function plotConections(self)
           self.plotter.plotNetworkStatus(self.layer);
       end

       function plotConfusionMatrix(self)
           self.plotter.drawConfusionMat(self.layer);
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
       function value = get.layer(self)
            nPL = self.neuronsPerLayer;
            last = 1;
            value = cell(self.nLayers-1,1);
            for i = 2:self.nLayers
                aux = nPL(i)*nPL(i-1) + nPL(i);
                next = last + aux;
                theta_i = self.thetavec(last:next-1);
                value{i-1} = Layer(theta_i,nPL(i-1),nPL(i));
                last = next;
            end
       end
   end
end

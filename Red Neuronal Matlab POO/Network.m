classdef Network < handle
 
    properties (GetAccess = public, SetAccess = private)
       data
       thetavec
       neuronsPerLayer
       nLayers
       lambda
       Costtype
       HUtype
       OUtype
       cost
       regularization
       loss
       gradient
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
           th     = [];
           for i = 2:self.nLayers
                if i ~= self.nLayers
                    b = zeros([1,nPL(i)]) + 0.1;
                else
                    b = zeros([1,nPL(i)]) + 1/nPL(i);
                end
                u = (6/(nPL(i-1)+nPL(i)))^0.5;
                W = (unifrnd(-u,u,[1,nPL(i-1)*nPL(i)]));
                th = [th,W,b];
           end      
           self.thetavec = th;
       end

       function h = getOutput(self,X)
            h = self.propagator.compute_last_H(X);
       end
       
       function computeCost(self,theta,Xb,Yb)
           self.thetavec = theta;
           [J,grad] = self.propagator.propagate(self.layer,Xb,Yb); 
           self.loss = self.propagator.loss;
           l = self.lambda;
           self.regularization = l*self.propagator.regularization;
           self.cost = J; 
           self.gradient = grad;
       end 

       function plotBoundary(self,type) 
           self.plotter.plotBoundary(type);
       end

       function plotConections(self)
           self.plotter.plotNetworkStatus();
       end

       function plotConfusionMatrix(self)
           self.plotter.drawConfusionMat();
       end

       function updateHyperparameter(self,h)
           switch h.type
               case 'lambda'
                    self.lambda = h.value;
                    self.propagator.lambda = self.lambda;
                    self.computeInitialTheta();
           end
       end
   end

   methods (Access = private)

       function init(self,s)
           self.neuronsPerLayer = s.Net_Structure;
           self.nLayers = length(s.Net_Structure);
           self.data = s.data;
           self.lambda = s.lambda;
           self.Costtype = s.costFunction;
           self.HUtype = s.activationFunction;
           self.OUtype = s.outputFunction;
           s.net = self;
           self.propagator = Propagator(s);
           self.plotter = Plotter(self);
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

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

       function obj = Network(s)
           obj.init(s);
           obj.computeInitialTheta();
       end

       function computeInitialTheta(obj)
           nPL    = obj.neuronsPerLayer;
           nLayer = obj.nLayers;
           nW     = 0;
           nb     = 0;
           for i = 2:nLayer
                nW = nW + nPL(i-1)*nPL(i);
                nb = nb + nPL(i);
           end
           obj.W = zeros(1,nW);
           obj.b = zeros(1,nb);
           th0 = [obj.W,obj.b];
           obj.theta0 = th0+rand([1,nW+nb])+10^-4;
       end

       function h = getOutput(obj,X)
            h = obj.propagator.compute_last_H(X,obj.theta_m);
       end
       
       function computeCost(obj,theta,I) 
           obj.theta = theta;
           [obj.W,obj.b] = obj.propagator.thetavec_to_thetamat(obj.theta);
           [J,grad] = obj.propagator.propagate(theta,I); 
           obj.computeLoss();
           obj.computeRegularization();
           obj.cost = J; 
           obj.gradient = grad;
       end 

       function computeLoss(obj)
           l = obj.propagator.loss;
           obj.loss = extractdata(l);
        end

        function computeRegularization(obj)
            r = obj.propagator.regularization;
            l = obj.lambda;
            obj.regularization = extractdata(r*l);
        end

       function plotBoundary(obj) 
           obj.plotter.plotBoundary(obj.W,obj.b);
       end

       function plotConections(obj)
           obj.plotter.plotNetworkStatus(obj.W);
       end

       function updateHyperparameter(obj,h)
           switch h.type
               case 'lambda'
                    obj.lambda = h.value;
                    obj.propagator.lambda = obj.lambda;
           end
       end
   end

   methods (Access = private)

       function init(obj,s)
           obj.neuronsPerLayer = s.Net_Structure;
           obj.nLayers = length(s.Net_Structure);
           obj.data = s.data;
           obj.lambda = s.lambda;
           s.obj = obj;
           obj.propagator = Propagator(s);
           obj.plotter = Plotter(s,obj.propagator);
       end  
   end

   methods
       function value = get.theta_m(obj)
           value = obj.propagator.thetavec_to_thetamat(obj.theta);
       end
   end
   
end

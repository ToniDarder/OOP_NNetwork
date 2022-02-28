classdef Network < handle
 
    properties (GetAccess = public, SetAccess = private)
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
           nF     = obj.data.nFeatures;
           nLayer = obj.nLayers;
           nTheta = nF*nPL(1);
           for i = 2:nLayer
                nTheta = nTheta + nPL(i-1)*nPL(i);
           end
           obj.theta0 = zeros(1,nTheta);
       end

       function h = getOutput(obj,X)
            h = obj.propagator.compute_last_H(X,obj.theta_m);
       end
       
       function computeCost(obj,theta) 
           %Here
           obj.theta = theta;
           [J,grad] = obj.propagator.propagate(theta); 
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

       function plotBoundary(obj,nFigure) 
           obj.plotter.plotBoundary(nFigure,obj.theta_m);
       end

       function plotConections(obj)
           obj.plotter.plotNetworkStatus(obj.theta_m);
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

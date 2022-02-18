classdef Network < handle
 
   properties (Access = public)  
       theta
       neuronsPerLayer
       nLayers
       lambda
       data
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
           obj.init(s)
       end

       function h = getOutput(obj,X)
            h = obj.propagator.compute_last_H(X,obj.theta_m);
       end
       
       function [J,gradient] = computeCost(obj,theta)          
            [J,gradient] = obj.propagator.propagate(theta); 
       end 

       function J = computeLoss(obj,theta)
           J = obj.propagator.computeLossFunction(theta);
        end

        function r = computeRegularization(obj,theta)
            r = obj.propagator.computeRegularizationTerm(theta);
        end

       function plotBoundary(obj,nFigure) 
           obj.plotter.plotBoundary(nFigure,obj.theta_m);
       end

       function plotNetwork(obj)
           obj.plotter.plotNetworkStatus();
       end
   end

   methods (Access = private)

       function init(obj,s)
           obj.neuronsPerLayer = s.Net_Structure;
           obj.nLayers = length(s.Net_Structure);
           obj.data = s.data;
           obj.lambda = s.lambda;
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

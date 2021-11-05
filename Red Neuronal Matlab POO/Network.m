%% Class for Network

classdef Network < handle
   properties (Access = public)  
      thetaOpt
      thetaOpt_mat
      sizes
   end
   properties (Access = private)
      num_layers
      theta0
   end

   methods (Access = public)
       function obj = Network(Net_Structure)
           obj.sizes = Net_Structure;
           obj.num_layers = length(Net_Structure);
       end

       function h = compute_last_H(obj,X)
            thetamat = obj.thetavec_to_thetamat(obj.thetaOpt);
            h = hypothesisFunction(X,thetamat.(thetamat.name{1}));
            g = sigmoid(h);
            for i = 2:obj.num_layers
                h = hypothesisFunction(g,thetamat.(thetamat.name{i}));
                g = sigmoid(h);
            end
       end
   end

   methods (Access = private)
       function thetamat = thetavec_to_thetamat(obj,thetavec)
           thetamat.name = genvarname(repmat({'l'},1,obj.num_layers),'l');
           last = obj.sizes(1)*obj.sizes(end);
           aux = reshape(thetavec(1:last),[obj.sizes(end),obj.sizes(1)]);
           thetamat.(thetamat.name{1}) = aux;
            for i = 2:obj.num_layers
                aux = reshape(thetavec(last+1:(last+obj.sizes(i)*obj.sizes(i-1))),[obj.sizes(i-1),obj.sizes(i)]);
                thetamat.(thetamat.name{i}) = aux;
            end
       end       
   end
   
end
%% Class for Network

classdef Network < handle
   properties (Access = public)  
      thetaOpt
      thetaOpt_m
      sizes
      transfer_fcn
   end
   properties (Access = private)
      num_layers
   end

   methods (Access = public)
       function obj = Network(Net_Structure,fcn)
           obj.sizes = Net_Structure;
           obj.num_layers = length(Net_Structure);
           obj.transfer_fcn = fcn;
       end

       function h = compute_last_H(obj,X)
            %thetamat = obj.thetavec_to_thetamat(obj.thetaOpt);
            tOpt = obj.thetaOpt_m;
            h = obj.hypothesisfunction(X,tOpt.(tOpt.name{1}));
            g = sigmoid(h);
            for i = 2:obj.num_layers
                h = obj.hypothesisfunction(g,tOpt.(tOpt.name{i}));
                g = sigmoid(h);
            end
       end

       function h = hypothesisfunction(obj,X,theta)
           if strcmp(obj.transfer_fcn,'linear')
                h = X*theta;
           elseif strcmp(obj.transfer_fcn,'linear+1')
                h = X*theta + 1;
           end
       end
   end

   methods (Access = private)
       function t_m = thetavec_to_thetamat(obj,thetavec)
           t_m.name = genvarname(repmat({'l'},1,obj.num_layers),'l');
           last = obj.sizes(1)*obj.sizes(end);
           aux = reshape(thetavec(1:last),[obj.sizes(end),obj.sizes(1)]);
           t_m.(t_m.name{1}) = aux;
            for i = 2:obj.num_layers
                aux = reshape(thetavec(last+1:(last+obj.sizes(i)*obj.sizes(i-1))),[obj.sizes(i-1),obj.sizes(i)]);
                t_m.(t_m.name{i}) = aux;
            end
       end       
   end
   
end
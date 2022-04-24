classdef Propagator < handle
    
    properties (Access = public)
        lambda
    end
    
    properties (GetAccess = public, SetAccess = private)
        regularization
        loss
        cost
        gradient
    end

    properties (Access = private)
        data
        neuronsPerLayer
        nLayers
        a_fcn
        network
        nData
        propType
        costFCNtype
        activationFCNtype
        batchSize
    end
    
    methods (Access = public)

        function self = Propagator(init)
            self.data = init.data;
            self.lambda = init.lambda;
            self.neuronsPerLayer = init.Net_Structure;
            self.nLayers = length(init.Net_Structure);
            self.nData = length(init.data.Ytrain);
            self.network = init.self;
            self.propType = init.prop;
            self.costFCNtype = init.costFunction;
            self.activationFCNtype = init.activationFunction;
        end

        function [J,gradient] = propagate(self,layer,I)
            if I == 0
                self.batchSize = length(self.data.Ytrain);
            else
                self.batchSize = I;
            end
            switch self.propType
                case 'autodiff'
                    th = dlarray(layer);
                    j = @(theta) self.f_adiff(layer);
                    [j_e,g_e] = dlfeval(j,th);
                    gradient = extractdata(g_e);   
                    J = extractdata(j_e); 
                    self.cost = extractdata(self.cost);
                    self.loss = extractdata(self.loss);
                    self.regularization = extractdata(self.regularization);
                case 'backprop'
                    self.forwardprop(layer);
                    J = self.cost;
                    gradient = self.backprop(layer);
            end
        end       

       function g = compute_last_H(self,X,layer)
            h = self.hypothesisfunction(X,layer{1}.W,layer{1}.b);
            [g,~] = self.actFCN(h,1);
            for i = 2:self.nLayers-1
                h = self.hypothesisfunction(g,layer{i}.W,layer{i}.b);
                [g,~] = self.actFCN(h,i);
            end
       end      
    end

    methods (Access = private)

        % ARREGLAR AUTODIFF
       function [J,grad] = f_adiff(self,layer)         
           self.forwardprop(layer);
           J = self.cost;
           grad = dlgradient(J,layer);
       end

       function forwardprop(self,layer)
           self.computeLossFunction(layer);
           self.computeRegularizationTerm(layer);
           c = self.loss;
           r = self.regularization;
           l = self.lambda;
           self.cost = c + l*r;
       end

       function computeLossFunction(self,layer)
           self.computeActivationFCN(layer);
           a = self.a_fcn;
           I = self.batchSize;
           y =  self.data.Ytrain(1:I,:);
           [J,~] = self.costFunction(y,a);
           self.loss = J;
       end

       function computeRegularizationTerm(self,layer)
           I = self.batchSize;
           nLy = self.nLayers;
           s = 0;
           for i = 2:nLy
                s = s + layer{i-1}.theta*layer{i-1}.theta';
           end
           r = 0.5/I*s;
           self.regularization = r;
       end

       function computeActivationFCN(self,layer)
           x  = self.data.Xtrain;
           I = self.batchSize;
           nLy = self.nLayers;
           a = cell(nLy,1);
           a{1} = x(1:I,:);
           for i = 2:nLy
               g_prev = a{i-1};
               h = self.hypothesisfunction(g_prev,layer{i-1}.W,layer{i-1}.b);
               [g,~] = self.actFCN(h,i);
               a{i} = g;
           end
           self.a_fcn = a;
       end  

       function grad = backprop(self,layer)
           a = self.a_fcn;
           y = self.data.Ytrain;
           nPl = self.neuronsPerLayer;
           nLy = self.nLayers;
           I = self.batchSize;
           delta = cell(nLy,1);
           gradW = cell(nLy-1,1);
           gradb = cell(nLy-1,1);
           for k = nLy:-1:2    
               [~,a_der] = self.actFCN(a{k},k); 
               if k == nLy
                   [~,t1] = self.costFunction(y(1:I,:),a);  
                   delta{k} = t1.*a_der;
               else                    
                   delta{k} = (layer{k}.W*delta{k+1}')'.*a_der;
               end
               gradW{k-1} = (1/I)*(a{k-1}'*delta{k} + self.lambda*layer{k-1}.W);
               gradb{k-1} = (1/I)*(sum(delta{k},1) + self.lambda*layer{k-1}.b);
           end
           grad = [];
           for i = 2:nLy
               aux = [reshape(gradW{i-1},[1,nPl(i-1)*nPl(i)]),gradb{i-1}];
               grad = [grad,aux];
           end
       end

       function [J,gc] = costFunction(self,y,a)
            type = self.costFCNtype;
            J = 0;
            yp = a{end};
            switch type
                case '-loglikelihood-softmax'
                    c = sum(y.*-log(yp),2);
                    J = mean(c);                        
                    gc = yp-y;
                case '-loglikelihood-sigmoid'
                    c = sum((1-y).*-log(1-yp) + y.*-log(yp),2);
                    J = mean(c);                        
                    gc = (yp-y)./(yp.*(1-yp));
                case 'l2'
                    c = ((yp-y).^2);
                    J = sum(mean(c,1));
                    gc = (yp-y);
                otherwise
                    msg = [type,' is not a valid cost function'];
                    error(msg)
            end
       end

       function [g,g_der] = actFCN(self,z,k)
            % OJO amb com estic usant les derivades
            if k == self.nLayers
                type = 'softmax';
            else
                type = self.activationFCNtype;
            end
            switch type 
                case 'sigmoid'
                    g = 1./(1+exp(-z));
                    g_der = z.*(1-z);
                case 'ReLU'
                    g = gt(z,0).*z;
                    g_der = gt(z,0);
                case 'tanh'
                    g = (exp(z)-exp(-z))./(exp(z)+exp(-z));
                    g_der = (1-z.^2);
                case 'softmax'
                    g = (exp(z))./(sum(exp(z),2));
                    g_der = 1;                    
                otherwise
                    msg = [type,' is not a valid activation function'];
                    error(msg)
            end
        end
    end
    
    methods (Access = private, Static)      
        function h = hypothesisfunction(X,W,b)
          h = X*W + b;
        end     
    end  
end
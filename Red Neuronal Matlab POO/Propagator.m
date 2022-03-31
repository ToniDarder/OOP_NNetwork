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
        Ibatch
        propType
        costFCNtype
        activationFCNtype
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

        function [J,gradient] = propagate(self,theta,I)
            if I == 0
                self.Ibatch = length(self.data.Ytrain);
            else
                self.Ibatch = I;
            end
            switch self.propType
                case 'autodiff'
                    th = dlarray(theta);
                    j = @(theta) self.f_adiff(theta);
                    [j_e,g_e] = dlfeval(j,th);
                    gradient = extractdata(g_e);   
                    J = extractdata(j_e); 
                    self.cost = extractdata(self.cost);
                    self.loss = extractdata(self.loss);
                    self.regularization = extractdata(self.regularization);
                case 'backprop'
                    self.forwardprop(theta);
                    J = self.cost;
                    gradient = self.backprop(theta);
            end
        end       

       function g = compute_last_H(self,X,W,b)
            h = self.hypothesisfunction(X,W{1},b{1});
            [g,~] = self.actFCN(h,1);
            for i = 2:self.nLayers-1
                h = self.hypothesisfunction(g,W{i},b{i});
                [g,~] = self.actFCN(h,i);
            end
       end

       function [W_m,b_m] = theta_to_Wb(self,thetavec)
           nL = self.nLayers;
           nPL = self.neuronsPerLayer;
           Wvec = thetavec(1:end-sum(nPL(2:end)));
           bvec = thetavec(end-sum(nPL(2:end))+1:end);
           W_m = cell(nL-1,1);
           b_m = cell(nL-1,1);
           lastW = 0;
           lastb = 0;
           for i = 2:nL
               aux = reshape(Wvec(lastW+1:(lastW+nPL(i)*nPL(i-1))),[nPL(i-1),nPL(i)]);
               W_m{i-1} = aux; 
               b_m{i-1} = bvec(lastb+1:lastb+nPL(i));
               lastW = lastW + nPL(i)*nPL(i-1);
               lastb = lastb + nPL(i);
           end
       end        
    end

    methods (Access = private)

       function [J,grad] = f_adiff(self,theta)         
           self.forwardprop(theta);
           J = self.cost;
           grad = dlgradient(J,theta);
       end

       function forwardprop(self,theta)
           self.computeLossFunction(theta);
           self.computeRegularizationTerm(theta);
           c = self.loss;
           r = self.regularization;
           l = self.lambda;
           self.cost = c + l*r;
       end

       function computeLossFunction(self,theta)
           [W,b] = self.theta_to_Wb(theta);
           self.computeActivationFCN(W,b);
           a = self.a_fcn;
           I = self.Ibatch;
           g = a{end}(1:I,:);
           y =  self.data.Ytrain(1:I,:);
           [J,~] = self.costFunction(y,g);
           self.loss = J;
       end

       function computeRegularizationTerm(self,theta)
           nD = self.Ibatch;
           r = 0.5/nD*(theta*theta');
           self.regularization = r;
       end

       function computeActivationFCN(self,W,b)
           x  = self.data.Xtrain;
           I = self.Ibatch;
           nLy = self.nLayers;
           a = cell(nLy,1);
           a{1} = x(1:I,:);
           for i = 2:nLy
               g_prev = a{i-1};
               h = self.hypothesisfunction(g_prev,W{i-1},b{i-1});
               [g,~] = self.actFCN(h,i);
               a{i} = g;
           end
           self.a_fcn = a;
       end  

       function grad = backprop(self,theta)
           [W,~]  = self.theta_to_Wb(theta);
           a = self.a_fcn;
           y = self.data.Ytrain;
           nLy = self.nLayers;
           I = self.Ibatch;
           delta = cell(nLy,1);
           gradW = cell(nLy-1,1);
           gradb = cell(nLy-1,1);
           for k = nLy:-1:2
               [~,a_der] = self.actFCN(a{k},k); 
               if k == nLy
                   [~,t1] = self.costFunction(y(1:I,:),a{end});  
                   delta{k} = t1.*a_der;
               else  
                   delta{k} = (W{k}*delta{k+1}')'.*a_der;
               end
               gradW{k-1} = (1/length(y))*a{k-1}'*delta{k};
               gradb{k-1} = (1/length(y))*sum(delta{k},1);
           end
           grad = self.Wbgrad_to_gradvec(gradW,gradb);
       end

       function grad = Wbgrad_to_gradvec(self,gradW,gradb)
           nLy = self.nLayers;
           nPl = self.neuronsPerLayer;
           bvec = [];
           Wvec = [];
           for i = 1:nLy-1
               bvec = [bvec,gradb{i}];
               aux = reshape(gradW{i},[1,nPl(i)*nPl(i+1)]);
               Wvec = [Wvec,aux];
           end
           grad = [Wvec,bvec];
       end

       function [J,gc] = costFunction(self,y,yp)
            type = self.costFCNtype;
            J = 0;
            switch type
                case '-loglikelihood'
                    for i = 1:size(y,2)
                        c = (1-y(:,i)).*-log(1-yp(:,i)) + y(:,i).*-log(yp(:,i));
                        J = J + mean(c);                        
                    end
                    gc = (1-y)./(1-yp) + y./(-yp);
                case 'l2'
                    for i = 1:size(y,2)
                        c = ((yp(:,i)-y(:,i)).^2);
                        J = J + mean(c);
                    end
                    gc = (yp-y);
                otherwise
                    msg = [type,' is not a valid cost function'];
                    error(msg)
            end
       end

       function [g,g_der] = actFCN(self,z,k)
            % OJO amb com estic usant les derivades
            if k == self.nLayers
                type = 'sigmoid';
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
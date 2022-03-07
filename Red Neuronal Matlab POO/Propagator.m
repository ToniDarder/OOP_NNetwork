classdef Propagator < handle

    properties (GetAccess = public, SetAccess = private)
        regularization
        loss
        cost
        gradient
    end

    properties (Access = private)
        data
        lambda
        neuronsPerLayer
        nLayers
        a_fcn
        network
        nData
        Ibatch
        indexV
    end
    
    methods (Access = public)

     % Here

        function self = Propagator(init)
            self.data = init.data;
            self.lambda = init.lambda;
            self.neuronsPerLayer = init.Net_Structure;
            self.nLayers = length(init.Net_Structure);
            self.nData = length(init.data.Ytrain);
            self.network = init.obj;
        end

        function [J,gradient] = propagate(self,theta,I)
            self.Ibatch = I;
            self.indexV = self.Ibatch;
            th = dlarray(theta);
            j = @(theta) self.f_adiff(theta);
            [j_e,g_e] = dlfeval(j,th);
            gradient = extractdata(g_e);   
            J = extractdata(j_e); 
        end       

       function h = compute_last_H(self,X,th_m)
            h = self.hypothesisfunction(X,th_m{1});
            g = self.sigmoid(h);
            for i = 2:self.nLayers-1
                h = self.hypothesisfunction(g,th_m{i});
                g = self.sigmoid(h);
            end
       end

       function th_m = thetavec_to_thetamat(self,thetavec)
           nL = self.nLayers;
           nPL = self.neuronsPerLayer;
           th_m = cell(nL-1,1);
           last = 0;
           for i = 2:nL
               aux = reshape(thetavec(last+1:(last+nPL(i)*nPL(i-1))),[nPL(i-1),nPL(i)]);
               th_m{i-1} = aux; 
               last = last+nPL(i)*nPL(i-1);
           end
       end        
    end

    methods (Access = private)

       function [J,grad] = f_adiff(self,theta)         
           self.forwardprop(theta);
           J = self.cost;
           grad = dlgradient(J,theta);
           %grad = self.bacwardprop();
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
           th_m = self.thetavec_to_thetamat(theta);
           self.computeActivationFCN(th_m);
           a = self.a_fcn;
           I = self.indexV;
           g = a{end}(1:I,:);
           y =  self.data.Ytrain(1:I,:);
           nLb = self.data.nLabels;
           m = I;
           J = 0;
           for i = 1:nLb
               err1 = (1-y(:,i)).*(-log(1-g(:,i)));
               err0 = y(:,i).*(-log(g(:,i)));
               j = (1/m)*sum(err1+err0);
               J = J + j;
           end
           self.loss = J;
       end

       function computeRegularizationTerm(self,theta)
           nD = self.indexV;
           r = 0.5/nD*(theta*theta');
           self.regularization = r;
       end

       function computeActivationFCN(self,theta)
           x  = self.data.Xtrain;
           I = self.indexV;
           nLy = self.nLayers;
           a = cell(nLy,1);
           a{1} = x(1:I,:);
           for i = 2:nLy
               g_prev = a{i-1};
               h = self.hypothesisfunction(g_prev,theta{i-1});
               g = self.sigmoid(h);
               a{i} = g;
           end
           self.a_fcn = a;
       end  

%%%%%%%%%%% Backward prop %%%%%%%%%%%

%        function grad = backwardprop(self,theta) 
%             th_m = self.thetavec_to_thetamat(theta);
%             a = self.a_fcn;
%             y = self.data.Ytrain;
%             nL = self.nLayers;      
%             delta = create_delta(self,a,y);
%             [grad,last] = self.create_gradient(delta,theta);
%             for i = nL:-1:2
%                 [grad,delta,last] = self.save_gradient_i(grad,th_m,delta,i,last);
%             end
%        end           
% 
%        function delta = create_delta(self)
%             a = self.a_fcn;
%             y = self.data.Ytrain;
%             nLy = self.nLayers;
%             delta = cell(nLy,1);
%             delta{end} = a{end} - y;
%        end
% 
%        function [grad,last] = create_gradient(self,delta,theta)
%            th_m = self.thetavec_to_thetamat(theta);
%            a = self.a_fcn;
%            y = self.data.Ytrain;
%            nTh = length(theta);
%            grad = zeros(1,nTh);   
%            delta_e = delta{end};
%            a_e1 = a{end-1};
%            th_e = th_m{end};
%            grad_e = (1/length(y))*((delta_e'*a_e1)' + self.lambda*th_e);
%            last = size(grad_e,1)*size(grad_e,2);
%            grad((end-last+1):end) = reshape(grad_e,[1,size(grad_e,1)*size(grad_e,2)]);
%            last = nTh - last;
%        end
% 
%        function [grad,delta,last] = save_gradient_i(self,grad,th,delta,i,last)
%             a = self.a_fcn; 
%             y = self.data.Ytrain;
%             th_i = th{i};
%             th_1i = th{i-1};
%             a_i = a{i};
%             a_1i = a{i-1};        
%             delta_i1 = delta{i+1};
%             delta_i = (th_i*delta_i1')'.*(a_i.*(1 - a_i));
%             delta{i} = delta_i;
%             grad_i = (1/length(y))*((delta_i'*a_1i)' + self.lambda*th_1i);
%             grad(last-size(grad_i,1)*size(grad_i,2)+1:last) = reshape(grad_i,[1,size(grad_i,1)*size(grad_i,2)]);
%             last = last - size(grad_i,1)*size(grad_i,2);
%        end
    end
    
    methods (Access = private, Static)      
        function h = hypothesisfunction(X,theta)
          h = X*theta;
        end     

        function g = sigmoid(z)
            g = 1./(1+exp(-z));
        end
    end  
end
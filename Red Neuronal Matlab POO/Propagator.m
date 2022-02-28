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
    end
    
    methods (Access = public)

     % Here

        function self = Propagator(init)
            self.data = init.data;
            self.lambda = init.lambda;
            self.neuronsPerLayer = init.Net_Structure;
            self.nLayers = length(init.Net_Structure);
            self.network = init.obj;
        end

        function [J,gradient] = propagate(self,theta)
            th = dlarray(theta);
            j = @(theta) self.f_adiff(theta);
            [j_e,g_e] = dlfeval(j,th);
            gradient = extractdata(g_e);   
            J = extractdata(j_e); 
        end       

       function h = compute_last_H(self,X,th_m)
            h = self.hypothesisfunction(X,th_m.(th_m.name{1}));
            g = self.sigmoid(h);
            for i = 2:self.nLayers
                h = self.hypothesisfunction(g,th_m.(th_m.name{i}));
                g = self.sigmoid(h);
            end
       end

       function th_m = thetavec_to_thetamat(self,thetavec)
           nL = self.nLayers;
           nPL = self.neuronsPerLayer;
           nF = self.data.nFeatures;
           th_m.name = genvarname(repmat({'l'},1,nL),'l');
           last = nPL(1)*nF;
           aux = reshape(thetavec(1:last),[nF,nPL(1)]);
           th_m.(th_m.name{1}) = aux;
           for i = 2:nL
               aux = reshape(thetavec(last+1:(last+nPL(i)*nPL(i-1))),[nPL(i-1),nPL(i)]);
               th_m.(th_m.name{i}) = aux;
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
           g = a.(a.name{end});
           y =  self.data.Ytrain;
           nPL = self.data.nLabels;
           nD = self.data.nData;
           J = 0;
           for i = 1:nPL
               err1 = (1-y(:,i)).*(-log(1-g(:,i)));
               err0 = y(:,i).*(-log(g(:,i)));
               j = (1/nD)*sum(err1+err0);
               J = J + j;
           end
           self.loss = J;
       end

       function computeRegularizationTerm(self,theta)
           nD = self.data.nData;
           r = 0.5/nD*(theta*theta');
           self.regularization = r;
       end

       function computeActivationFCN(self,theta)
           x  = self.data.Xtrain;
           nL = self.nLayers;
           a.name = genvarname(repmat({'l'},1,nL+1),'l');
           a.(a.name{1}) = x;
           for i = 1:nL
               g_prev = a.(a.name{i});
               h = self.hypothesisfunction(g_prev,theta.(theta.name{i}));
               g = self.sigmoid(h);
               a.(a.name{i+1}) = g;
           end
           self.a_fcn = a;
       end     

       function grad = backwardprop(self,theta) 
            th_m = self.thetavec_to_thetamat(theta);
            a = self.a_fcn;
            y = self.data.Ytrain;
            nL = self.nLayers;      
            delta = create_delta(self,a,y);
            [grad,last] = self.create_gradient(delta,theta);
            for i = nL:-1:2
                [grad,delta,last] = self.save_gradient_i(grad,th_m,delta,i,last);
            end
       end           

       function delta = create_delta(self)
            a = self.a_fcn;
            y = self.data.Ytrain;
            nL = self.nLayers;
            delta.name = genvarname(repmat({'l'},1,nL+1),'l');
            delta.(delta.name{end}) = a.(a.name{end}) - y;
       end

       function [grad,last] = create_gradient(self,delta,theta)
           th_m = self.thetavec_to_thetamat(theta);
           a = self.a_fcn;
           y = self.data.Ytrain;
           nTh = length(theta);
           grad = zeros(1,nTh);   
           delta_e = delta.(delta.name{end});
           a_e1 = a.(a.name{end-1});
           th_e = th_m.(th_m.name{end});
           grad_e = (1/length(y))*((delta_e'*a_e1)' + self.lambda*th_e);
           last = size(grad_e,1)*size(grad_e,2);
           grad((end-last+1):end) = reshape(grad_e,[1,size(grad_e,1)*size(grad_e,2)]);
           last = nTh - last;
       end

       function [grad,delta,last] = save_gradient_i(self,grad,th,delta,i,last)
            a = self.a_fcn; 
            y = self.data.Ytrain;
            th_i = th.(th.name{i});
            th_1i = th.(th.name{i-1});
            a_i = a.(a.name{i});
            a_1i = a.(a.name{i-1});        
            delta_i1 = delta.(delta.name{i+1});
            delta_i = (th_i*delta_i1')'.*(a_i.*(1 - a_i));
            delta.(delta.name{i}) = delta_i;
            grad_i = (1/length(y))*((delta_i'*a_1i)' + self.lambda*th_1i);
            grad(last-size(grad_i,1)*size(grad_i,2)+1:last) = reshape(grad_i,[1,size(grad_i,1)*size(grad_i,2)]);
            last = last - size(grad_i,1)*size(grad_i,2);
       end
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
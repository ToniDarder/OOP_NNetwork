classdef Network < handle
 
   properties (Access = public)  
       theta
       neuronsPerLayer
       nLayers
       lambda
       data
   end

   properties (Access = private)
              
   end

   properties (Dependent)
       theta_m
   end

   methods (Access = public)

       function obj = Network(s)
           obj.init(s)
       end

       function h = compute_last_H(obj,X)
            th_m = obj.theta_m;
            h = obj.hypothesisfunction(X,th_m.(th_m.name{1}));
            g = sigmoid(h);
            for i = 2:obj.nLayers
                h = obj.hypothesisfunction(g,th_m.(th_m.name{i}));
                g = sigmoid(h);
            end
       end
       
       function [j_adiff,g_adiff] = computeCost(obj,theta)
           % [a, J,loss] = obj.forwardprop(x,y,theta,sizes,numfeat);
           % grad = obj.backwardprop(a,y,theta,sizes,numfeat);           
            th = dlarray(theta);
            J = @(theta) obj.f_adiff(theta);
            [j_adiff,g_adiff] = dlfeval(J,th);
            g_adiff = extractdata(g_adiff);   
            j_adiff = extractdata(j_adiff); 
       end

       function [J,a] = computeLossFunction(obj,th)
           th_m = obj.thetavec_to_thetamat(th);
           a = obj.computeActivationFCN(th_m);
           g = a.(a.name{end});
           y =  obj.data.Ytrain;
           nPL = obj.data.nLabels;
           nD = obj.data.nData;
           J = 0;
           for i = 1:nPL
               err1 = (1-y(:,i)).*(-log(1-g(:,i)));
               err0 = y(:,i).*(-log(g(:,i)));
               j = (1/nD)*sum(err1+err0);
               J = J + j;
           end
       end

       function r = computeRegularizationTerm(obj,th)
           nD = obj.data.nData;
           r = 0.5/nD*(th*th');
       end

       function plotBoundary(obj,nFigure) 
           X = obj.data.Xtrain;
           nF = size(X,2);
           nPL = obj.neuronsPerLayer;
           n_pts = 30;
           graphzoom = 10;
           [x1, x2] = createMesh();
           h = obj.computeHeights(x1,x2,n_pts,nF);
           figure(nFigure)
           subplot(3,3,[1,2,3,4,5,6])
           obj.data.plotdata();
           colors = ['b','g','r','c','m','y','k'];
           C = x1*x2';
           for i = 1:nPL(end)
               hold on
               contour(x1,x2,h(:,:,end+1-i)',[0 0],'color',colors(i))
               title('Contour 0')
           end
           for i = 1:nPL(end)
               subplot(3,3,i+6)                           
               surf(x1,x2,h(:,:,end+1-i),C)
               txt = ['3D Surface  ',colors(i)];  
               title(txt)
           end

           function [x1, x2] = createMesh()
               gZ = graphzoom;
               extra_f1 = mean(X(:,2))*gZ;
               extra_f2 = mean(X(:,3))*gZ;
               x1 = linspace(min(X(:,2))-extra_f1,max(X(:,2))+extra_f1,n_pts)';
               x2 = linspace(min(X(:,3))-extra_f2,max(X(:,3))+extra_f2,n_pts)';
           end
       end       
   end

   methods (Access = private)

       function init(obj,s)
           obj.neuronsPerLayer = s.Net_Structure;
           obj.nLayers = length(s.Net_Structure);
           obj.data = s.data;
           obj.lambda = s.lambda;
       end       
        
       function [J,grad] = f_adiff(obj,theta)
           [~,J] = obj.forwardprop(theta);
           grad = dlgradient(J,theta);
       end

       function [a, J] = forwardprop(obj,theta)
           [c,a] = obj.computeLossFunction(theta);
           r = obj.computeRegularizationTerm(theta);
           l = obj.lambda;
           J = c + l*r;
       end
       
       function grad = backwardprop(obj,a,y,theta)  
            nL = obj.nLayers;
            th_m = obj.thetavec_to_thetamat(theta);
            delta = create_delta(obj,a,y);
            [grad,last] = obj.create_gradient(y,a,delta,theta);
            for i = nL:-1:2
                [grad,delta,last] = obj.save_gradient_i(grad,th_m,delta,a,y,i,last);
            end
       end      

       function a = computeActivationFCN(obj,th)
           x  = obj.data.Xtrain;
           nL = obj.nLayers;
           a.name = genvarname(repmat({'l'},1,nL+1),'l');
           a.(a.name{1}) = x;
           for i = 1:nL
               g_prev = a.(a.name{i});
               h = obj.hypothesisfunction(g_prev,th.(th.name{i}));
               g = sigmoid(h);
               a.(a.name{i+1}) = g;
           end
       end

       function delta = create_delta(obj,a,y)
            nL = obj.nLayers;
            delta.name = genvarname(repmat({'l'},1,nL+1),'l');
            delta.(delta.name{end}) = a.(a.name{end}) - y;
       end

       function [grad,last] = create_gradient(obj,y,a,delta,theta)
           th_m = obj.thetavec_to_thetamat(theta);
           nTh = length(theta);
           grad = zeros(1,nTh);   
           delta_e = delta.(delta.name{end});
           a_e1 = a.(a.name{end-1});
           th_e = th_m.(th_m.name{end});
           grad_e = (1/length(y))*((delta_e'*a_e1)' + obj.lambda*th_e);
           last = size(grad_e,1)*size(grad_e,2);
           grad((end-last+1):end) = reshape(grad_e,[1,size(grad_e,1)*size(grad_e,2)]);
           last = nTh - last;
       end

       function [grad,delta,last] = save_gradient_i(obj,grad,th,delta,a,y,i,last)
            th_i = th.(th.name{i});
            th_1i = th.(th.name{i-1});
            a_i = a.(a.name{i});
            a_1i = a.(a.name{i-1});        
            delta_i1 = delta.(delta.name{i+1});

            delta_i = (th_i*delta_i1')'.*(a_i.*(1 - a_i));
            delta.(delta.name{i}) = delta_i;
            grad_i = (1/length(y))*((delta_i'*a_1i)' + obj.lambda*th_1i);
            grad(last-size(grad_i,1)*size(grad_i,2)+1:last) = reshape(grad_i,[1,size(grad_i,1)*size(grad_i,2)]);
            last = last - size(grad_i,1)*size(grad_i,2);
       end

       function h_3D = computeHeights(obj,x1,x2,n_pts,nF)
           nPL = obj.neuronsPerLayer;
           X_test = zeros(n_pts,nF,n_pts);
           h = zeros(n_pts*nPL(end),n_pts);
           h_3D = zeros(n_pts,n_pts,nPL(end));
           for i = 1:n_pts
               x2_aux = ones(n_pts,1)*x2(i);
               b = ones(n_pts,1);
               xdata_test = [b, x1 , x2_aux];
               X_test(:,:,i) = xdata_test;
               h(:,i) = reshape(obj.compute_last_H(X_test(:,:,i)),[n_pts*nPL(end),1]);
           end
           for j = 1:nPL(end)
               h_3D(:,:,j) = h((j-1)*n_pts+1:j*n_pts,:);
           end
       end

       function th_m = thetavec_to_thetamat(obj,thetavec)
           nL = obj.nLayers;
           nPL = obj.neuronsPerLayer;
           nF = obj.data.nFeatures;
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

    methods (Access = private, Static)
       
        function h = hypothesisfunction(X,theta)
          h = X*theta;
        end
        
    end  

    methods

       function value = get.theta_m(obj)
          value = obj.thetavec_to_thetamat(obj.theta);
       end

    end
   
end
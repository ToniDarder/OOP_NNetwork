%% Class for Network

classdef Network < handle
   properties (Access = public)  
      thetaOpt
      thetaOpt_m
      neuronsPerLayer
      transfer_fcn
   end
   properties (Access = private)
      num_layers
   end

   methods (Access = public)
       function obj = Network(Net_Structure,fcn)
           obj.neuronsPerLayer = Net_Structure;
           obj.num_layers = length(Net_Structure);
           obj.transfer_fcn = fcn;
       end

       function h = compute_last_H(obj,X,th_m)
            %thetamat = obj.thetavec_to_thetamat(obj.thetaOpt);
            h = obj.hypothesisfunction(X,th_m.(th_m.name{1}));
            g = sigmoid(h);
            for i = 2:obj.num_layers
                h = obj.hypothesisfunction(g,th_m.(th_m.name{i}));
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

       function plotBoundary(obj,data,th_m) 
           X = data.Xfull;
           nF = size(X,2);
           nPL = obj.neuronsPerLayer;
           n_pts = 30;
           graphzoom = 10;
           [x1, x2] = obj.createMesh(X,n_pts,graphzoom);
           h = obj.computeHeights(x1,x2,n_pts,nF,th_m);
           figure
           subplot(3,3,[1,2,3,4,5,6])
           data.plotdata();
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
       end
   end

   methods (Access = private)
       function th_m = thetavec_to_thetamat(obj,thetavec)
           nL = obj.neuronsPerLayer;
           nF = obj.numFeatures;
           th_m.name = genvarname(repmat({'l'},1,length(nL)),'l');
           last = nL(1)*nF;
           aux = reshape(thetavec(1:last),[nF,nL(1)]);
           th_m.(th_m.name{1}) = aux;
           for i = 2:length(nL)
               aux = reshape(thetavec(last+1:(last+nL(i)*nL(i-1))),[nL(i-1),nL(i)]);
               th_m.(th_m.name{i}) = aux;
               last = last+nL(i)*nL(i-1);
           end
       end 

       function [x1, x2] = createMesh(obj,X,n_pts,gZ)
           extra_f1 = mean(X(:,2))*gZ;
           extra_f2 = mean(X(:,3))*gZ;
           x1 = linspace(min(X(:,2))-extra_f1,max(X(:,2))+extra_f1,n_pts)';
           x2 = linspace(min(X(:,3))-extra_f2,max(X(:,3))+extra_f2,n_pts)';
       end

       function h_3D = computeHeights(obj,x1,x2,n_pts,nF,th_m)
           nPL = obj.neuronsPerLayer;
           X_test = zeros(n_pts,nF,n_pts);
           h = zeros(n_pts*nPL(end),n_pts);
           h_3D = zeros(n_pts,n_pts,nPL(end));
           for i = 1:n_pts
               x2_aux = ones(n_pts,1)*x2(i);
               b = ones(n_pts,1);
               xdata_test = [b, x1 , x2_aux];
               X_test(:,:,i) = xdata_test;
               h(:,i) = reshape(obj.compute_last_H(X_test(:,:,i),th_m),[n_pts*nPL(end),1]);
           end
           for j = 1:nPL(end)
               h_3D(:,:,j) = h((j-1)*n_pts+1:j*n_pts,:);
           end
       end
   end
   
end
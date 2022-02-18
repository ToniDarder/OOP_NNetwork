classdef Plotter < handle
    properties (Access = public)

    end

    properties (Access = private)
        data
        neuronsPerLayer
        propagator
    end

    methods (Access = public)

        function self = Plotter(init,P)
            self.data = init.data;
            self.neuronsPerLayer = init.Net_Structure;
            self.propagator = P;
        end

        function plotBoundary(self,nFigure,th_m) 
           X = self.data.Xtrain;
           nF = size(X,2);
           nPL = self.neuronsPerLayer;
           n_pts = 30;
           graphzoom = 10;
           [x1, x2] = createMesh();
           h = self.computeHeights(x1,x2,n_pts,nF,th_m);
           figure(nFigure)
           subplot(3,3,[1,2,3,4,5,6])
           self.data.plotdata();
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

        function plotNetworkStatus(self)
            NPL = self.neuronsPerLayer;
            numLayers = length(NPL);
            x_sep = 40;
            y_sep = 30;
            figure
            xlim([-20 100])
            ylim([-20 max(NPL)*y_sep-10])
            hold on
            for i = 1:numLayers
                for j = 1:NPL(i)
                    self.plotcircle((i-1)*x_sep,(j-1)*y_sep);
                end
            end
            hold off
        end
    end

    methods (Access = private)
        function h_3D = computeHeights(self,x1,x2,n_pts,nF,th_m)
           nPL = self.neuronsPerLayer;
           X_test = zeros(n_pts,nF,n_pts);
           h = zeros(n_pts*nPL(end),n_pts);
           h_3D = zeros(n_pts,n_pts,nPL(end));
           for i = 1:n_pts
               x2_aux = ones(n_pts,1)*x2(i);
               b = ones(n_pts,1);
               xdata_test = [b, x1 , x2_aux];
               X_test(:,:,i) = xdata_test;
               h(:,i) = reshape(self.propagator.compute_last_H(X_test(:,:,i),th_m),[n_pts*nPL(end),1]);
           end
           for j = 1:nPL(end)
               h_3D(:,:,j) = h((j-1)*n_pts+1:j*n_pts,:);
           end
       end 
    end

    methods (Static)
        function h = plotcircle(cx,cy)
            r = 10;
            th = 0:pi/100:2*pi;
            xunit = r * cos(th) + cx;
            yunit = r * sin(th) + cy;
            h = plot(xunit, yunit);
        end
    end
end
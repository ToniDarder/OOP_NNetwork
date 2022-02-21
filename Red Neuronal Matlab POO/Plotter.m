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

        function plotNetworkStatus(self,th_m)      
            NPL = self.neuronsPerLayer;
            numLayers = length(NPL);
            neurons = cell(max(NPL),numLayers);
            for i = 1:numLayers
                if i == 1
                    maxTH = max(th_m.(th_m.name{i})(:));
                else
                    if maxTH < max(th_m.(th_m.name{i})(:))
                        maxTH = max(th_m.(th_m.name{i})(:));
                    end
                end
            end
            x_sep = 50;
            y_sep = 30;
            figure
            xlim([-20 numLayers*x_sep-10])
            ylim([-20 max(NPL)*y_sep+10])
            set(gca,'XTick',[], 'YTick', [])
            box on
            hold on
            for i = 1:numLayers
                for j = 1:NPL(i)
                    if i == 1
                        color = [1 0 0];
                    elseif i == numLayers
                        color = [0 0.45 0.74];
                    else
                        color = [1 .67 .14];
                    end
                    prop.cx = (i-1)*x_sep;
                    prop.cy = (j-1)*y_sep;
                    prop.color = color;
                    prop.r = 10;
                    circlei = Circle(prop);
                    neurons{i,j} = circlei;
                    circlei.plot();
                end
            end
            for i = 1:numLayers-1
                for j = 1:NPL(i)
                    neuronb = neurons{i,j};
                    for k = 1:NPL(i+1)
                        neuronf = neurons{i+1,k};
                        wth = abs(th_m.(th_m.name{i+1})(j,k)/maxTH);
                        lw = 3*wth;
                        %linecolor = [sin(wth*pi/2),.5,cos(wth*pi/2)];
                        linecolor = [0 , 0, 1];
                        line([neuronb.fx,neuronf.bx],[neuronb.fy,neuronf.by],'Color',linecolor,'LineWidth',lw)
                    end
                end
            end
%             for i = 1:NPL(1)
%                 hold on
%                 neuroni = neurons{i,1};
%                 v1 = [neuroni.bx-10,neuroni.by];
%                 v2 = [neuroni.bx,neuroni.by];
%                 dv = v2 - v1;
%                 quiver(v1(1),v1(2),dv(1),dv(2),0)
%             end
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
end
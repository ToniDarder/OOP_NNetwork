classdef Plotter < handle
    properties (Access = public)

    end

    properties (Access = private)
        data
        neuronsPerLayer
        propagator
    end

    methods (Access = public)

        %HERE

        function self = Plotter(init,P)
            self.data = init.data;
            self.neuronsPerLayer = init.Net_Structure;
            self.propagator = P;
        end

        function plotBoundary(self,W,b,type) 
           X = self.data.Xtrain;
           nF = size(X,2);
           nPL = self.neuronsPerLayer;
           n_pts = 30;
           graphzoom = 1.5;
           x = createMesh();
           h = self.computeHeights(x(:,1),x(:,2),n_pts,nF,W,b);
           figure(10)
           clf(10)     
           colorsc = ['r','g','b','c','m','y','k'];
           colorsc = fliplr(colorsc(1:size(self.data.Ytrain,2)));
           switch type
               case 'contour'
                   for i = 1:nPL(end)
                       hold on
                       contour(x(:,1),x(:,2),h(:,:,i)',[0.5,0.5],'color',colorsc(i))
                   end
               case 'filled'
                   im = cell(size(self.data.Ytrain,2),1);
                   mymap = colormaps();
                   for i = 1:nPL(end)
                       hold on
                       h(i) = axes;
                       im{i} = imagesc(x(:,1),x(:,2),h(:,:,end+1-i)');
                       im{i}.AlphaData = .5;
                       colormap(h(i),mymap{i})
                       if i > 1
                           set(h(i),'color','none','visible','off')
                       end
                       set(h(i),'ydir','normal');
                   end
                   linkaxes(h)
           end  
           hold on
           title('Contour 0')
           self.data.plotdata();
           hold off

           function x = createMesh()
               gZ = graphzoom;
               extra_f1 = mean(X(:,1))*gZ;
               extra_f2 = mean(X(:,2))*gZ;
               x1 = linspace(min(X(:,1))-extra_f1,max(X(:,1))+extra_f1,n_pts)';
               x2 = linspace(min(X(:,2))-extra_f2,max(X(:,2))+extra_f2,n_pts)';
               x = [x1,x2];
           end
           function mymap = colormaps()
               colors = [0,0,1;   % b
                         0,1,0;   % g
                         1,0,0;   % r
                         0,1,1;   % c
                         1,0,1;   % m
                         1,1,0;   % y
                         0,0,0];  % k
               colors = 0.45*colors; 
               mymap = cell(size(self.data.Ytrain,2),1);
               n = 100;
               grey = linspace(1,1,n/2)';
               w = [grey,grey,grey];
               nLb = size(self.data.Ytrain,2);     
               for k = 1:nLb
                   mymap{k} = zeros(n/2,3);
                   mymap{k}(1:n/2,:) = w;
                   r = linspace(1,colors(k,1),n/2)';
                   g = linspace(1,colors(k,2),n/2)';
                   bl = linspace(1,colors(k,3),n/2)';
                   mymap{k}(51:100,:) = [r,g,bl];
               end
           end
        end

        function plotNetworkStatus(self,W)      
            NPL = self.neuronsPerLayer;
            nLy = length(NPL);
            neurons = cell(max(NPL),nLy);
            for i = 1:nLy-1
                if i == 1
                    maxTH = max(W{i}(:));
                else
                    if maxTH < max(W{i}(:))
                        maxTH = max(W{i}(:));
                    end
                end
            end
            x_sep = 50;
            y_sep = 30;
            figure
            xlim([-20 nLy*x_sep-10])
            ylim([-20 max(NPL)*y_sep+10])
            set(gca,'XTick',[], 'YTick', [])
            box on
            hold on
            for i = 1:nLy
                for j = 1:NPL(i)
                    if i == 1
                        color = [1 0 0];
                    elseif i == nLy
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
            r = [linspace(1,1,50)',linspace(0.75,0,50)',linspace(0.75,0,50)'];
            b = [linspace(0,0.75,50)',linspace(0,0.75,50)',linspace(1,1,50)'];
            rgb = [b;r];
            for i = 1:nLy-1
                for j = 1:NPL(i)
                    neuronb = neurons{i,j};
                    for k = 1:NPL(i+1)
                        neuronf = neurons{i+1,k};
                        wth = abs(W{i}(j,k)/maxTH);
                        lw = 3*wth;
                        idx = round(wth*100);
                        if idx == 0
                            idx = 1;
                        end
                        linecolor = rgb(idx,:);
                        %linecolor = [0 , 0, 1];
                        line([neuronb.fx,neuronf.bx],[neuronb.fy,neuronf.by],'Color',linecolor,'LineWidth',lw)
                    end
                end
            end
            hold off
        end

        function drawConfusionMat(self,W,bm)
            targets = self.data.Ytest;
            x = self.data.Xtest;
            nPL = self.neuronsPerLayer;
            outputs = self.propagator.compute_last_H(x,W,bm);
            plotconfusion(targets',outputs')
        end
    end

    methods (Access = private)
        function h_3D = computeHeights(self,x1,x2,n_pts,nF,W,bm)
           nPL = self.neuronsPerLayer;
           X_test = zeros(n_pts,nF,n_pts);
           h = zeros(n_pts*nPL(end),n_pts);
           h_3D = zeros(n_pts,n_pts,nPL(end));
           for i = 1:n_pts
               x2_aux = ones(n_pts,1)*x2(i);
               xdata_test = [x1 , x2_aux];
               X_test(:,:,i) = xdata_test;
               h(:,i) = reshape(self.propagator.compute_last_H(X_test(:,:,i),W,bm),[n_pts*nPL(end),1]);
           end
           for j = 1:nPL(end)
               h_3D(:,:,j) = h((j-1)*n_pts+1:j*n_pts,:);
           end
       end 
    end
end
classdef Data < handle

    properties (Access = public)
        Xtrain
        Ytrain       
        Xtest
        Ytest
        nData
        nFeatures
        nLabels
    end

    properties (Access = private)
        Xfullfeat        
        Xdata
        Ydata    
        testPercentage
        polyGrade    
        X
        Y
    end


    methods (Access = public)
        function obj = Data(File_Name,TP,d)
            obj.loadData(File_Name);
            obj.nData = size(obj.Xdata,1);
            obj.testPercentage = TP;
            obj.polyGrade = d;
            obj.nLabels = size(obj.Ydata,2);
            obj.splitdata()
            obj.computefullvars(obj.X,obj.polyGrade)
            obj.nFeatures = size(obj.Xtrain,2);
        end

        function plotdata(obj)
            % Plots all the data labeled by colour in one figure 
            gscatter(obj.Xdata(:,1),obj.Xdata(:,2),obj.Ydata,'bgrcmyk','xo*+.sd')
            xlabel("X3");
            ylabel("X4");
        end

        function var_corrmatrix(obj)
            x = obj.Xfullfeat;
            y = obj.Ydata;
            nf = size(x,2);
            figure            
            t = tiledlayout(nf,nf,'TileSpacing','Compact'); 
            title(t,'Features correlation matrix');
            for i = 1:nf
                for j = 1:nf
                    nexttile((i-1)*nf+j)
                    if i == j
                        histogram(x(:,i))
                    elseif j > i
                        gscatter(x(:,j),x(:,i),y,'bgrcmyk','xo*+.sd',[5 5 5],'off')
                    end
                    if j == 1
                        txt = ['X',num2str(i)];
                        ylabel(txt)
                    end
                    if i == nf
                        txt = ['X',num2str(j)];
                        xlabel(txt)
                    end                    
                end
            end
        end
    end

    methods (Access = private)
        function loadData(obj,FN)
            f = fullfile('Datasets', FN);
            data = load(f);
            feat = input('Features to be used: ');
            x = data(:, feat);
            ydata = data(:, end);
            y = zeros(length(ydata),max(ydata));
            u = unique(ydata);
            for i=1:length(ydata)
                for j = 1:length(u)
                    if ydata(i) == u(j)
                        y(i,j) = 1;
                    end
                end
            end
            obj.Xdata = x;
            obj.Ydata = y;
            obj.Xfullfeat = data(:,1:(end-1));
        end
        
        function splitdata(obj)
            % Shuffles the data and splits data in train and test
            nD = obj.nData;
            TP = obj.testPercentage;
            r = randperm(nD);
            ntest = round(TP/100*nD);
            ntrain = nD - ntest;
            obj.X = obj.Xdata(r(1:ntrain),:);
            obj.Xtest = obj.Xdata(r((ntrain + 1):end),:);
            obj.Y = obj.Ydata(r(1:ntrain),:);
            obj.Ytest = obj.Ydata(r((ntrain + 1):end),:);
        end

        function computefullvars(obj,x,d)
            % Builds a X matrix with more features using lineal combinations
            x1 = x(:,1); 
            x2 = x(:,2);
            cont = 1;
            for g = 0:d
                for a = 0:g
                       Xful(:,cont) = x2.^(a).*x1.^(g-a);
                       cont = cont+1;
                end
            end
            obj.Xtrain = Xful;
            obj.Ytrain = obj.Y;
        end
       
    end
end
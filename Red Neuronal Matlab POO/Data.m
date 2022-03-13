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
        function self = Data(File_Name,TP,d)
            self.init(File_Name,TP,d)
        end

        function plotdata(self)
            gscatter(self.Xdata(:,1),self.Xdata(:,2),self.Ydata,'bgrcmyk','xo*+.sd')
            xlabel("X3");
            ylabel("X4");
        end

        function drawCorrMatrix(self)
            x = self.Xfullfeat;
            y = self.Ydata;
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

        function drawCorrRow(self,k)
            x = self.Xfullfeat;
            y = self.Ydata;
            nf = size(x,2);
            n = ceil(nf^0.5);
            figure            
            t = tiledlayout(n,n,'TileSpacing','Compact');
            txt = ['Correlation with var',num2str(k)];
            title(t,txt);
            for i = 1:nf
                nexttile(i)
                if i == k
                    histogram(x(:,i))
                else
                    gscatter(x(:,k),x(:,i),y,'bgrcmyk','xo*+.sd',[5 5 5],'off')
                end
            end
        end

        function updateHyperparameter(self,h)
           switch h.type
               case 'testPercentage'
                   self.testPercentage = h.value;
                   self.splitdata()
               case 'polyGrade'
                   self.polyGrade = h.value;
                   self.computefullvars(self.X,self.polyGrade);
           end
       end
    end

    methods (Access = private)

        function init(self,File_Name,TP,d)
            self.loadData(File_Name);
            self.nData = size(self.Xdata,1);
            self.testPercentage = TP;
            self.polyGrade = d;
            self.nLabels = size(self.Ydata,2);
            self.splitdata()
            self.computefullvars(self.X,d)
            self.nFeatures = size(self.Xtrain,2);
        end

        function loadData(self,FN)
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
            self.Xdata = x;
            self.Ydata = y;
            self.Xfullfeat = data(:,1:(end-1));
        end
        
        function splitdata(self)
            % Shuffles the data and splits data in train and test
            nD = self.nData;
            TP = self.testPercentage;
            r = randperm(nD);
            ntest = round(TP/100*nD);
            ntrain = nD - ntest;
            self.X = self.Xdata(r(1:ntrain),:);
            self.Xtest = self.Xdata(r((ntrain + 1):end),:);
            self.Y = self.Ydata(r(1:ntrain),:);
            self.Ytest = self.Ydata(r((ntrain + 1):end),:);
        end

        function computefullvars(self,x,d)
            % Builds a X matrix with more features using lineal combinations
            x1 = x(:,1); 
            x2 = x(:,2);
            cont = 1;
            for g = 1:d
                for a = 0:g
                       Xful(:,cont) = x2.^(a).*x1.^(g-a);
                       cont = cont+1;
                end
            end
            self.Xtrain = Xful;
            self.Ytrain = self.Y;
        end 
    end
end
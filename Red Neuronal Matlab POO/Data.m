%% Class for data objects, stores Xdata and Ydata matrices and has methods to Split Train-Test, Compute Xfull...
% The input has to be a matrix .csv/.mat ..., where the columns are the features and the
% last one being the labels for the groups numbered (1,2,3,...)
classdef Data < handle
    properties (Access = public)
        Xfullfeat
        Xdata
        Ydata
        Xfull
        Yfull       
        Xtest
        Ytest
        Num_Experiences
        Num_Features
        TestPercentage
        full_d
        Num_Labels
    end
    properties (Access = private)
        Xtrain
        Ytrain
    end
    methods (Access = public)
        function obj = Data(File_Name,Tper,d)
            obj.loadData(File_Name);
            obj.Num_Experiences = size(obj.Xdata,1);
            obj.Num_Features = size(obj.Xdata,2);
            obj.TestPercentage = Tper;
            obj.full_d = d;
            obj.Num_Labels = size(obj.Ydata,2);
            obj.splitdata()
            obj.computefullvars(obj.Xtrain,obj.full_d)
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
            f = size(x,2);
            figure            
            t = tiledlayout(f,f,'TileSpacing','Compact'); 
            title(t,'Features correlation matrix');
            for i = 1:f
                for j = 1:f
                    nexttile((i-1)*f+j)
                    if i == j
                        
                    elseif j > i
                        gscatter(x(:,j),x(:,i),y,'bgrcmyk','xo*+.sd',[5 5 5],'off')
                    end
                    if j == 1
                        txt = ['X',num2str(i)];
                        ylabel(txt)
                    end
                    if i == f
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
            X = data(:, [3 4]);
            ydata = data(:, end);
            y = zeros(length(ydata),max(ydata));
            c = unique(ydata);
            for i=1:length(ydata)
                for j = 1:length(c)
                    if ydata(i) == c(j)
                        y(i,j) = 1;
                    end
                end
            end
            obj.Xdata = X;
            obj.Ydata = y;
            obj.Xfullfeat = data(:,1:(end-1));
        end
        
        function splitdata(obj)
            % Shuffles the data and splits data in train and test
            Nexp = obj.Num_Experiences;
            Tper = obj.TestPercentage;
            r = randperm(Nexp);
            ntest = round(Tper/100*Nexp);
            ntrain = Nexp - ntest;
            obj.Xtrain = obj.Xdata(r(1:ntrain),:);
            obj.Xtest = obj.Xdata(r((ntrain + 1):end),:);
            obj.Ytrain = obj.Ydata(r(1:ntrain),:);
            obj.Ytest = obj.Ydata(r((ntrain + 1):end),:);
        end

        function computefullvars(obj,X,d)
            % Builds a X matrix with more features using lineal combinations
            X1 = X(:,1); 
            X2 = X(:,2);
            contador = 1;
            for g = 0:d
                for a = 0:g
                       Xful(:,contador) = X2.^(a).*X1.^(g-a);
                       contador = contador+1;
                end
            end
            obj.Xfull = Xful;
            obj.Yfull = obj.Ytrain;
        end
       
    end
end
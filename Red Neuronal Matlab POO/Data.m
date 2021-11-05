%% Class for data objects, stores Xdata and Ydata matrices and has methods to Split Train-Test, Compute Xfull...

classdef Data < handle
    properties (Access = public)
        Xdata
        Ydata
        Num_Experiences
        Num_Features
        TestPercentage
        Xtrain
        Ytrain
        Xtest
        Ytest
        Xfull
        Yfull
    end
    properties (Access = private)

    end
    methods (Access = public)
        function obj = Data(File_Name,TestPercentage)
            obj.loadData(File_Name);
            obj.Num_Experiences = size(obj.Xdata,1);
            obj.Num_Features = size(obj.Xdata,2);
            obj.TestPercentage = TestPercentage;
            obj.splitdata()
            obj.computefullvars(obj.Xtrain,1)
        end

        function splitdata(obj)
            % Shuffles the data and splits data in train and test
            r = randperm(obj.Num_Experiences);
            ntest = round(obj.TestPercentage/100*obj.Num_Experiences);
            ntrain = obj.Num_Experiences-ntest;
            obj.Xtrain = obj.Xdata(r(1:ntrain),:);
            obj.Xtest = obj.Xdata(r((ntrain+1):end),:);
            obj.Ytrain = obj.Ydata(r(1:ntrain),:);
            obj.Ytest = obj.Ydata(r((ntrain+1):end),:);
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

        function plotdata(obj)
            % Plots all the data labeled by colour in one figure 
            gscatter(obj.Xdata(:,1),obj.Xdata(:,2),obj.Ydata,'bgr','xo*')
            xlabel("X3");
            ylabel("X4");
        end
    end

    methods (Access = private)
        function loadData(obj,FN)
            data = load(FN);
            X = data.meas(:, [3 4]);
            ydata = data.meas(:, 5);
            y = zeros(length(ydata),max(ydata));
            for i=1:length(ydata)
                if ydata(i)==1
                    y(i,1)=1;
                elseif ydata(i)==2
                    y(i,2)=1;
                else
                    y(i,3)=1;
                end
            end
            obj.Xdata = X;
            obj.Ydata = y;
        end
       
    end
end
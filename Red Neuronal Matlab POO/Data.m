%% Class for data objects, stores Xdata and Ydata matrices and has methods to Split Train-Test, Compute Xfull...

classdef Data
    properties
        Xdata
        Ydata
        Num_Experiences
        Num_Features
        TestPercentage
    end
    methods
        function obj = Data(File_Name,TestPercentage)
            data = load(File_Name);
            xdata = data.meas(:, [3 4]);
            y = data.meas(:, 5);
            ydata = zeros(length(y),max(y));
            for i = 1:length(y)
                if y(i) == 1
                    ydata(i,1) = 1;
                elseif y(i) == 2
                    ydata(i,2) = 1;
                else
                    ydata(i,3) = 1;
                end
            end
            obj.Xdata = xdata;
            obj.Ydata = ydata;
            obj.Num_Experiences = size(xdata,1);
            obj.Num_Features = size(xdata,2);
            obj.TestPercentage = TestPercentage;
        end
        function [Xtrain,Ytrain,Xtest,Ytest] = SplitData(obj)
            % Shuffles the data and splits data in train and test
            r = randperm(obj.Num_Experiences);
            ntest = round(obj.TestPercentage/100*obj.Num_Experiences);
            ntrain = obj.Num_Experiences-ntest;
            Xtrain = obj.Xdata(r(1:ntrain),:);
            Xtest = obj.Xdata(r((ntrain+1):end),:);
            Ytrain = obj.Ydata(r(1:ntrain),:);
            Ytest = obj.Ydata(r((ntrain+1):end),:);
        end
        function Xfull = ComputeFullX(obj,X,d)
            % Builds a X matrix with more features using lineal combinations
            X1 = X(:,1); 
            X2 = X(:,2);
            contador = 1;
            for g = 0:d
                for a = 0:g
                       Xfull(:,contador) = X2.^(a).*X1.^(g-a);
                       contador = contador+1;
                end
            end
        end
        function PlotData(obj)
            % Plots all the data labeled by colour in one figure 
            gscatter(obj.Xdata(:,2),obj.Xdata(:,3),y,'bgr','xo*')
            xlabel("X3");
            ylabel("X4");
        end
    end
end
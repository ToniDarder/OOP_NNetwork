classdef Data
    properties
        Xdata
        Ydata
        Num_Experiences
        Num_Features
        TestPercentage
    end
    methods
        function [Xtrain,Ytrain,Xtest,Ytest] = SplitData(obj)
            % Shuffles the data and splits data in train and test
            r = randperm(nData);
            ntest = round(obj.TestPercentage/100*obj.Num_Experiences);
            ntrain = obj.Num_Experiences-ntest;

            Xtrain = obj.Xdata(r(1:ntrain),:);
            Xtest = obj.Xdata(r((ntrain+1):end),:);

            Ytrain = obj.Ydata(r(1:ntrain),:);
            Ytest = obj.Ydata(r((ntrain+1):end),:);
        end
        
        function Xfull = ComputeFullX(obj,X,d)
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
            gscatter(obj.Xdata(:,2),obj.Xdata(:,3),y,'bgr','xo*')
            xlabel("X3");
            ylabel("X4");
        end
    end
end
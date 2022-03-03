classdef Fminunc_Optimizer < Trainer
    properties (Access = public)

    end

    properties (Access = private)

    end

    methods(Access = public)

        function self = Fminunc_Optimizer(s)
            self.init(s);
        end

        function train(self)
            opt = self.setSolverOptions();
            x0  = self.network.theta0;
            F = @(theta) self.costFunction(theta);
            fminunc(F,x0,opt); 
        end
    end

    methods (Access = protected)
        function opt = setSolverOptions(self)
            opt = setSolverOptions@Trainer(self);
        end

        function [J,g] = costFunction(self,theta)
            [J,g] = costFunction@Trainer(self,theta);
        end
    end
end
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
            F = @(theta) self.costFunction(theta,0);
            fminunc(F,x0,opt); 
        end
    end

    methods(Access = private)

        function opt = setSolverOptions(self)
           opt = optimoptions(@fminunc);
           opt.SpecifyObjectiveGradient = true;
           opt.Algorithm = 'quasi-newton';
           opt.StepTolerance = 10^-6;
           opt.MaxFunctionEvaluations = 3000;              
           if self.isDisplayed == true
                args = [];
                opt.Display = 'iter';
                opt.CheckGradients = true;
                opt.OutputFcn = @(theta,optimvalues,state)self.myoutput(theta,optimvalues,state,args);
           end
        end 
    end
end
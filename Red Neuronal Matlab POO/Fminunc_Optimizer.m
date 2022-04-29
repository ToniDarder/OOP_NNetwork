classdef Fminunc_Optimizer < Trainer
    properties (Access = public)

    end

    properties (Access = private)
        opt
    end

    methods(Access = public)
        function self = Fminunc_Optimizer(s)
            self.init(s);
            self.opt = self.setSolverOptions(s);
            self.nPlot = s.nPlot;
        end

        function train(self)
            x0  = self.network.thetavec;
            F = @(theta) self.costFunction(theta,0);
            fminunc(F,x0,self.opt); 
        end
    end

    methods(Access = private)

        function opt = setSolverOptions(self,s)
           opt = optimoptions(@fminunc);
           opt.SpecifyObjectiveGradient = true;
           opt.Algorithm = 'quasi-newton';
           opt.OptimalityTolerance = s.optTolerance;
           opt.MaxIterations = s.maxevals*5;
           opt.MaxFunctionEvaluations = s.maxevals; 
           if self.isDisplayed == true
                args = [];
                opt.Display = 'iter';
                opt.CheckGradients = true;
                opt.OutputFcn = @(theta,optimvalues,state)self.myoutput(theta,optimvalues,state,args);
           end
        end 

        function stop = myoutput(self,x,optimvalues,state,args)
            stop = false;
            f = optimvalues.fval;
            opti.epsilon = optimvalues.stepsize;
            opti.gnorm = optimvalues.firstorderopt;
            iter = optimvalues.iteration;
            if iter == 0
                opti.epsilon = 1;
            end
            self.storeValues(x,f,state,opti);
            self.plotMinimization(iter);                                
        end
    end
end
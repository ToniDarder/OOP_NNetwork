classdef Nesterov < SGD

    properties (Access = private)
       alpha
       v
    end

    methods(Access = public)

        function self = Nesterov(s)
            self@SGD(s); 
            self.alpha = s.alpha;
            self.v     = 0;
        end 

        function [x,grad] = step(self,x,e,grad,F,Xb,Yb)
            x_hat    = x + self.alpha*self.v;
            [~,grad] = F(x_hat,Xb,Yb);
            self.v   = self.alpha*self.v - e*grad;
            x        = x + self.v;     
        end     
    end 
end
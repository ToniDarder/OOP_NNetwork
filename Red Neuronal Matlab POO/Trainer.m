%% Trainer class

classdef Trainer < handle
    properties (Access = public)
        lambda
        transfer_fcn
    end

    properties (Access = private)        
        theta0
    end

    methods (Access = public)
        function obj = Trainer(l,fcn)
            obj.lambda = l;
            obj.transfer_fcn = fcn;                   
        end

        function [tOpt_v,tOpt_m] = train(obj,nn,data,showgraph)
           sizes = nn.sizes;
           X = data.Xfull;
           Y = data.Yfull;
           obj.computeinitialtheta(sizes);     
           tOpt_v = obj.solveTheta(X,Y,sizes,showgraph);
           tOpt_m = obj.thetavec_to_thetamat(tOpt_v,sizes);
        end
    end
    

    methods (Access = private)

        function theta = solveTheta(obj,x,y,sizes,dis)
           if dis == true
                options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','PlotFcn',{@optimplotfval},'StepTolerance',10^(-6),'MaxFunEvals',3000,'CheckGradients',true);
           else
                options = optimoptions(@fminunc,'Algorithm','quasi-newton','StepTolerance',10^(-6),'MaxFunEvals',1000);
           end
           F = @(theta) obj.computeCost(x,y,sizes,theta);
           theta = fminunc(F,obj.theta0,options);
        end

        function [J,grad] = computeCost(obj,x,y,sizes,theta)
           [a, J, ~] = obj.forwardprop(x,y,theta,sizes);
            grad = obj.backwardprop(a,y,theta,sizes);
        end

        function [a, J, h] = forwardprop(obj,x,y,theta,sizes)
            thetamat = obj.thetavec_to_thetamat(theta,sizes);
            h = obj.hypothesisfunction(x,thetamat.(thetamat.name{1}));
            g = sigmoid(h);
            a.name = genvarname(repmat({'l'},1,length(sizes)+1),'l');
            a.(a.name{1}) = x;
            a.(a.name{2}) = g; 
            for i = 2:length(sizes)
                h = obj.hypothesisfunction(g,thetamat.(thetamat.name{i}));
                g = sigmoid(h);
                a.(a.name{i+1}) = g; 
            end
            for i = 1:sizes(end)
                J_vec(i) = (1/length(y))*sum((1-y(:,i)).*(-log(1-g(:,i)))+y(:,i).*(-log(g(:,i))));
            end
            J = sum(J_vec) + 0.5/length(y)*obj.lambda*(theta*theta');
       end
       
       function grad = backwardprop(obj,a,y,theta,sizes)
            grad = zeros(1,length(theta));
            thetamat = obj.thetavec_to_thetamat(theta,sizes);
            delta.name = genvarname(repmat({'l'},1,length(sizes)+1),'l');
            delta.(delta.name{end}) = a.(a.name{end}) - y;
            aux = (1/length(y))*(delta.(delta.name{end})'*a.(a.name{end-1}))' + obj.lambda*thetamat.(thetamat.name{end});
            newend = size(aux,1)*size(aux,2);
            grad((end-newend+1):end) = reshape(aux,[1,size(aux,1)*size(aux,2)]);
            newend = length(theta) - newend;
            for i = length(sizes):-1:2
                product = (thetamat.(thetamat.name{i})*delta.(delta.name{i+1})')';
                delta.(delta.name{i}) = product*(a.(a.name{i})'*(1 - a.(a.name{i})));
                product2 = (delta.(delta.name{i})'*a.(a.name{i-1}))';
                aux = (1/length(y))*product2 + obj.lambda*thetamat.(thetamat.name{i-1});
                grad(newend-size(aux,1)*size(aux,2)+1:newend) = reshape(aux,[1,size(aux,1)*size(aux,2)]);
                newend = newend - size(aux,1)*size(aux,2);
            end
       end
       
       function computeinitialtheta(obj,sizes)
           count = sizes(end)*sizes(1);
           for i = 2:length(sizes)
                count = count + sizes(i-1)*sizes(i);
           end
           obj.theta0 = zeros(1,count);
       end
       
       function thetamat = thetavec_to_thetamat(obj,thetavec,sizes)
           thetamat.name = genvarname(repmat({'l'},1,length(sizes)),'l');
           last = sizes(1)*sizes(end);
           aux = reshape(thetavec(1:last),[sizes(end),sizes(1)]);
           thetamat.(thetamat.name{1}) = aux;
           for i = 2:length(sizes)
               aux = reshape(thetavec(last+1:(last+sizes(i)*sizes(i-1))),[sizes(i-1),sizes(i)]);
               thetamat.(thetamat.name{i}) = aux;
           end
       end  

       function a = save_a(obj,a,n,s)
           if n == 1
                a.(a.name{n}) = x;
           else
                a.(a.name{n}) = g;
           end
       end

       function h = hypothesisfunction(obj,X,theta)
           if strcmp(obj.transfer_fcn,'linear')
                h = X*theta;
           elseif strcmp(obj.transfer_fcn,'linear+1')
                h = X*theta + 1;
           end
       end
    end
end
%% Trainer class

classdef Trainer < handle
    properties (Access = public)
        thetaOpt_vec
        thetaOpt_mat
        num_features
    end

    properties (Access = private)
        nn
        data
        lambda
        theta0
    end

    methods (Access = public)
        function obj = Trainer(nn,data,l)
            obj.nn = nn;
            obj.data = data;
            obj.lambda = l;
            obj.num_features = size(obj.data.Xfull,2);
            obj.computeinitialtheta();            
        end

        function train(obj,showgraph)
           obj.thetaOpt_vec = obj.solveTheta(obj.data.Xfull,obj.data.Yfull,showgraph);
           obj.thetaOpt_mat = obj.thetavec_to_thetamat(obj.thetaOpt_vec);
        end
    end
    

    methods (Access = private)
        function computeinitialtheta(obj)
           count = obj.num_features*obj.nn.sizes(1);
           for i = 2:length(obj.nn.sizes)
                count = count + obj.nn.sizes(i-1)*obj.nn.sizes(i);
           end
           obj.theta0 = zeros(1,count);
        end

        function theta = solveTheta(obj,x,y,dis)
           if dis == true
                options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','PlotFcn',{@optimplotfval},'StepTolerance',10^(-6),'MaxFunEvals',3000,'CheckGradients',true);
           else
                options = optimoptions(@fminunc,'Algorithm','quasi-newton','StepTolerance',10^(-6),'MaxFunEvals',1000);
           end
           F = @(theta) obj.computeCost(x,y,theta);
           theta = fminunc(F,obj.theta0,options);
        end

        function [J,grad] = computeCost(obj,x,y,theta)
           [a, J, ~] = obj.forwardprop(x,theta,y);
            grad = obj.backwardprop(a,theta,y);
        end

        function [a, J, h] = forwardprop(obj,x,theta,y)
            thetamat = obj.thetavec_to_thetamat(theta);
            h = hypothesisFunction(x,thetamat.(thetamat.name{1}));
            g = sigmoid(h);
            a.name = genvarname(repmat({'l'},1,length(obj.nn.sizes)+1),'l');
            a.(a.name{1}) = x;
            a.(a.name{2}) = g; 
            for i = 2:length(obj.nn.sizes)
                h = hypothesisFunction(g,thetamat.(thetamat.name{i}));
                g = sigmoid(h);
                a.(a.name{i+1}) = g; 
            end
            for i = 1:obj.nn.sizes(end)
                J_vec(i) = (1/length(y))*sum((1-y(:,i)).*(-log(1-g(:,i)))+y(:,i).*(-log(g(:,i))));
            end
            J = sum(J_vec) + 0.5/length(y)*obj.lambda*(theta*theta');
       end
       
       function grad = backwardprop(obj,a,theta,y)
            grad = zeros(1,length(theta));
            thetamat = obj.thetavec_to_thetamat(theta);
            delta.name = genvarname(repmat({'l'},1,length(obj.nn.sizes)+1),'l');
            delta.(delta.name{end}) = a.(a.name{end}) - y;
            aux = (1/length(y))*(delta.(delta.name{end})'*a.(a.name{end-1}))' + obj.lambda*thetamat.(thetamat.name{end});
            newend = size(aux,1)*size(aux,2);
            grad((end-newend+1):end) = reshape(aux,[1,size(aux,1)*size(aux,2)]);
            newend = length(theta) - newend;
            for i = length(obj.nn.sizes):-1:2
                product = (thetamat.(thetamat.name{i})*delta.(delta.name{i+1})')';
                delta.(delta.name{i}) = product*(a.(a.name{i})'*(1 - a.(a.name{i})));
                product2 = (delta.(delta.name{i})'*a.(a.name{i-1}))';
                aux = (1/length(y))*product2 + obj.lambda*thetamat.(thetamat.name{i-1});
                grad(newend-size(aux,1)*size(aux,2)+1:newend) = reshape(aux,[1,size(aux,1)*size(aux,2)]);
                newend = newend - size(aux,1)*size(aux,2);
            end
       end

       function thetamat = thetavec_to_thetamat(obj,thetavec)
           thetamat.name = genvarname(repmat({'l'},1,length(obj.nn.sizes)),'l');
           last = obj.nn.sizes(1)*obj.num_features;
           aux = reshape(thetavec(1:last),[obj.num_features,obj.nn.sizes(1)]);
           thetamat.(thetamat.name{1}) = aux;
           for i = 2:length(obj.nn.sizes)
               aux = reshape(thetavec(last+1:(last+obj.nn.sizes(i)*obj.nn.sizes(i-1))),[obj.nn.sizes(i-1),obj.nn.sizes(i)]);
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
           h = X*theta;
       end
    end
end
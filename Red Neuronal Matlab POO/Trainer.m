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
                options = optimoptions(@fminunc,'SpecifyObjectiveGradient',true,'Display','iter','Algorithm','quasi-newton','PlotFcn',{@optimplotfval},'StepTolerance',10^(-6),'MaxFunEvals',3000,'CheckGradients',true);
           else
                options = optimoptions(@fminunc,'Algorithm','quasi-newton','StepTolerance',10^(-6),'MaxFunEvals',1000);
           end
           F = @(theta) obj.computeCost(x,y,sizes,theta);     
           theta = fminunc(F,obj.theta0,options);          
        end

        function [J,grad_auto] = computeCost(obj,x,y,sizes,theta)
           [a, J] = obj.forwardprop(x,y,theta,sizes);
            grad = obj.backwardprop(a,y,theta,sizes);

            theta_dl = dlarray(theta);
            J_dl = @(theta) obj.f_autodiff(x,y,theta,sizes);
            [loss,gradval] = dlfeval(J_dl,theta_dl);
            grad_auto = extractdata(gradval);
            g_err = norm(grad_auto-grad)/norm(grad_auto);
            J_err = abs(loss-J);
            error = false;
            if (g_err > 1e-2)
                disp('The error in loss is: ')
                disp(J_err)
                disp('The gradient error is: ')
                disp(g_err)
                error = true;
            end         
        end
        
        function [J,dF_dtheta] = f_autodiff(obj,x,y,theta,sizes)
            [~,J] = obj.forwardprop(x,y,theta,sizes);
            dF_dtheta = dlgradient(J,theta);
        end

        function [a, J] = forwardprop(obj,x,y,theta,sizes)
            thetamat = obj.thetavec_to_thetamat(theta,sizes);            
            a = obj.create_activation_fcn(sizes,x);
            for i = 1:length(sizes)               
                a = obj.save_activation_fcn_i(a,thetamat,i);
            end
            J = obj.get_fcn_value(sizes,theta,y,a.(a.name{end}));           
       end
       
       function grad = backwardprop(obj,a,y,theta,sizes)  
            len = length(theta);
            thetamat = obj.thetavec_to_thetamat(theta,sizes);
            delta = create_delta(obj,sizes,a,y);
            [grad,last] = obj.create_gradient(y,a,delta,thetamat,len);
            for i = length(sizes):-1:2
                [grad,delta,last] = obj.save_gradient_i(grad,thetamat,delta,a,y,i,last);
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
               last = last+sizes(i)*sizes(i-1);
           end
       end  
       
       function a = create_activation_fcn(obj,sizes,x)
            a.name = genvarname(repmat({'l'},1,length(sizes)+1),'l');
            a.(a.name{1}) = x;
       end

       function a = save_activation_fcn_i(obj,a,th,i)
                previous_g = a.(a.name{i});
                h = obj.hypothesisfunction(previous_g,th.(th.name{i}));
                g = sigmoid(h);
                a.(a.name{i+1}) = g;
       end

       function J = get_fcn_value(obj,sizes,theta,y,g)
            for i = 1:sizes(end)
                err1 = (1-y(:,i)).*(-log(1-g(:,i)));
                err0 = y(:,i).*(-log(g(:,i)));
                J_vec(i) = (1/length(y))*sum(err1+err0);
            end
            J = sum(J_vec) + 0.5/length(y)*obj.lambda*(theta*theta');
       end

       function delta = create_delta(obj,sizes,a,y)
            delta.name = genvarname(repmat({'l'},1,length(sizes)+1),'l');
            delta.(delta.name{end}) = a.(a.name{end}) - y;
       end

       function [grad,last] = create_gradient(obj,y,a,delta,th,len)
           grad = zeros(1,len);   
           delta_e = delta.(delta.name{end});
           a_e1 = a.(a.name{end-1});
           th_e = th.(th.name{end});
           aux = (1/length(y))*(delta_e'*a_e1)' + obj.lambda*th_e;
           last = size(aux,1)*size(aux,2);
           grad((end-last+1):end) = reshape(aux,[1,size(aux,1)*size(aux,2)]);
           last = len - last;
       end

       function [grad,delta,last] = save_gradient_i(obj,grad,th,delta,a,y,i,last)
            th_i = th.(th.name{i});
            th_1i = th.(th.name{i-1});
            a_i = a.(a.name{i});
            a_1i = a.(a.name{i-1});        
            delta_i1 = delta.(delta.name{i+1});

            delta_i = (th_i*delta_i1')'.*(a_i.*(1 - a_i));
            delta.(delta.name{i}) = delta_i;
            aux = (1/length(y))*(delta_i'*a_1i)' + obj.lambda*th_1i;
            grad(last-size(aux,1)*size(aux,2)+1:last) = reshape(aux,[1,size(aux,1)*size(aux,2)]);
            last = last - size(aux,1)*size(aux,2);
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
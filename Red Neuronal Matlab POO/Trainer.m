%% Trainer class

classdef Trainer < handle
    properties (Access = public)
        lambda
        transfer_fcn
    end

    properties (Access = private)        
        theta0
        hist_Cost
        numData
        Xdata
        Ydata
        neuronsPerLayer
        numFeatures
        nLayers
    end

    methods (Access = public)
        function obj = Trainer(l,fcn)
            obj.lambda = l;
            obj.transfer_fcn = fcn;                   
        end

        function [tOpt_v,tOpt_m] = train(obj,nn,data,showgraph)
           obj.neuronsPerLayer = nn.sizes;
           obj.Xdata       = data.Xfull;
           obj.Ydata       = data.Yfull;
           obj.numData     = size(obj.Ydata ,1);
           obj.numFeatures = size(obj.Xdata,2);
           obj.nLayers     = length(obj.neuronsPerLayer);
           obj.computeinitialtheta();     
           tOpt_v = obj.solveTheta(showgraph);
           tOpt_m = obj.thetavec_to_thetamat(tOpt_v);
        end
    end
    

    methods (Access = private)

        function theta = solveTheta(obj,isDisplayed)
                opt = optimoptions(@fminunc);
                opt.SpecifyObjectiveGradient = true;
                opt.Algorithm = 'quasi-newton';
                opt.StepTolerance = 10^-6;
                opt.MaxFunctionEvaluations = 3000;              
           if isDisplayed == true
                opt.Display = 'iter';
                opt.CheckGradients = true;
                opt.OutputFcn = @obj.myoutput;
                %opt = optimoptions(@fminunc,'SpecifyObjectiveGradient',true,'Display','iter','Algorithm','quasi-newton','StepTolerance',10^(-6),'MaxFunEvals',3000,'CheckGradients',true,'OutputFcn',@obj.myoutput);             
                %opt = optimoptions(@fminunc,'Algorithm','quasi-newton','StepTolerance',10^(-6),'MaxFunEvals',1000);
           end
           F = @(theta) obj.computeCost(theta);     
           theta = fminunc(F,obj.theta0,opt); 
           figure
           plot(2:size(obj.hist_Cost,2),obj.hist_Cost(1,2:end),'d',2:size(obj.hist_Cost,2),obj.hist_Cost(2,2:end),'d')
           legend('Loss','Function value')
        end

        function stop = myoutput(obj,theta,optimvalues,state)
            stop = false;  
            iter = optimvalues.iteration;
            if isequal(state,'init')
%               Thistory = [];
              obj.hist_Cost = [0;0];
            end        
            if isequal(state,'iter')
%               Thistory = [Thistory, x];
              f = optimvalues.fval;
              [c,~] = obj.computeLossFunction(theta);
               r = obj.computeRegularizationTerm(theta);
 
             % c = f - obj.computeRegularizationTerm(theta);
%               if mod(iter,10) == 0
%                 PlotBoundary(data,NN)
%               end
              obj.hist_Cost = [obj.hist_Cost(1,:), c;
                               obj.hist_Cost(2,:), f];
            end
        end

        function [j_auto,grad_auto] = computeCost(obj,theta)
%            [a, J,loss] = obj.forwardprop(x,y,theta,sizes,numfeat);
%             grad = obj.backwardprop(a,y,theta,sizes,numfeat);

            theta_dl = dlarray(theta);
            J_dl = @(theta) obj.f_autodiff(theta);
            [j_auto,grad_auto] = dlfeval(J_dl,theta_dl);
            grad_auto = extractdata(grad_auto);   
            j_auto = extractdata(j_auto); 
        end
        
        function [J,dF_dtheta] = f_autodiff(obj,theta)
            [~,J] = obj.forwardprop(theta);
            dF_dtheta = dlgradient(J,theta);
        end

        function [a, J] = forwardprop(obj,theta)
            [c,a] = obj.computeLossFunction(theta);
            r = obj.computeRegularizationTerm(theta);
            l = obj.lambda;
            J = c + l*r;
       end
       
       function grad = backwardprop(obj,a,y,theta,sizes,numfeat)  
            len = length(theta);
            thetamat = obj.thetavec_to_thetamat(theta,sizes,numfeat);
            delta = create_delta(obj,sizes,a,y);
            [grad,last] = obj.create_gradient(y,a,delta,thetamat,len);
            for i = length(sizes):-1:2
                [grad,delta,last] = obj.save_gradient_i(grad,thetamat,delta,a,y,i,last);
            end
       end
       
       function computeinitialtheta(obj)
           nL  = obj.neuronsPerLayer;
           nF = obj.numFeatures;
           nLayer = length(nL);
           nTheta = nF*nL(1);
           for i = 2:nLayer
                nTheta = nTheta + nL(i-1)*nL(i);
           end
           obj.theta0 = zeros(1,nTheta);
       end
       
       function thetamat = thetavec_to_thetamat(obj,thetavec)
           nL = obj.neuronsPerLayer;
           nF = obj.numFeatures;
           thetamat.name = genvarname(repmat({'l'},1,length(nL)),'l');
           last = nL(1)*nF;
           aux = reshape(thetavec(1:last),[nF,nL(1)]);
           thetamat.(thetamat.name{1}) = aux;
           for i = 2:length(nL)
               aux = reshape(thetavec(last+1:(last+nL(i)*nL(i-1))),[nL(i-1),nL(i)]);
               thetamat.(thetamat.name{i}) = aux;
               last = last+nL(i)*nL(i-1);
           end
       end  
       
       function a = create_activation_fcn(obj)
            x  = obj.Xdata;
            nL = obj.neuronsPerLayer;
            a.name = genvarname(repmat({'l'},1,length(nL)+1),'l');
            a.(a.name{1}) = x;
       end

       function a = save_activation_fcn_i(obj,a,th,i)
           gOld = a.(a.name{i});
           h = obj.hypothesisfunction(gOld,th.(th.name{i}));
           g = sigmoid(h);
           a.(a.name{i+1}) = g;
       end

       function [J,a] = computeLossFunction(obj,theta)
            thetamat = obj.thetavec_to_thetamat(theta);            
            a = obj.create_activation_fcn();
            for i = 1:obj.nLayers
                a = obj.save_activation_fcn_i(a,thetamat,i);
            end
            g = a.(a.name{end});

           y = obj.Ydata;
           nL = obj.neuronsPerLayer;           
           nD = obj.numData;
           J = 0;
           for i = 1:nL(end)
               err1 = (1-y(:,i)).*(-log(1-g(:,i)));
               err0 = y(:,i).*(-log(g(:,i)));
               j = (1/nD)*sum(err1+err0);
               J = J + j;
           end
       end


       function r = computeRegularizationTerm(obj,theta)
           nD = obj.numData;
           r = 0.5/nD*(theta*theta');
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
           aux = (1/length(y))*((delta_e'*a_e1)' + obj.lambda*th_e);
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
            aux = (1/length(y))*((delta_i'*a_1i)' + obj.lambda*th_1i);
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
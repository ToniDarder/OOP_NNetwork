%% Trainer class

classdef Trainer < handle
    properties (Access = public)
        lambda
        transfer_fcn
    end

    properties (Access = private)        
        theta0
        Xdata
        Ydata        
        numData
        numFeatures
        nLayers
        neuronsPerLayer
    end

    methods (Access = public)
        function obj = Trainer(l,fcn)
            obj.lambda = l;
            obj.transfer_fcn = fcn;                   
        end

        function train(obj,nn,data,showgraph)
           obj.neuronsPerLayer = nn.neuronsPerLayer;
           obj.Xdata       = data.Xfull;
           obj.Ydata       = data.Yfull;
           obj.numData     = data.numData;
           obj.numFeatures = data.numFeatures;
           obj.nLayers     = length(obj.neuronsPerLayer);
           obj.computeinitialtheta();     
           thOpt_v = obj.solveTheta(showgraph);
           thOpt_m = obj.thetavec_to_thetamat(thOpt_v);
           nn.thetaOpt = thOpt_v;
           nn.thetaOpt_m = thOpt_m;
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
           end

           F = @(theta) obj.computeCost(theta);     
           theta = fminunc(F,obj.theta0,opt); 
        end

        function stop = myoutput(obj,theta,optimvalues,state)
            persistent optfig hist_Cost Thistory
            stop = false;  
            switch state
                case 'init'
                    Thistory = [];
                    hist_Cost = [0;0;0];
                    optfig = figure;
                    
                case 'iter'
                    Thistory = [Thistory, theta];
                    iter = optimvalues.iteration;
                    f = optimvalues.fval;
                    [c,~] = obj.computeLossFunction(theta);
                    r = obj.computeRegularizationTerm(theta)*obj.lambda;
                    if mod(iter,10) == 0
%                       PlotBoundary(obj.Xdata,NN)
                        v = 0:10:iter;
                        hist_Cost = [hist_Cost(1,:), f;
                                     hist_Cost(2,:), c;
                                     hist_Cost(3,:), r];
                        figure(optfig)
                        plot(v,hist_Cost(1,2:end),'dr',v,hist_Cost(2,2:end),'db',v,hist_Cost(3,2:end),'dg')
                        legend('Fval','Loss','Regularization')
                        xlabel('Iterations')
                        ylabel('Function Values')
                        drawnow
                    end
                case 'done'
            end
        end

        function [j_adiff,g_adiff] = computeCost(obj,theta)
           % [a, J,loss] = obj.forwardprop(x,y,theta,sizes,numfeat);
           % grad = obj.backwardprop(a,y,theta,sizes,numfeat);

            th = dlarray(theta);
            J = @(theta) obj.f_adiff(theta);
            [j_adiff,g_adiff] = dlfeval(J,th);
            g_adiff = extractdata(g_adiff);   
            j_adiff = extractdata(j_adiff); 
        end
        
        function [J,grad] = f_adiff(obj,theta)
            [~,J] = obj.forwardprop(theta);
            grad = dlgradient(J,theta);
        end

        function [a, J] = forwardprop(obj,theta)
            [c,a] = obj.computeLossFunction(theta);
            r = obj.computeRegularizationTerm(theta);
            l = obj.lambda;
            J = c + l*r;
       end
       
       function grad = backwardprop(obj,a,y,theta)  
            nTh = length(theta);
            nL = obj.nLayers;
            th_m = obj.thetavec_to_thetamat(theta);
            delta = create_delta(obj,a,y);
            [grad,last] = obj.create_gradient(y,a,delta,theta);
            for i = nL:-1:2
                [grad,delta,last] = obj.save_gradient_i(grad,th_m,delta,a,y,i,last);
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
       
       function th_m = thetavec_to_thetamat(obj,thetavec)
           nL = obj.neuronsPerLayer;
           nF = obj.numFeatures;
           th_m.name = genvarname(repmat({'l'},1,length(nL)),'l');
           last = nL(1)*nF;
           aux = reshape(thetavec(1:last),[nF,nL(1)]);
           th_m.(th_m.name{1}) = aux;
           for i = 2:length(nL)
               aux = reshape(thetavec(last+1:(last+nL(i)*nL(i-1))),[nL(i-1),nL(i)]);
               th_m.(th_m.name{i}) = aux;
               last = last+nL(i)*nL(i-1);
           end
       end  
       
       function [J,a] = computeLossFunction(obj,theta)
           thetamat = obj.thetavec_to_thetamat(theta);
           a = obj.computeActivationFCN(thetamat);
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

       function a = computeActivationFCN(obj,th)
           x  = obj.Xdata;
           nL = obj.nLayers;
           a.name = genvarname(repmat({'l'},1,nL+1),'l');
           a.(a.name{1}) = x;
           for i = 1:nL
               g_prev = a.(a.name{i});
               h = obj.hypothesisfunction(g_prev,th.(th.name{i}));
               g = sigmoid(h);
               a.(a.name{i+1}) = g;
           end
       end

       function delta = create_delta(obj,a,y)
            nL = obj.nLayers;
            delta.name = genvarname(repmat({'l'},1,nL+1),'l');
            delta.(delta.name{end}) = a.(a.name{end}) - y;
       end

       function [grad,last] = create_gradient(obj,y,a,delta,thetavec)
           th = obj.thetavec_to_thetamat();
           nTh = length(thetavec);
           grad = zeros(1,nTh);   
           delta_e = delta.(delta.name{end});
           a_e1 = a.(a.name{end-1});
           th_e = th.(th.name{end});
           grad_e = (1/length(y))*((delta_e'*a_e1)' + obj.lambda*th_e);
           last = size(grad_e,1)*size(grad_e,2);
           grad((end-last+1):end) = reshape(grad_e,[1,size(grad_e,1)*size(grad_e,2)]);
           last = nTh - last;
       end

       function [grad,delta,last] = save_gradient_i(obj,grad,th,delta,a,y,i,last)
            th_i = th.(th.name{i});
            th_1i = th.(th.name{i-1});
            a_i = a.(a.name{i});
            a_1i = a.(a.name{i-1});        
            delta_i1 = delta.(delta.name{i+1});

            delta_i = (th_i*delta_i1')'.*(a_i.*(1 - a_i));
            delta.(delta.name{i}) = delta_i;
            grad_i = (1/length(y))*((delta_i'*a_1i)' + obj.lambda*th_1i);
            grad(last-size(grad_i,1)*size(grad_i,2)+1:last) = reshape(grad_i,[1,size(grad_i,1)*size(grad_i,2)]);
            last = last - size(grad_i,1)*size(grad_i,2);
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
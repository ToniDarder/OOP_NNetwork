classdef Trainer < handle

    properties (Access = private)        
        theta0    
    end

    properties (Access = private)
       lambda 
       network
       data
       isDisplayed
    end

    methods (Access = public)

        function obj = Trainer(s)
           obj.init(s);
        end
        
        function train(obj)
           nn = obj.network;
           d  = obj.data;
           obj.computeinitialtheta();     
           nn.thetaOpt = obj.solveTheta(d,nn);
           nn.thetaOpt_m = obj.thetavec_to_thetamat(nn.thetaOpt);
        end

    end    

    methods (Access = private)

        function init(obj,s)
            obj.lambda      = s.lambda;
            obj.network     = s.network;
            obj.data        = s.data;
            obj.isDisplayed = s.isDisplayed;
        end

        function theta = solveTheta(obj,data,nn)
                opt = optimoptions(@fminunc);
                opt.SpecifyObjectiveGradient = true;
                opt.Algorithm = 'quasi-newton';
                opt.StepTolerance = 10^-6;
                opt.MaxFunctionEvaluations = 3000;              
           if obj.isDisplayed == true
                opt.Display = 'iter';
                opt.CheckGradients = true;
                opt.OutputFcn = @(theta,optimvalues,state)obj.myoutput(theta,optimvalues,state,data,nn);
           end

           F = @(theta) obj.computeCost(theta);     
           theta = fminunc(F,obj.theta0,opt); 
        end

        function stop = myoutput(obj,theta,optimvalues,state,data,nn)
            persistent optfig bound_ev hist_Cost Thistory
            stop = false;  
            switch state
                case 'init'
                    Thistory = [];
                    hist_Cost = [0;0;0];
                    optfig = figure;
                    bound_ev = figure;
                    
                case 'iter'
                    Thistory = [Thistory, theta];
                    iter = optimvalues.iteration;
                    f = optimvalues.fval;
                    [c,~] = obj.computeLossFunction(theta);
                    r = obj.computeRegularizationTerm(theta)*obj.lambda; 
                    nIter = 1;
                    if mod(iter,nIter) == 0                       
                        v = 0:nIter:iter;
                        hist_Cost = [hist_Cost(1,:), f;
                                     hist_Cost(2,:), c;
                                     hist_Cost(3,:), r];
                        figure(optfig)
                        plot(v,hist_Cost(1,2:end),'+-r',v,hist_Cost(2,2:end),'+-b',v,hist_Cost(3,2:end),'+-k')
                        legend('Fval','Loss','Regularization')
                        xlabel('Iterations')
                        ylabel('Function Values')
                        drawnow
                    end
                    if mod(iter,25) == 0
                        th_m = obj.thetavec_to_thetamat(theta);
                        nFigure = bound_ev;
                        figure(nFigure);
                        obj.network.plotBoundary(data,th_m,nFigure)
                        %PlotBoundary  
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
            nT = length(theta);
            nL = obj.nLayers;
            th_m = obj.thetavec_to_thetamat(theta);
            delta = create_delta(obj,a,y);
            [grad,last] = obj.create_gradient(y,a,delta,theta);
            for i = nL:-1:2
                [grad,delta,last] = obj.save_gradient_i(grad,th_m,delta,a,y,i,last);
            end
       end
       
       function computeinitialtheta(obj)
           nPL = obj.network.neuronsPerLayer;
           nF = obj.data.nFeatures;
           nLayer = obj.network.nLayers;
           nTheta = nF*nPL(1);
           for i = 2:nLayer
                nTheta = nTheta + nPL(i-1)*nPL(i);
           end
           obj.theta0 = zeros(1,nTheta);
       end
       
       function th_m = thetavec_to_thetamat(obj,thetavec)
           nL = obj.network.nLayers;
           nPL = obj.network.neuronsPerLayer;
           nF = obj.data.nFeatures;
           th_m.name = genvarname(repmat({'l'},1,nL),'l');
           last = nPL(1)*nF;
           aux = reshape(thetavec(1:last),[nF,nPL(1)]);
           th_m.(th_m.name{1}) = aux;
           for i = 2:nL
               aux = reshape(thetavec(last+1:(last+nPL(i)*nPL(i-1))),[nPL(i-1),nPL(i)]);
               th_m.(th_m.name{i}) = aux;
               last = last+nPL(i)*nPL(i-1);
           end
       end  
       
       function [J,a] = computeLossFunction(obj,theta)
           thetamat = obj.thetavec_to_thetamat(theta);
           a = obj.computeActivationFCN(thetamat);
           g = a.(a.name{end});
           y =  obj.data.Ytrain;
           nPL = obj.data.numLabels;
           nD = obj.data.nData;
           J = 0;
           for i = 1:nPL
               err1 = (1-y(:,i)).*(-log(1-g(:,i)));
               err0 = y(:,i).*(-log(g(:,i)));
               j = (1/nD)*sum(err1+err0);
               J = J + j;
           end
       end

       function r = computeRegularizationTerm(obj,theta)
           nD = obj.data.nData;
           r = 0.5/nD*(theta*theta');
       end

       function a = computeActivationFCN(obj,th)
           x  = obj.data.Xtrain;
           nL = obj.network.nLayers;
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
            nL = obj.network.nLayers;
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

    end

    methods (Access = private, Static)
       
        function h = hypothesisfunction(X,theta)
          h = X*theta;
        end

    end
end
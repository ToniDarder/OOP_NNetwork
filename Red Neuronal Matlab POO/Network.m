%% Class for Network

classdef Network
   properties (Access = public)
      Sizes
      Num_layers
      Theta0
      ThetaOpt
      Lambda
      Num_Features
   end
   properties (Dependent = true)
      Theta0mat
   end
   methods (Access = public)

       function obj = Network(Net_Structure,Lambda)
           obj.Num_Features = 0;
           obj.Sizes = Net_Structure;
           obj.Num_layers = length(Net_Structure);
           %Thetamat = obj.Thetavec_to_Thetamat(obj.Theta0);
           obj.Lambda = Lambda;
           
       end

       function [ThetaOpt,Num_Features] = Train(obj,data)
           [Xtrain,Ytrain,Xtest,Ytest] = data.SplitData();
           Xfull = data.ComputeFullX(Xtrain,1);
           Yfull = Ytrain;
           Num_Features = size(Xfull,2);
           obj.Num_Features = size(Xfull,2);
           obj.Theta0 = obj.ComputeInitialTheta();
           ThetaOpt1 = obj.SolveTheta(Xfull,Yfull);
           [J,grad] = obj.ComputeCost(Xfull,Yfull,ThetaOpt1);
           ThetaOpt = ThetaOpt1;
       end

       function h = Xpropagation(obj,X,theta)
            thetamat = obj.Thetavec_to_Thetamat(theta);
            h = hypothesisFunction(X,thetamat.(thetamat.name{1}));
            g = sigmoid(h);
            for i = 2:obj.Num_layers
                h = hypothesisFunction(g,thetamat.(thetamat.name{i}));
                g = sigmoid(h);
            end
       end
   end
   methods (Access = private)

       function Theta0 = ComputeInitialTheta(obj)
           count = obj.Num_Features*obj.Sizes(1);
           for i = 2:obj.Num_layers
                count = count + obj.Sizes(i-1)*obj.Sizes(i);
           end
           Theta0 = zeros(1,count);
       end

       function theta = SolveTheta(obj,X,Y)
           %options = optimoptions(@fminunc,'Algorithm','quasi-newton','StepTolerance',10^(-6),'MaxFunEvals',5000);
           options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','PlotFcn',{@optimplotfval},'StepTolerance',10^(-6),'MaxFunEvals',1000,'CheckGradients',true);
           %options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton','StepTolerance',10^(-6),'MaxFunEvals',5000,'CheckGradients',true);
           F = @(theta) obj.ComputeCost(X,Y,theta);
           [theta,fval,exitflag,output] = fminunc(F,obj.Theta0,options);
       end

       function [J,grad] = ComputeCost(obj,X,y,theta)
           [a, J, ~] = obj.forwardprop(X,theta,y);
            grad = obj.backwardprop(a,theta,y);
       end

       function [a, J, h] = forwardprop(obj,X,theta,y)
            thetamat = obj.Thetavec_to_Thetamat(theta);
            h = hypothesisFunction(X,thetamat.(thetamat.name{1}));
            g = sigmoid(h);
            a.name = genvarname(repmat({'l'},1,obj.Num_layers+1),'l');
            a.(a.name{1}) = X;
            a.(a.name{2}) = g; 
            for i = 2:obj.Num_layers
                %g = reshape(g,[size(X,1) obj.Sizes(i-1)]);
                h = hypothesisFunction(g,thetamat.(thetamat.name{i}));
                g = sigmoid(h);
                a.(a.name{i+1}) = g; 
            end
            %g = reshape(g,[size(X,1) obj.Sizes(end)]);
            for i = 1:obj.Sizes(end)
                J_vec(i) = (1/length(y))*sum((1-y(:,i)).*(-log(1-g(:,i)))+y(:,i).*(-log(g(:,i))));
            end
            J = sum(J_vec) + 0.5/length(y)*obj.Lambda*(theta*theta');
       end
       
       function grad = backwardprop(obj,a,theta,y)
            grad = zeros(1,length(theta));
            thetamat = obj.Thetavec_to_Thetamat(theta);
            delta.name = genvarname(repmat({'l'},1,obj.Num_layers+1),'l');
            delta.(delta.name{end}) = a.(a.name{end}) - y;
            aux = (1/length(y))*(delta.(delta.name{end})'*a.(a.name{end-1}))' + obj.Lambda*thetamat.(thetamat.name{end});
            newend = size(aux,1)*size(aux,2);
            grad((end-newend+1):end) = reshape(aux,[1,size(aux,1)*size(aux,2)]);
            newend = length(theta) - newend;
            for i = obj.Num_layers:-1:2
                product = (thetamat.(thetamat.name{i})*delta.(delta.name{i+1})')';
                delta.(delta.name{i}) = product*(a.(a.name{i})'*(1 - a.(a.name{i})));
                product2 = (delta.(delta.name{i})'*a.(a.name{i-1}))';
                aux = (1/length(y))*product2 + obj.Lambda*thetamat.(thetamat.name{i-1});
                grad(newend-size(aux,1)*size(aux,2)+1:newend) = reshape(aux,[1,size(aux,1)*size(aux,2)]);
                newend = newend - size(aux,1)*size(aux,2);
            end
       end
       
       function thetamat = Thetavec_to_Thetamat(obj,thetavec)
           thetamat.name = genvarname(repmat({'l'},1,obj.Num_layers),'l');
           last = obj.Sizes(1)*obj.Num_Features;
           aux = reshape(thetavec(1:last),[obj.Num_Features,obj.Sizes(1)]);
           thetamat.(thetamat.name{1}) = aux;
            for i = 2:obj.Num_layers
                aux = reshape(thetavec(last+1:(last+obj.Sizes(i)*obj.Sizes(i-1))),[obj.Sizes(i-1),obj.Sizes(i)]);
                thetamat.(thetamat.name{i}) = aux;
            end
       end           
   end   
   
end
%% Class for Network

classdef Network
   properties
      Sizes
      Num_layers
      Theta0
      ThetaOpt
      Lambda
   end
   methods
       function obj = Network(Net_Structure,Lambda)
           obj.Sizes = Net_Structure;
           obj.Num_layers = len(Net_Structure);
           obj.ThetaOpt = zeros(3); %%%%%%%%%
           obj.Lambda = Lambda;
       end
       function [a, J, h] = forwardprop(obj,X,theta,y)
            h = hypothesisFunction(X,theta(:,1));
            g = sigmoid(h);
            a(:,1) = X(:);
            a(:,2) = g; 
            for i = 2:obj.Num_layers
                g = reshape(g,[size(X,1) obj.Sizes(i-1)]);
                h = hypothesisFunction(g,theta(:,i));
                g = sigmoid(h);
                a(:,i+1) = g; 
            end
            g = reshape(g,[size(X,1) obj.Sizes(end)]);
            for i = 1:obj.Sizes(end)
                J_vec(i) = (1/length(y))*sum((1-y(:,i)).*(-log(1-g(:,i)))+y(:,i).*(-log(g(:,i))));
            end
            theta_vec = reshape(theta, size(X,2)*obj.Num_layers*obj.Sizes(end), []); %%%% No es tan facil
            J = sum(J_vec) + 0.5/length(y)*obj.Lambda*(theta_vec'*theta_vec);
       end
       
       % Revisar todo el backward, crear un func thetavec_to_thetamat
       
       function grad = backwardprop(obj,a,theta,X,y,class)
            delta(:,obj.Num_layers + 1) = a(:,end) - reshape(y,class*size(X,1),[]);
            for i = obj.Num_layers:-1:1
                theta_aux = reshape(theta(:,i), [size(X,2) class]);
                delta_aux = reshape(delta(:,i+1), [size(X,1) class]);
                if i ~= 1
                    product = reshape(theta_aux*delta_aux',class*size(X,1),[]);
                    delta(:,i) = product.*(a(:,i)'*(1 - a(:,i)));
                end
                a_aux = reshape(a(:,i), [size(X,1) class]);
                product = reshape(delta_aux'*a_aux,class*size(X,2),[]);
                grad(:,i)=(1/length(y))*product + obj.Lambda*theta(:,i);
            end
            grad = reshape(grad, class*obj.Num_layers*size(X,2), []);   
       end
       function [J,grad] = ComputeCost(obj,X,y,theta)
           [a, J, ~] = obj.forward(X,theta,y);
            grad = obj.backward(a,theta,X,y,class);
       end
    end
end
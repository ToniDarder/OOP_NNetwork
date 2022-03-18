%%%%%%%%%%% Old Backward prop %%%%%%%%%%%

%        function grad = backwardprop(self,theta) 
%             th_m = self.thetavec_to_thetamat(theta);
%             a = self.a_fcn;
%             y = self.data.Ytrain;
%             nL = self.nLayers;      
%             delta = create_delta(self,a,y);
%             [grad,last] = self.create_gradient(delta,theta);
%             for i = nL:-1:2
%                 [grad,delta,last] = self.save_gradient_i(grad,th_m,delta,i,last);
%             end
%        end           
% 
%        function delta = create_delta(self)
%             a = self.a_fcn;
%             y = self.data.Ytrain;
%             nLy = self.nLayers;
%             delta = cell(nLy,1);
%             delta{end} = a{end} - y;
%        end
% 
%        function [grad,last] = create_gradient(self,delta,theta)
%            th_m = self.thetavec_to_thetamat(theta);
%            a = self.a_fcn;
%            y = self.data.Ytrain;
%            nTh = length(theta);
%            grad = zeros(1,nTh);   
%            delta_e = delta{end};
%            a_e1 = a{end-1};
%            th_e = th_m{end};
%            grad_e = (1/length(y))*((delta_e'*a_e1)' + self.lambda*th_e);
%            last = size(grad_e,1)*size(grad_e,2);
%            grad((end-last+1):end) = reshape(grad_e,[1,size(grad_e,1)*size(grad_e,2)]);
%            last = nTh - last;
%        end
% 
%        function [grad,delta,last] = save_gradient_i(self,grad,th,delta,i,last)
%             a = self.a_fcn; 
%             y = self.data.Ytrain;
%             th_i = th{i};
%             th_1i = th{i-1};
%             a_i = a{i};
%             a_1i = a{i-1};        
%             delta_i1 = delta{i+1};
%             delta_i = (th_i*delta_i1')'.*(a_i.*(1 - a_i));
%             delta{i} = delta_i;
%             grad_i = (1/length(y))*((delta_i'*a_1i)' + self.lambda*th_1i);
%             grad(last-size(grad_i,1)*size(grad_i,2)+1:last) = reshape(grad_i,[1,size(grad_i,1)*size(grad_i,2)]);
%             last = last - size(grad_i,1)*size(grad_i,2);
%        end
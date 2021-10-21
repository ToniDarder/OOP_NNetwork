%% class for neurons

classdef Neuron
    properties
       position         % Number of the neuron inside the layer
       TransferValues   % Thetas
    end
    methods
        function obj = Neuron(position)
            obj.position = position;           
        end
        function r = Neuron_propagation(obj,inputs)
            totalinput = sum(inputs);
            r = totalinput*obj.TransferValues; % Horizontal pls
        end
    end    
end

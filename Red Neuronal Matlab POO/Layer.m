%% class for layer which is compound by neurons

classdef Layer
    properties
        layer         % Which layer it is
        N_Neurons
        Neurons       % List of neurons inside this layer
        currentvalues % Values passed, one to each neuron
    end
    methods
        function obj = Layer(N_Neurons,i)
            obj.layer = i;
            obj.N_Neurons = N_Neurons;
            obj.Neurons = Neuron.epmty(N_Neurons,0);
            for i = 1:N_Neurons
                obj.Neurons(i) = Neuron(i);
            end
        end
        function r = Layer_propagation(obj)
            r1 = zeros(len(obj.currentvalues),obj.N_Neurons);
            for i = 1:obj.N_Neurons
                r1(i,:) = obj.Neurons(i).Neuron_propagation(obj.currentvalues(i));
            end
            r = sum(r1,1);
        end
    end
end

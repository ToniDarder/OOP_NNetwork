function    analyzeHyperparameter(h,network,data,trainer)
    n = 10;
    object = h.class;
    hp.type = h.type;
    %v = linspace(h.min,h.max,n);
    v = [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1];
    Tcost = zeros([n,1]);
    VtestError = zeros([n,1]);
    trainer.isDisplayed = false;
    for i = 1:n
        hp.value = v(i);
        switch object
            case 'data'
                data.updateHyperparameter(hp);
            case 'network'
                network.updateHyperparameter(hp)
            case 'trainer'
                trainer.updateHyperparameter(hp)
        end
        trainer.train();
        Tcost(i) = network.cost;
        [~,y_pred] = max(network.getOutput(data.Xtest),[],2);
        [~,y_target] = max(data.Ytest,[],2);
        VtestError(i) = mean(y_pred ~= y_target);
        disp(string(i))
    end
    figure(1)
    semilogx(v,Tcost)
    xlabel(hp.type)
    ylabel('Cost function after training')
    figure(2)
    semilogx(v,VtestError)
    xlabel(hp.type)
    ylabel('Test accuracy after training')
end

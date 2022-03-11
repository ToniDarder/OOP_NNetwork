function    analyzeHyperparameter(h,network,data,trainer)
    n = 10;
    object = h.class;
    hp.type = h.type;
    v = linspace(h.min,h.max,n);
    cost = zeros([n,1]);
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
        cost(i) = network.cost;
        disp(string(i))
    end
    plot(v,cost)
    txt = ['Cost in function of hyperparameter ',hp.type];
    xlabel(hp.type)
    ylabel('Cost function after training')
    title(txt)
end

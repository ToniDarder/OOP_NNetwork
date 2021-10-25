%% Plot the data and the boundary conds
function PlotBoundary(data,NN)
    X = data.Xdata;
    n = size(X,2);
    m = size(X,1);
    n_points = 200;
    
    x1 = linspace(1.5*min(X(:,1)),1.5*max(X(:,1)),n_points);
    x1 = x1';
    x2 = min(X(:,2)) + zeros(n_points,1);
    x2_aux = zeros(n_points,1);
    
    X_test = zeros(n_points,NN.Num_Features,n_points);
    h = zeros(n_points*NN.Sizes(end),n_points);
    for i = 1:n_points 
        x2 = x2 + (1.5*max(X(:,2)) - 1.5*min(X(:,2)))/n_points;
        x2_aux(i) = x2(1);
        xdata_test = [x1 , x2];
        X_test(:,:,i) = data.ComputeFullX(xdata_test,1);
        h(:,i) = reshape(NN.Xpropagation(X_test(:,:,i),NN.ThetaOpt),[n_points*NN.Sizes(end),1]);
    end
    
    figure(1)
    data.PlotData();
    for j = 1:NN.Sizes(end)
        
        h_1 = h((j-1)*size(X_test,1)+1:j*size(X_test,1),:);
        
        %contour(x1,x2_aux,h',[0 0])
        hold on

        if j == 1
            contour(x1,x2_aux,h_1',[0 0],'r')
        elseif j == 2
            contour(x1,x2_aux,h_1',[0 0],'g')
        elseif j==3
            contour(x1,x2_aux,h_1',[0 0],'b')
        end
    end
end


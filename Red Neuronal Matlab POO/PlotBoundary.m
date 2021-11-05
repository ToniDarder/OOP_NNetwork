%% Plot the data and the boundary conds
function PlotBoundary(data,NN)
    X = data.Xfull;
    n = size(X,2);
    m = size(X,1);
    n_points = 200;
    
    x1 = linspace(1.5*min(X(:,2)),1.5*max(X(:,2)),n_points);
    x1 = x1';
    x2 = min(X(:,3)) + zeros(n_points,1);
    x2_aux = zeros(n_points,1);
    
    X_test = zeros(n_points,size(X,2),n_points);
    h = zeros(n_points*NN.sizes(end),n_points);
    for i = 1:n_points 
        x2 = x2 + (1.5*max(X(:,3)) - 1.5*min(X(:,3)))/n_points;
        x2_aux(i) = x2(1);
        b = ones(n_points,1);
        xdata_test = [b, x1 , x2];
        X_test(:,:,i) = xdata_test;
        %X_test(:,:,i) = data.computefullX(xdata_test,1);
        h(:,i) = reshape(NN.compute_last_H(X_test(:,:,i)),[n_points*NN.sizes(end),1]);
    end
    
    figure(1)
    data.plotdata();
    for j = 1:NN.sizes(end)
        
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


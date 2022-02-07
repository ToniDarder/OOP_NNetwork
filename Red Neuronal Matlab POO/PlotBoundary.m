%% Plot the data and the boundary conds
function PlotBoundary(data,NN,th_m)
    X = data.Xfull;
    nF = size(X,2);
    nPL = NN.neuronsPerLayer;
    n_pts = 200;
    graphzoom = 10;
    extra_f1 = mean(X(:,2))*graphzoom;
    extra_f2 = mean(X(:,3))*graphzoom;
    x1 = linspace(min(X(:,2))-extra_f1,max(X(:,2))+extra_f1,n_pts)';
    x2 = min(X(:,3)) - extra_f2 + zeros(n_pts,1);
    x2_aux = zeros(n_pts,1);
    
    X_test = zeros(n_pts,nF,n_pts);
    h = zeros(n_pts*nPL(end),n_pts);
    deltaX2 = (max(X(:,3)) - min(X(:,3)) + 2*extra_f2)/n_pts;
    for i = 1:n_pts 
        x2 = x2 + deltaX2;
        x2_aux(i) = x2(1);
        b = ones(n_pts,1);
        xdata_test = [b, x1 , x2];
        X_test(:,:,i) = xdata_test;
        %X_test(:,:,i) = data.computefullX(xdata_test,1);
        h(:,i) = reshape(NN.compute_last_H(X_test(:,:,i),th_m),[n_pts*nPL(end),1]);
    end
    
    data.plotdata();
    hold on
    colors = ['r','g','b','c','m','y','k'];
    for j = 1:nPL(end)      
        h_j = h((j-1)*size(X_test,1)+1:j*size(X_test,1),:);        
        %contour(x1,x2_aux,h',[0 0])

        contour(x1,x2_aux,h_j',[0 0],'color',colors(j))
    end
end


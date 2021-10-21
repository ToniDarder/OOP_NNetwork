# ----------------------- Main Libraries ------------------------- #
import numpy as np
from numpy.core.fromnumeric import reshape
from scipy.optimize import minimize
from scipy.optimize import check_grad
import random
import matplotlib
import matplotlib.pyplot as plt
import csv

# ---------------------- Classes: Data, Network ---------------------- #
class Data():
    def __init__(self,Xdata,Ydata,PercentageSplit):
        self.Xdata = Xdata
        self.Ydata = Ydata
        self.Num_Experiences = len(Xdata)
        self.Num_Features = len(Xdata[0])
        self.PercentageSplit = PercentageSplit

    def SplitData(self):
        # Shuffles the data and splits data in train and test 
        shuffled_list = []
        Xtrain = []
        Ytrain = []
        Xtest = []
        Ytest = []
        for i in range(self.Num_Experiences):
            shuffled_list.append(i)
        random.shuffle(shuffled_list)
        shuffled_Xdata = self.Xdata.copy()
        shuffled_Ydata = self.Ydata.copy()
        for i in range(self.Num_Experiences):
            shuffled_Xdata[i] = self.Xdata[shuffled_list[i]]
            shuffled_Ydata[i] = self.Ydata[shuffled_list[i]]
        SizeTest = round(self.PercentageSplit/100*self.Num_Experiences)
        SizeTrain = self.Num_Experiences - SizeTest     
        for i in range (SizeTrain):
            Xtrain.append(shuffled_Xdata[i])
            Ytrain.append(shuffled_Ydata[i])
        for i in range(SizeTest):
            Xtest.append(shuffled_Xdata[SizeTrain + i])
            Ytest.append(shuffled_Ydata[SizeTrain + i])
        Xtrain = np.array(Xtrain)
        Ytrain = np.array(Ytrain)
        Xtest = np.array(Xtest)
        Ytest = np.array(Ytest)
        return Xtrain,Ytrain,Xtest,Ytest

    def ComputeFullX(self,X,d):
        Xfull = []
        Xfull.append(np.ones(len(X)))
        if d ==1:
            for i in range(self.Num_Features):
                Xfull.append(X[:,i].T)
        else:
            for i in range(1,d+1):
                for j in range(self.Num_Features):
                    Xfull.append(X[:,j]**i)
            for i in range(self.Num_Features-1):
                for j in range(i+1,self.Num_Features):
                    for n in range(2,d+1):
                        for k in range(1,n):
                            Xfull.append((X[:,i]**(n-k)*X[:,j]**k).T)
        #print(np.shape(Xfull[0]))
        #print(np.shape(Xfull[1]))
        Xfull = np.array(Xfull).T
        #print(Xfull[0,:])
        return Xfull

    def PrintDataSplit(self):
        # Prints all arrays of data
        print('Xdata: ',self.Xdata)
        print('Ydata: ',self.Ydata)
        print('Xtest: ',self.Xtest)
        print('Xtrain: ',self.Xtrain)
        print('Ytest: ',self.Ytest)
        print('Ytrain: ',self.Ytrain)

    def PlotData(self,feature1,feature2):
        # Now it is not working as I changed the essence of Ydata array
        markers = ["o","+","^","d","*","x","2","s"]
        #for i,c in enumerate(np.unique(self.Ydata)):
            #plt.scatter(self.Xdata[:,feature1][self.Ydata==c],self.Xdata[:,feature2][self.Ydata==c],c=self.Ydata[self.Ydata==c], marker = markers[i])
        plt.scatter(self.Xdata[:,feature1],self.Xdata[:,feature2],c=self.Ydata)
        plt.show()

class Network():
    # For now Network and Data are independent classes but maybe class Network could inherit Data 
    # attributes, or maybe it's just ok to have them separately
    def __init__(self,Net_Structure):
        # Net_Structure is a vector with the number of neurons per layer
        self.num_layers = len(Net_Structure)
        self.sizes = Net_Structure
        #self.Theta0 = self.computeInitialTheta()
        self.Theta0 = []
        self.Num_Features = 0
        self.ThetaOpt = []

    def computeInitialTheta(self):
        # Gaussian distribution to initialize theta, other option is to initialize at 0
        Theta = []
        Thetacount = self.Num_Features*self.sizes[0]
        #Theta = [np.zeros((y, x)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        for x, y in zip(self.sizes[:-1], self.sizes[1:]):
            Thetacount += x*y
        Theta = np.zeros((Thetacount,))
        #print(Theta)
        return Theta

    def hypothesisFunction(self,X,theta):
        # The function which relates every layer of the Network
        return X@theta

    def forwardprop(self,Thetavec,X,Y,Lambda):
        # With input parameters Xtrain, Ytrain and the lambda coef  
        # Gives back all 'a' which contain X and sigmoid(hypothesisfunc(layer(i-1),layer(i)))
        # Also gives the final cost J and the last h calculated as well (usefull in backprop)
        J_vec = []
        Theta = self.Thetavec_to_ThetaMat(Thetavec)
        #print(Thetavec)
        #print(Theta)
        a = [[]]
        a[0] = X.copy()
        h = self.hypothesisFunction(X,Theta[0][:,:])
        g = sigmoid(h)
        a.append(g)
        for i in range(1,self.num_layers):
            h = self.hypothesisFunction(g,Theta[i][:,:])
            g = sigmoid(h)
            a.append(g)
        for i in range(self.sizes[-1]):
            J_vec.append((1/(len(Y)))*np.sum((1-Y[:,i])*(-np.log(1-g[:,i])) + Y[:,i]*(-np.log(g[:,i]))))
        J = np.sum(J_vec) + 0.5*Lambda*(Thetavec.T@Thetavec)/len(Y)
        return a, J, h

    def xprop(self,Thetavec,X):
        # Similar to forwardprop, but it only propagates the result, does not get the cost and the a matrix
        Theta = self.Thetavec_to_ThetaMat(Thetavec)
        h = self.hypothesisFunction(X,Theta[0][:,:])
        g = sigmoid(h)
        for i in range(1,self.num_layers):
            h = self.hypothesisFunction(g,Theta[i][:,:])
            g = sigmoid(h)
        return h

    def backwardprop(self,Thetavec,X,Y,Lambda,a):
        # With given inputs Xtrain,Yrain, Lambda and a returns the gradiend of the theta
        # Note that a is an input therefore forwardprop must have been called before
        delta = []
        grad = []
        Theta = self.Thetavec_to_ThetaMat(Thetavec)
        delta.append(a[-1] - Y)
        grad.append((1/len(Y))*(a[-2].T@delta[-1]) + Lambda*Theta[-1])
        for i in range(1,self.num_layers):
            if i != self.num_layers:
                '''
                print(Theta[-i].shape)
                print(Theta[-i])
                print(delta[-1].shape)
                print(delta[-1])
                print(a[-i-1].shape)
                print(a[-i-1])
                print((a[-1-i].T@(1-a[-1-i])).shape)
                '''
                #delta.append(Theta[-i]@(delta[-1]@(a[-i-1].T@(1-a[-i-1]))).T) # one option i think
                delta.append(Theta[-i]@delta[-1].T*np.sum(a[-i-1]*(1-a[-i-1]))) # other option
            #print(delta)
            grad.append((1/len(Y))*(delta[-1]@a[-2-i]).T + Lambda*Theta[-1-i])    
        grad = list(reversed(grad))
        gradvec = grad[0].reshape(-1)
        for i in range(1,self.num_layers):
            gradvec = np.hstack((gradvec,grad[i].reshape(-1)))
        #print(grad)
        #print(gradvec)
        
        return gradvec

    def ComputeCost(self,Theta,X,Y,Lambda):
        _,J,_ = self.forwardprop(Theta,X,Y,Lambda)
        return J

    def ComputeGradient(self,Theta,X,Y,Lambda):
        # Uses forward and backward propagation to eval Cost and Gradient
        a,_,_ = self.forwardprop(Theta,X,Y,Lambda)
        grad = self.backwardprop(Theta,X,Y,Lambda,a)
        return grad

    def SolveTheta(self,X,Y,Lambda):
        err = check_grad(self.ComputeCost,self.ComputeGradient,(self.Theta0+1),X,Y,Lambda)    
        print("Error in gradient at start = ",err)
        opt = {'disp': True,'gtol':1e-8}
        # Methods dont need gradient: Nelder-Mead, powell | Methods which need gradient: BFGS,Newton-CG
        ThetaOpt = minimize(self.ComputeCost,self.Theta0,args=(X,Y,Lambda),method='BFGS',jac=False,options=opt)
        #ThetaOpt = minimize(self.ComputeCost,self.Theta0,args=(X,Y,Lambda),method='Nelder-Mead',options=opt)
        return ThetaOpt.x

    def Train(self,data):
        # Trains the Net with the dataset given as an input and choosing some parameters
        #  {iterations, d polynomial degree, Lambda...}
        iterations = 1
        d = 1
        Lambda = 0
        
        for i in range(iterations):
            Xtrain,Ytrain,Xtest,Ytest = data.SplitData()
            Xfull = data.ComputeFullX(Xtrain,d)
            Yfull = Ytrain
            self.Num_Features = np.shape(Xfull)[1]
            # Add a compute fullx
            self.Theta0 = self.computeInitialTheta()
            #ThetaOpt = self.Theta0
            ThetaOpt1 = self.SolveTheta(Xfull,Yfull,Lambda)
            print(ThetaOpt1)
            J = self.ComputeCost(ThetaOpt1,Xfull,Yfull,Lambda) # Aqui hay que hacer las ops con Xtest...
            grad = self.ComputeGradient(ThetaOpt1,Xfull,Yfull,Lambda)
        self.ThetaOpt = ThetaOpt1

    def Thetavec_to_ThetaMat(self,Thetavec):
        # Reshape theta into 3D array for easy operations
        Theta = []
        last = self.sizes[0]*self.Num_Features
        Theta_aux = Thetavec[0:self.sizes[0]*self.Num_Features].reshape(self.Num_Features,self.sizes[0])
        Theta.append(Theta_aux)
        for x, y in zip(self.sizes[:-1], self.sizes[1:]):
            Theta_aux = Thetavec[last:(last+x*y)].reshape(x,y)
            Theta.append(Theta_aux)
            last += x*y
        return Theta

# ---------------------- Functions ---------------------- #
def sigmoid(z):
    # The sigmoid function
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    # Derivative of the sigmoid function
    return sigmoid(z)*(1-sigmoid(z))

def load_data(fileCSV): 
    # Function to load a csv and save it into three variables
    # Maybe this should be a method insidde class Data, dunno
    Xdata = []
    Ydata = []
    Headings = []
    with open(fileCSV) as file:
        reader = csv.reader(file,delimiter=';')
        data = list(reader)
    data = np.array(data)
    #Xdata = np.array(data[1:np.shape(data)[0],0:-1:1],dtype='float64')  
    Xdata = np.array(data[1:np.shape(data)[0],2:4],dtype='float64')  
    ydata = np.array(data[1:np.shape(data)[0],-1],dtype='int32')
    Ydata = np.zeros((len(ydata),np.amax(ydata)-np.amin(ydata)+1))
    for i in range(len(ydata)):
        if ydata[i] == 1:
            Ydata[i,0] = 1
        elif ydata[i] == 2:
            Ydata[i,1] = 1
        else:
            Ydata[i,2] = 1
    Headings = np.array(data[0,:])
    return Xdata,Ydata,Headings

def PlotBoundary(data,NN):
    
    X = data.Xdata
    Y = data.Ydata
    n_points = 200
    x1 = np.linspace(1.5*min(X[:,0]),1.5*max(X[:,0]),n_points).T
    x1 = np.reshape(x1,(n_points,1))
    x2 = np.zeros((n_points,1)) + 1.5*min(X[:,1])
    y = np.zeros((n_points,3))
    # xdata = np.concatenate(x1,x2)
    # ydata = np.ones((len(xdata),1))
    # data = Data(xdata,ydata,0)
    x2_aux = np.zeros((n_points,1))
    xtest = np.zeros((n_points,NN.Num_Features,n_points))
    h = np.zeros((n_points*NN.sizes[-1],n_points))

    #Theta = np.array([-11.685, 2.568, 4.425, -10.666, 1.332, 2.515, -10.68, 1.34, 2.498, -6.226, -1.278, -1.271, 7.783, -7.88, -7.9, -7.923, 8.009, 8.018])
    Theta = NN.ThetaOpt
    for i in range(n_points):
        x2 = x2 + 1.5*(max(X[:,1])-min(X[:,1]))/n_points
        x2_aux[i] = x2[0]
        xdata_test = np.concatenate((x1,x2),axis=1)
        xtest[:,:,i] = data.ComputeFullX(xdata_test,1) 

        #_,_,haux = NN.forwardprop(Theta,xtest[:,:,i],y,0)
        haux = NN.xprop(Theta,xtest[:,:,i])
        
        h[:,i] = np.reshape(haux.T,(n_points*NN.sizes[-1],))
        #print(haux)
        #print(h[:,0])

    data.PlotData(0,1)
    x1_draw = np.reshape(x1,(n_points,))
    x2_draw = np.reshape(x2_aux,(n_points,))
    x1_draw,x2_draw = np.meshgrid(x1_draw,x2_draw)
    for j in range(NN.sizes[-1]):
        h_draw = h[(j*n_points):((j+1)*n_points),:].T    
        if j == 0:
            plt.contour(x1_draw,x2_draw,h_draw,levels = [0],colors = 'red')
        elif j == 1:
            plt.contour(x1_draw,x2_draw,h_draw,levels = [0], colors = 'green')
        elif j == 2:
            plt.contour(x1_draw,x2_draw,h_draw,levels = [0],colors = 'blue')
    aaaa = 1

# -------------------- MAIN CODE --------------------- #
pruebaX,pruebaY,Headings = load_data('iris.csv')
data1 = Data(pruebaX,pruebaY,0)
'''
data1.PrintDataSplit()
plt.figure(1)
data1.PlotData(data1.Xdata[:,0],data1.Xdata[:,1],data1.Ydata)
plt.figure(2)
data1.PlotData(data1.Xtrain[:,0],data1.Xtrain[:,1],data1.Ytrain)
plt.figure(3)
data1.PlotData(data1.Xtest[:,0],data1.Xtest[:,1],data1.Ytest)
'''
# Ara mateix sembla que funciona amb una layer pero no amb mes?
# Afegir dimensions als features
Net_Structure = [3]
Mynet = Network(Net_Structure)
Mynet.Train(data1)
PlotBoundary(data1,Mynet)
print(Mynet.ThetaOpt)

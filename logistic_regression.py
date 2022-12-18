import numpy as np

class GradientDescentLogisticRegression:
    
    def __init__(self, iteration=500, learning_rate=0.001):
        self.__iteration = iteration
        self.__learning_rate = learning_rate
        self.__epsilon = 1e-5
        self.__wights = None
        self.__bias = 0
    
    def fit(self,x,y):
        num_of_observation, num_of_feature = x.shape # Training Size Capture
        self.__wights = np.random.normal(loc=0,scale=1/num_of_feature, size=num_of_feature )
        #self.__wights = np.zeros(shape=num_of_feature ) # Weights initialization
        self.__bias = 0                                  # bias initialization  
        i ,history = 0, []
        while i < self.__iteration:                #    Training Loop Strt Here ##########
            yhat = self.__forward_propagation(x)   #    Forward Propagation     ##########
            grd_w, grd_b = self.__compute_grediant(x,y,yhat) # Gredient Calculation   ####
            self.__update_weights(grd_w, grd_b)    #    Weights and bais Update    #######   
            loss = self.__compute_loss(yhat,y)     #    Loss Computation    ##############
            TP,TN,FP,FN,precision,recall,F1 = self.__compute_accuracy(yhat,y) #  Accuracy Computation  #######
            history.append({                       #   Training History ##################
                            'iteration':i,
                            'loss': loss,
                            'TP':TP,'TN':TN,'FP':FP,'FN':FN,
                            'precision':precision,'recall':recall,'F1':F1
                           })
            if loss <= self.__epsilon:              # End Traing if Loss bellow Threshold
                break
            i += 1                                  # Traing Loop End Here
        print(f'Iteration {self.__iteration} Training Loss: {loss} TP:{TP},TN:{TN},FP:{FP},FN:{FN}')    
        return history
    
    def __forward_propagation(self,x):
        z = x.dot(self.__wights) + self.__bias
        return 1/( 1 + np.exp(-z) )
    
    def __compute_grediant(self, x,y,yhat):
        error_vector = yhat - y    
        num_of_observation, _ = x.shape
        grad_w = np.matmul(np.transpose(x),error_vector)/num_of_observation
        grad_b = np.mean(error_vector)
        return grad_w, grad_b
    
    def __update_weights(self, grd_w, grd_b):
        self.__wights = self.__wights - self.__learning_rate*grd_w
        self.__bias = self.__bias - self.__learning_rate*grd_b
    
    def __compute_loss(self,yhat,y):
        loss_1 = -1*y*np.log(yhat+self.__epsilon)
        loss_0 = -1*(1-y)*np.log(1-yhat+self.__epsilon)
        loss = np.mean(loss_1+loss_0)
        return loss
    
    def __compute_accuracy(self,yhat,y):
        y_pred = np.where(yhat >=.5,1,0)
        TP = np.sum(np.logical_and(y==1,y_pred==1))
        TN = np.sum(np.logical_and(y==0,y_pred==0))
        FP = np.sum(np.logical_and(y==0,y_pred==1))
        FN = np.sum(np.logical_and(y==1,y_pred==0))
        precision = TP/(TP+FP)  # Ratio of TP and positive predictions
        recall = TP/(TP+FN)     # Ratio of TP and Actual positive
        F1 = 2*(precision*recall)/(precision+recall)
        return TP,TN,FP,FN,precision,recall,F1
        
    def predict(self,x):
        y_pred =  self.__forward_propagation(x)
        return np.where(y_pred >=.5,1,0)
import numpy as np
from math import floor,ceil

def generate_binary_classification(size):
    class_1_size = ceil(size/2)
    class_2_size = size - class_1_size
    class_1_mean, class_1_std  = 5, 1.25
    class_2_mean, class_2_std  = 12.5, 2.25
    
    x_1 = np.random.normal(loc=class_1_mean,scale=class_1_std, size=class_1_size) 
    y_1 = np.random.normal(loc=class_1_mean,scale=class_1_std, size=class_1_size)  
    class_1_target = np.full(class_1_size, 0)
    class_1_data = np.column_stack((x_1,y_1, class_1_target))
    
    x_2 = np.random.normal(loc=class_2_mean,scale=class_2_std, size=class_2_size) 
    y_2 = np.random.normal(loc=class_2_mean,scale=class_2_std, size=class_2_size)  
    class_2_target = np.full(class_2_size, 1)
    class_2_data = np.column_stack((x_2,y_2, class_2_target))
    
    data = np.vstack((class_1_data,class_2_data))
    np.random.shuffle(data)
    
    return data[:,:-1],data[:,-1].astype(int)

def test_train_split(dataset,test_train_ratio = 0.2):
    np.random.shuffle(dataset)
    num_of_observtions, num_predictors = dataset.shape
    print(f'num_of_observtions {num_of_observtions}, num_predictors {num_predictors}')
    test_train_ratio = 0.2
    test_size = ceil(num_of_observtions * test_train_ratio)
    test_data = dataset[:test_size]
    train_data = dataset[test_size:]
    return train_data, test_data

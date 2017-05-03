import h5py
from sklearn.model_selection import train_test_split

def load():
    f = h5py.File("dataset.h5")
    x = f['x'].value
    y = f['y'].value
    f.close()
    x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=100)
    return x_train, x_test, y_train, y_test
    

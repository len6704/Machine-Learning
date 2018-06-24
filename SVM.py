import matplotlib.pyplot as plt
#from matplotlib import style
import numpy as np
#style.use('ggplot')
import sys
class SVM:
    def __init__(self,visualization=True):
        self.visualization = visualization
		#self.colors = {1:'r',-1:'b'}
        self.max_feature_value = 0
        self.min_feature_value = 0
        self.data = 0
        #if self.visualization:
        #self.fig = plt.figure()
        #self.ax = self.fig.add_subplot(1,1,1)
    def fit(self,data):
        opt_dict = {}
        self.data = data
        transforms = [[1,1],[-1,1],[-1,-1],[1,-1]]
        all_data = []           
        for yi in self.data:
			for featureset in self.data[yi]:
				for feature in featureset:
					all_data.append(feature)
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None
        #step_size=[self.max_feature_value*0.1,self.max_feature_value*0.01,self.max_feature_value*0.001]
        step_size=[self.max_feature_value*0.1,self.max_feature_value*0.05]
        #step_size=[0.8,0.08,0.008]
        b_range_multiple = 5
        b_multiple = 10
        latest_optimum = (self.max_feature_value)*10       
        for step in step_size:
            w = np.array([latest_optimum,latest_optimum])
            optimized = False
            while not optimized:
                for b in np.arange(-1*self.max_feature_value*b_range_multiple,
                                    1*self.max_feature_value*b_range_multiple,
                                    step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        if b%10 == 0:
                            print "w_t =", w_t, "w =",w, "transformation =",transformation,"b =",b
                        found_option = True
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if b%10 == 0:
                                    print "yi =",yi ,"dot =",np.dot(w_t,xi)+b,"xi =",xi
                                if not yi*(np.dot(w_t,xi)+b) >= 1:                                   
                                    found_option = False
                                    print "FALSE", "step =",step,"b =",b, "yi =",yi ,"dot =",np.dot(w_t,xi)+b,"xi =",xi
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]
                if w[0] < 0:
                    optimized = True
                    print("optimized")
                else:
                    w = w - step	
            norm = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norm[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]+step*2
            #print "w =",self.w,"b =",self.b







    def predict(self,features):
		classification = np.sign(np.dot(np.array(feature),self.w)+self.b)	
		return classification




data_dict = { 1:np.array([[5,1],[6,-1],[7,3]]),
              -1:np.array([[1,7],[2,8],[3,8]])
            }
svm = SVM(visualization=False)
svm.fit(data_dict)

print (sys.path)


            

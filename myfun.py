from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
def plot_learning_curve(estimator,title,x,y,cv=10,n_jobs=1,train_sizes=np.linspace(0.1,1.0,20)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes,train_scores,test_scores =learning_curve(estimator,x,y,cv=cv,n_jobs=-1, train_sizes=train_sizes)
    train_scores_std=np.std(train_scores, axis=1)
    train_scores_mean=np.mean(train_scores, axis=1)
    test_scores_mean=np.mean(test_scores, axis=1)
    test_scores_std=np.std(train_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes,train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,alpha=0.1, color="r")
    plt.fill_between(train_sizes,test_scores_mean-test_scores_std,test_scores_mean+test_scores_std,alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean,'o-',color="r",label="Training Score")
    plt.plot(train_sizes, test_scores_mean,'o-',color="g",label="Cross Validation Score")
    plt.legend(loc="best")
    plt.show()
    return plt 
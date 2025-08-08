import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plotsCatter(true_label, predict_label, label_name, R21, RMSE1, RPD1, MAE1, range1, range2, x, y, a, b, c):
    dpi = 1000
    plt.rcParams['font.family'] = 'Times New Roman' 
    plt.rcParams['font.size'] = 28 

    plt.rcParams['xtick.direction'] = 'in'  
    plt.rcParams['ytick.direction'] = 'in'  

    plt.gca().set_aspect(1)
    ax = plt.gca()
    ax.set_xticks(range1)
    ax.set_yticks(range1)

    plt.text(x,y, '    R$^2$=%.3f' % R21, color='black') # 11
    plt.text(x,y-a, '    RMSE=%.2f' % RMSE1, color='black')
    plt.text(x,y-a-b, '    RPD=%.2f' % RPD1, color='black')
    plt.text(x,y-a-b-c, '    MAE=%.2f' % MAE1, color='black')

    # plt.title(title, y=-0.16, fontsize=12, fontdict={'fontname': 'SimSun'}) 
    plt.plot(range2, range2, label='1:1 line', color='green', linestyle='dashdot')
    plt.xlabel('True value', fontdict={'fontname': 'Times New Roman'})
    plt.ylabel('Predicted value', fontdict={'fontname': 'Times New Roman'})

    plt.scatter(true_label, predict_label, color='tomato', label=label_name, marker='o', edgecolor='red', s=100)
    plt.legend(frameon=False, loc="upper left", prop={'family': 'Times New Roman'})  
    plt.tight_layout()
    plt.show()


def plotBar(y_test, y_pred, y_range, title):
    # labels = ['A', 'B', 'C', 'D']    
    plt.rcParams['font.family'] = 'Times New Roman' 
    plt.rcParams['font.size'] = 25 
    
    x = np.array(range(len(y_test)))    
    width = 0.35    
    
    fig, ax = plt.subplots()  
    fig.set_size_inches(8, 3)  
    rects1 = ax.bar(x - width/2, y_test, width, label='Predicted value')    
    rects2 = ax.bar(x + width/2, y_pred, width, label='True value')   
      
    ax.set_ylabel('Value') # , fontdict={'fontname': 'Times New Roman'})  
    ax.set_xlabel('Sample')  
    ax.set_title(title, y=-0.3)  

    # label = []
    # for i in x:
    #     if i%10 == 0:
    #         label.append(i)
    #     else:
    #         label.append(' ')
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70])
    # ax.set_yticks([20, 30, 40, 50, 60, 70])
    # ax.set_xticks(x)  
    # ax.set_xticklabels(x)  
    ax.set_xticklabels([0, 10, 20, 30, 40, 50, 60, 70], fontdict={'fontname': 'Times New Roman'})  
    ax.set_yticks(y_range)
    ax.set_yticklabels(y_range, fontdict={'fontname': 'Times New Roman'})  
    ax.legend(loc='upper center',           
              bbox_to_anchor=(0.25, 1.15),   
              ncol=2,                       
              frameon=False)                 
    plt.tight_layout()   
    plt.show()


def plotHyper(path):
    value = pd.read_excel(path).values[:, 1:]
    hyper = value[:, 5:5+204]

    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    for i in hyper:
        plt.plot(i)
    plt.xlabel('波段', fontdict={'fontname': 'SimSun'})  
    plt.ylabel('反射光谱强度', fontdict={'fontname': 'SimSun'})
    plt.show()



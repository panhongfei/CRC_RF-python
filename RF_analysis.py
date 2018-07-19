#!/usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
from sklearn.externals.six import StringIO
from sklearn import tree
import pydotplus
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

from sklearn.tree import export_graphviz
#from mlxtend.evaluate import plot_decision_regions
from mlxtend.plotting import plot_decision_regions

# path = r'D:\learn\Bioinfo\zzz_paper\RF_program\RandomForest\RF_data60_156.csv'
# path = r'D:\learn\Bioinfo\zzz_paper\RF_program\RandomForest\CRCdata_all.csv'
# path = r'D:\learn\Bioinfo\zzz_paper\RF_program\RandomForest\RFdata_2_N_SLC.csv'
# path = r'D:\learn\Bioinfo\zzz_paper\RF_program\RandomForest\RFdata_MINI_top300_2_N_C.csv'
# path = r'D:\learn\Bioinfo\zzz_paper\RF_program\RandomForest\RFdata_MINI_top300_2_N_SC.csv'
# path = r'D:\learn\Bioinfo\zzz_paper\RF_program\RandomForest\RFdata_chayi_top32_2_N_SLC.csv'
# path = r'D:\learn\Bioinfo\zzz_paper\RF_program\RandomForest\RFdata_2_NS_LC.csv'
path = r'D:\learn\Bioinfo\zzz_paper\RF_program\RandomForest\new_old\new_old_2_N_C.csv'

# path = r'D:\learn\Bioinfo\zzz_paper\RF_program\RandomForest\RFdata_nozhong\RFdata_gra_732_2_N_SLC.csv'

data = pd.read_csv(path,header=0,sep=",")
x_prime = data.iloc[:,1:data.columns.size-1]
# x_prime = data.iloc[:, 1:300]

# pca = PCA(n_components="mle")

# x_prime = data.iloc[:, 1:60]
# x_prime = pca.fit_transform(x_prime1)
y = pd.Categorical(data.iloc[:,data.columns.size-1]).codes
print data.columns.size
x_prime = Imputer().fit_transform(x_prime)
# y = pd.Categorical(data.iloc[1:, [data.columns.size-1]]).codes

x_prime_train, x_prime_test, y_train, y_test = train_test_split(x_prime, y, train_size=0.95, random_state=1)
# x_train = x_prime_train
# x_test = x_prime_test
x_train = x_prime_train
x_test = x_prime_train
y_test = y_train

# 决策树学习 entropy
model = RandomForestClassifier(n_estimators=200, criterion='gini',max_depth=6, oob_score=True)
model.fit(x_train, y_train)
# plot_decision_regions(x_prime,y,model,legend = 299)
# 训练集上的预测结果
y_train_pred = model.predict(x_train)
print y_train_pred
print y_train

print len(y_train_pred)
acc_train = accuracy_score(y_train, y_train_pred)
y_test_pred = model.predict(x_test)
acc_test = accuracy_score(y_test, y_test_pred)
print y_test
print y_test_pred
model.get_params()

print 'OOB Score:', model.oob_score_
print '\t训练集准确率: %.4f%%' % (100 * acc_train)
print '\t测试集准确率: %.4f%%\n' % (100 * acc_test)
# print '所有的树:%s' % model.estimators_

# print model.classes_
# print model.n_classes_
# predictions_validation = model.predict_proba(x_test)
# print predictions_validation
# fpr, tpr, _ = roc_curve(y_test, predictions_validation[:, 1])
# roc_auc = auc(fpr, tpr)
# plt.title('ROC Validation')
# plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
# plt .legend(loc='lower right')
# plt.plot([0, 1], [0, 1], 'r--')
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()

# print '各feature的重要性：%s' % model.feature_importances_


 # plot the tree in the RF
for i in range(10):
    tree_in_forest = model.estimators_[i]
    dot_data = StringIO()
    tree.export_graphviz(tree_in_forest, out_file=dot_data)
    tmp = []
    tmp.append([0,0,0]);
    k = 1
    for line in  dot_data.buflist:
        tline = line
        rindex = tline.find("->")
        if rindex >= 1:
            print(tline)
            father = int(tline[0:(rindex-1)])
            son = int(tline[(rindex+3):])
            level = -1
            for j in range(k):
                if father == tmp[j][0]:
                    level = tmp[j][2]+1
                    k = k + 1
                    tmp.append([son, father, level])
                    break
    print(tmp)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf( "aaaaRF_fenxi_"+str(i)+".pdf")
    fileName="aairsRF_"+str(i)+".dot"
    with open(fileName,"w")as f:
        f=tree.export_graphviz(model.estimators_[i].tree_, out_file=f)
        print(model.estimators_)

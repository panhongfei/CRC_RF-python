# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score

# path = r'D:\learn\Bioinfo\zzz_paper\RF_program\RandomForest\RFdata_2_N_C.csv'
# path = r'D:\learn\Bioinfo\zzz_paper\RF_program\RandomForest\CRCdata_all.csv'
# path = r'D:\learn\Bioinfo\zzz_paper\RF_program\RandomForest\RFdata_nozhong\RFdata_gra_732_2_N_SLC.csv'
# path = r'D:\learn\Bioinfo\zzz_paper\RF_program\RandomForest\RFdata_MINI_top300_2_N_C.csv'
path = r'D:\learn\Bioinfo\zzz_paper\RF_program\RandomForest\new_old\new_old_2_N_C.csv'

data = pd.read_csv(path,header=0,sep=",")

X = data.iloc[:,1:data.columns.size-1]

# X = data.iloc[:,1:10]
X = np.array(X)
y = pd.Categorical(data.iloc[:,data.columns.size-1]).codes
y= np.array(y)


cv = StratifiedKFold(n_splits=6)
RFmodel = RandomForestClassifier(n_estimators=500, criterion='gini',max_depth=5, oob_score=True)
mean_tpr_RF = 0.0
mean_fpr_RF = np.linspace(0, 1, 100)
acc_test_RF = 0

# all_tpr = []

for train_index, test_index in cv.split(X, y):
    #probas_ = model.fit(X[train], y[train]).predict_proba(X[test])
    #print(X)

    RFmodel.fit(X[train_index], y[train_index])
    # 测试集
    y_test_pred = RFmodel.predict(X[test_index])
    acc_test_tem = accuracy_score(y_test_pred, y[test_index])
    RFprobas_ =RFmodel.predict_proba(X[test_index])
    fpr_RF, tpr_RF, thresholds_RF = roc_curve(y[test_index], RFprobas_[:, 1])
    # # # 训练集
    # RFprobas_ =RFmodel.predict_proba(X[train_index])
    # fpr_RF, tpr_RF, thresholds_RF = roc_curve(y[train_index], RFprobas_[:, 1])
    mean_tpr_RF += interp(mean_fpr_RF, fpr_RF, tpr_RF)  # 对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
    mean_tpr_RF[0] = 0.0  # 初始处为0
    roc_auc_RF = auc(fpr_RF, tpr_RF)
    acc_test_RF =acc_test_RF + acc_test_tem
    #plt.plot(fpr, tpr, lw=1, label='ROC fold  (area = %0.2f)' % (roc_auc))


plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))


mean_tpr_RF /= 6  # 在mean_fpr100个点，每个点处插值插值多次取平均
mean_tpr_RF[-1] = 1.0  # 坐标最后一个点为（1,1）
mean_auc_RF = auc(mean_fpr_RF, mean_tpr_RF)  # 计算平均AUC值
acc_test_RF =acc_test_RF/6
# acc_test_RF = 1 - acc_test_RF
plt.plot(mean_fpr_RF, mean_tpr_RF, 'k-',color= ("blue") ,
         label='RF_MINI ROC (AUC = %0.2f,Acc_rate = %0.2f)' % (mean_auc_RF,acc_test_RF), lw=2)
#################################################################################
# SVM ROC curve

SVMmodel = svm.SVC(kernel='linear', probability=True)
# SVMmodel = svm.SVC(C=0.9, kernel='rbf', gamma=40, decision_function_shape='ovr', probability=True)

mean_tpr_SVM = 0.0
mean_fpr_SVM = np.linspace(0, 1, 100)

for train_index, test_index in cv.split(X, y):
    #probas_ = model.fit(X[train], y[train]).predict_proba(X[test])
    #print(X)

    SVMmodel.fit(X[train_index], y[train_index])
    # 测试集
    SVMprobas_ =SVMmodel.predict_proba(X[test_index])
    fpr_SVM, tpr_SVM, thresholds_SVM = roc_curve(y[test_index], SVMprobas_[:, 1])
    # # 训练集
    # SVMprobas_ =SVMmodel.predict_proba(X[train_index])
    # fpr_SVM, tpr_SVM, thresholds_SVM = roc_curve(y[train_index], SVMprobas_[:, 1])
    mean_tpr_SVM += interp(mean_fpr_SVM, fpr_SVM, tpr_SVM)  # 对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
    mean_tpr_SVM[0] = 0.0  # 初始处为0
    roc_auc_SVM = auc(fpr_SVM, tpr_SVM)
    #plt.plot(fpr, tpr, lw=1, label='ROC fold  (area = %0.2f)' % (roc_auc))
mean_tpr_SVM /= 6  # 在mean_fpr100个点，每个点处插值插值多次取平均
mean_tpr_SVM[-1] = 1.0  # 坐标最后一个点为（1,1）
mean_auc_SVM = auc(mean_fpr_SVM, mean_tpr_SVM)  # 计算平均AUC值

plt.plot(mean_fpr_SVM, mean_tpr_SVM, 'k-',color=("lime"),
         label='SVM ROC (area = %0.2f)' % mean_auc_SVM, lw=2)
#################################################################################
# LASSO ROC curve

from sklearn.linear_model import Lasso
alpha = 0.0001
LASSOmodel = Lasso(alpha=alpha)
mean_tpr_LASSO = 0.0
mean_fpr_LASSO = np.linspace(0, 1, 100)
for train_index, test_index in cv.split(X, y):
    #probas_ = model.fit(X[train], y[train]).predict_proba(X[test])
    #print(X)

    LASSOMODEL1=LASSOmodel.fit(X[train_index], y[train_index])
    #测试集
    LASSOprobas_ = LASSOmodel.predict(X[test_index])
    #print LASSOMODEL1.coef_
    fpr_LASSO, tpr_LASSO, thresholds_LASSO = roc_curve(y[test_index], LASSOprobas_)

    # 训练集
    # LASSOprobas_ = LASSOmodel.predict(X[train_index])
    # #print LASSOMODEL1.coef_
    # fpr_LASSO, tpr_LASSO, thresholds_LASSO = roc_curve(y[train_index], LASSOprobas_)
    mean_tpr_LASSO += interp(mean_fpr_LASSO, fpr_LASSO, tpr_LASSO)  # 对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
    mean_tpr_LASSO[0] = 0.0  # 初始处为0
    roc_auc_LASSO = auc(fpr_LASSO, tpr_LASSO)
    #plt.plot(fpr, tpr, lw=1, label='ROC fold  (area = %0.2f)' % (roc_auc))
mean_tpr_LASSO /= 6  # 在mean_fpr100个点，每个点处插值插值多次取平均
mean_tpr_LASSO[-1] = 1.0  # 坐标最后一个点为（1,1）
mean_auc_LASSO = auc(mean_fpr_LASSO, mean_tpr_LASSO)  # 计算平均AUC值

plt.plot(mean_fpr_LASSO, mean_tpr_LASSO, 'k-',color=("red"),
         label='LASSO ROC (area = %0.2f)' % mean_auc_LASSO, lw=2)


plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

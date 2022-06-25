#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 12:05:42 2022

@author: Christos
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import classification_report, roc_curve, auc




def visualize_missing_data(df, dataset_type, pth):
    
    
    fig=plt.figure(dpi=100, facecolor='w', edgecolor='w')
    sns.heatmap(df.isna().transpose(),
            cmap="YlGnBu",
            cbar_kws={'label': 'Missing Data'})
    fig.tight_layout(pad=2)
    plt.title(f'Missing values per feature. \n Dataset: {dataset_type}')
    plt.xticks([])
    plt.savefig(fname=os.path.join(pth, 'missing_values.png'),
                bbox_inches='tight')
    
    plt.savefig(fname=os.path.join(pth,
                                   f'target_distribution_{dataset_type}.png'),
                bbox_inches='tight')
    plt.show()





def plot_target_distribution(df, dataset_type, pth):
    
    fig=plt.figure(dpi=100, facecolor='w', edgecolor='w')
    fig.set_size_inches(20, 10)
    ax =sns.displot(df.TARGET_FLAG)
    plt.ylabel('# subjects')
    plt.xlabel('target')
    plt.xticks([0,1])
    plt.title(f'Distribution of target values. \n Dataset: {dataset_type}',
              style='oblique', fontweight='bold')
    fig.tight_layout(pad=2)
    
    
    plt.savefig(fname=os.path.join(pth, 
                                   f'target_distribution_{dataset_type}.png'),
                bbox_inches='tight')
    plt.show()
    
    

def plot_feature_correlation(df, dataset_type, pth):
    
    
    #df.drop(columns='TARGET_FLAG')
    
    fig=plt.figure(dpi=100, facecolor='w', edgecolor='w')
    fig.set_size_inches(20, 10)
    corr_df = df.corr()
    lower_triang_df = corr_df.where(np.tril(np.ones(corr_df.shape)).astype(np.bool_))
    sns.heatmap(lower_triang_df , annot=True, 
                cmap='coolwarm', clim=[-1,1],
                linewidths = 2, cbar=True, )
    
    
    
    
    
    plt.title(f'Feature correlation \n Dataset: {dataset_type}',
              style='oblique', fontweight='bold')
    #fig.tight_layout(pad=2)
    plt.savefig(fname=os.path.join(pth, 
                                   f'feature_correlation_{dataset_type}.png'),
                bbox_inches='tight')
    plt.show() 
    
    
def plot_roc_auc(y_test, y_proba, pth):
    

    
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    ax = plt.gca()
    ax.plot(
        fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    sns.despine(trim=True)
    plt.tight_layout()    
    plt.savefig(fname=os.path.join(pth, 
                                   'roc_curve.png'),
                bbox_inches='tight')
    plt.show()     
        
    
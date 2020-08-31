# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:42:31 2020

@author: Nicolai
"""

import seaborn as sns
import sys
sys.path.append("../../code/post_proc/")
import post_proc as pp
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    
    ###############################
    #    significantly better     #
    ###############################
    a = [0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,5,6,6,6,6,6,6,7,7,7,8,8,8,9]
    b = [0,1,2,3,3,3,4,4,4,5,5,5,7,8,8,9,9,9,9,9,9,10,10,10,10,10,10,10,10,11,11,11,11,11,11]
    b = [x + 5 for x in b]
    
    f,  ax_hist = plt.subplots()

    
    sns.distplot(a, ax=ax_hist)
    ax_hist.axvline(np.mean(a), color='g', linestyle='--')
    ax_hist.axvline(np.median(a), color='g', linestyle='-')
    
    
    sns.distplot(b, ax=ax_hist)
    ax_hist.axvline(np.mean(b), color='r', linestyle='--')
    ax_hist.axvline(np.median(b), color='r', linestyle='-')
    
    plt.legend({'mean(a)':np.mean(a),'median(a)':np.median(a), 'mean(b)':np.mean(b),'median(b)':np.median(b)})
    
    stat_result = pp.statsWilcoxon(a, b)
    plt.title("statsWilcoxon: " + "a is " + stat_result + " than b")
    plt.savefig("./pdf/sig_better.pdf", bbox_inches='tight')
    plt.show()
    
    ##############################
    #     signifcantly worse     #
    ##############################
    a = [0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,5,6,6,6,6,6,6,7,7,7,8,8,8,9]
    b = [0,1,2,3,3,3,4,4,4,5,5,5,7,8,8,9,9,9,9,9,9,10,10,10,10,10,10,10,10,11,11,11,11,11,11]
    a = [x + 5 for x in a]
    b = [x - 5 for x in b]
    
    f,  ax_hist = plt.subplots()

    sns.distplot(a, ax=ax_hist)
    ax_hist.axvline(np.mean(a), color='r', linestyle='--')
    ax_hist.axvline(np.median(a), color='r', linestyle='-')
    
    
    sns.distplot(b, ax=ax_hist)
    ax_hist.axvline(np.mean(b), color='g', linestyle='--')
    ax_hist.axvline(np.median(b), color='g', linestyle='-')
    
    plt.legend({'mean(a)':np.mean(a),'median(a)':np.median(a), 'mean(b)':np.mean(b),'median(b)':np.median(b)})
    
    stat_result = pp.statsWilcoxon(a, b)
    plt.title("statsWilcoxon: " + "a is " + stat_result + " than b")
    plt.savefig("./pdf/sig_worse.pdf", bbox_inches='tight')
    plt.show()
    
    #################################
    #      unsignificantly better   #
    #################################
    b = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,3,3,3,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,9,9]
    a = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,2,3,3,3,4,5,5,6,7,7,8,8,9,9,9,9,9,9,9,9,9,9,9,9]
    
    f,  ax_hist = plt.subplots()

    sns.distplot(a, ax=ax_hist)
    ax_hist.axvline(np.mean(a), color='g', linestyle='--')
    ax_hist.axvline(np.median(a), color='g', linestyle='-')
    
    
    sns.distplot(b, ax=ax_hist)
    ax_hist.axvline(np.mean(b), color='r', linestyle='--')
    ax_hist.axvline(np.median(b), color='r', linestyle='-')
    
    plt.legend({'mean(a)':np.mean(a),'median(a)':np.median(a), 'mean(b)':np.mean(b),'median(b)':np.median(b)})
    
    stat_result = pp.statsWilcoxon(a, b)
    plt.title("statsWilcoxon: " + "a is " + stat_result + " than b")
    plt.savefig("./pdf/unsig_better.pdf", bbox_inches='tight')
    plt.show()
    
    ##################################
    #      unsignificantly worse     #
    ##################################
    a = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,3,3,3,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,9,9]
    b = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,2,3,3,3,4,5,5,6,7,7,8,8,9,9,9,9,9,9,9,9,9,9,9,9]
    
    f,  ax_hist = plt.subplots()

    sns.distplot(a, ax=ax_hist)
    ax_hist.axvline(np.mean(a), color='r', linestyle='--')
    ax_hist.axvline(np.median(a), color='r', linestyle='-')
    
    
    sns.distplot(b, ax=ax_hist)
    ax_hist.axvline(np.mean(b), color='g', linestyle='--')
    ax_hist.axvline(np.median(b), color='g', linestyle='-')
    
    plt.legend({'mean(a)':np.mean(a),'median(a)':np.median(a), 'mean(b)':np.mean(b),'median(b)':np.median(b)})
    
    stat_result = pp.statsWilcoxon(a, b)
    plt.title("statsWilcoxon: " + "a is " + stat_result + " than b")
    plt.savefig("./pdf/unsig_worse.pdf", bbox_inches='tight')
    plt.show()
    
    #################################
    #   unsignificantly undecided   #
    #################################
    a = [0,0,0,0,0,1,1,1,1,1,2,2,2,3,3,4,5,5,5,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,9,9,9,9,9,9,9]
    b = [0,0,0,0,0,1,1,2,2,2,3,3,3,3,4,5,5,5,6,6,6,6,6,6,7,7,7,8,8,8,9,9,9,9,9,9,9,9,9,9,9]
    
    f,  ax_hist = plt.subplots()

    sns.distplot(a, ax=ax_hist)
    ax_hist.axvline(np.mean(a), color='g', linestyle='--')
    ax_hist.axvline(np.median(a), color='g', linestyle='-')
    
    
    sns.distplot(b, ax=ax_hist)
    ax_hist.axvline(np.mean(b), color='r', linestyle='--')
    ax_hist.axvline(np.median(b), color='r', linestyle='-')
    
    plt.legend({'mean(a)':np.mean(a),'median(a)':np.median(a), 'mean(b)':np.mean(b),'median(b)':np.median(b)})
    
    stat_result = pp.statsWilcoxon(a, b)
    plt.title("statsWilcoxon: " + "a is " + stat_result + " than b")
    plt.savefig("./pdf/unsig_undecided.pdf", bbox_inches='tight')
    plt.show()
    
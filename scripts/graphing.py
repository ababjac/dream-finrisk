import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np

#--------------------------------------------------------------------------------------------------#

def plot_points(y_pred, y_actual, title):
    plt.scatter(y_actual, y_pred)
    plt.title(title)
    
    plt.show()
    plt.clf()

#--------------------------------------------------------------------------------------------------#

def plot_confusion_matrix(y_pred, y_actual, title, path=None, color=None):
    if color == None:
        color = 'Oranges'

    plt.gca().set_aspect('equal')
    cf_matrix = confusion_matrix(y_actual, y_pred)
    if len(cf_matrix) != 2: #if it predicts perfectly then confusion matrix returns incorrect form
        val = cf_matrix[0][0]
        tmp = [val, 0]
        cf_matrix = np.array([tmp, [0, 0]])

    ax = sns.heatmap(cf_matrix, annot=True, cmap=color)

    ax.set_title(title+'\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values\n');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    ## Display the visualization of the Confusion Matrix.
    plt.tight_layout()
    #plt.savefig(path)
    plt.show()
    plt.clf()

#--------------------------------------------------------------------------------------------------#

def plot_auc(y_pred, y_actual, title, path=None):
    fpr, tpr, thresholds = roc_curve(y_actual, y_pred)
    roc_auc = auc(fpr, tpr)
    print('AUC:', roc_auc)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.title(title)
    plt.legend()
    #plt.savefig(path)
    plt.show()
    plt.clf()

#--------------------------------------------------------------------------------------------------#

def plot_pca(colors, pca, components, path=None):

    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    #print(labels)
    fig = px.scatter_matrix(
        components,
        labels=labels,
        dimensions=range(9),
        color=colors
    )

    fig.update_traces(diagonal_visible=False)
    #fig.write_image(path)
    fig.show()
    #fig.clf()

#--------------------------------------------------------------------------------------------------#

def plot_feature_importance(columns, importances, path):
    plt.figure(figsize=(16,8))
    sorted_idx = importances.argsort()
    sorted_idx = [i for i in sorted_idx if importances[i] > 0.01]
    plt.barh(columns[sorted_idx], importances[sorted_idx])
    plt.xlabel('Gini Values')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

#--------------------------------------------------------------------------------------------------#

#taken from here: https://stats.stackexchange.com/questions/288736/random-forest-positive-negative-feature-importance
def calculate_pseudo_coefficients(X, y, thr, probs, importances, nfeatures, path):
    dec = list(map(lambda x: (x> thr)*1, probs))
    val_c = X.copy()

    #scale features for visualization
    val_c = pd.DataFrame(StandardScaler().fit_transform(val_c), columns=X.columns)

    val_c = val_c[importances.sort_values('importance', ascending=False).index[0:nfeatures]]
    val_c['t']=y
    val_c['p']=dec
    val_c['err']=np.NAN
    #print(val_c)

    val_c.loc[(val_c['t']==0)&(val_c['p']==1),'err'] = 3#'fp'
    val_c.loc[(val_c['t']==0)&(val_c['p']==0),'err'] = 2#'tn'
    val_c.loc[(val_c['t']==1)&(val_c['p']==1),'err'] = 1#'tp'
    val_c.loc[(val_c['t']==1)&(val_c['p']==0),'err'] = 4#'fn'

    n_fp = len(val_c.loc[(val_c['t']==0)&(val_c['p']==1),'err'])
    n_tn = len(val_c.loc[(val_c['t']==0)&(val_c['p']==0),'err'])
    n_tp = len(val_c.loc[(val_c['t']==1)&(val_c['p']==1),'err'])
    n_fn = len(val_c.loc[(val_c['t']==1)&(val_c['p']==0),'err'])

    fp = np.round(val_c[(val_c['t']==0)&(val_c['p']==1)].mean(),2)
    tn = np.round(val_c[(val_c['t']==0)&(val_c['p']==0)].mean(),2)
    tp =  np.round(val_c[(val_c['t']==1)&(val_c['p']==1)].mean(),2)
    fn =  np.round(val_c[(val_c['t']==1)&(val_c['p']==0)].mean(),2)


    c = pd.concat([tp,fp,tn,fn],names=['tp','fp','tn','fn'],axis=1)
    pd.set_option('display.max_colwidth',900)
    c = c[0:-3]

    c.columns = ['TP','FP','TN','FN']

    c.plot.bar()
    plt.title('Relative Scaled Model Coefficients for True/False Positive Rates')
    plt.savefig(path)
    plt.close()
    
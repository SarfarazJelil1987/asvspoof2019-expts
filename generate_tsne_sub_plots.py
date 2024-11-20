from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


def tsne_transform(data, n_comp): 

    tsne = TSNE(n_components=n_comp, verbose=1, perplexity=50, learning_rate=200, n_iter=5000)
    z = tsne.fit_transform(data)
    return z
    
def pca_transform(data, n_comp):
    
    pca = PCA(n_components=n_comp)
    pca_result = pca.fit_transform(data)
    
    return pca_result
    
    
def load_data():
    # load the numpy file
    data_cqcc = np.load('tsne_data_CQCC/embeddings.npy')
    #data_cqcc = data_cqcc[0:1000,:]
    
    
    data_2d_ilrcc = np.load('tsne_data_2D_ILRCC/embeddings.npy')
    #data_2d_ilrcc= data_2d_ilrcc[0:1000, :]

    labels = np.load('tsne_data_CQCC/labels.npy')
    #labels = labels[0:1000]
    
    return [data_cqcc, data_2d_ilrcc, labels]

def plot_tsne_pca_subplots(cqcc_pca, cqcc_tsne, ilrcc_pca, ilrcc_tsne, labels):
  

    fig, axs = plt.subplots(2, 2)
    plt.rcParams['axes.titley'] = -0.01    # y is in axes-relative coordinates.
    plt.rcParams['axes.titlepad'] = -14
    plt.rcParams['axes.titlesize'] = 'medium'
    ### plot 2d_ilrcc pca 
    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = ilrcc_pca[:,0]
    df["comp-2"] = ilrcc_pca[:,1]
    
    p1 = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 2),
                data=df, s=2, ax=axs[0,0], legend=False)
                
    p1.set(xticklabels=[])
    p1.set_title('(a) PCA (2D-ILRCC)')
    p1.set(xlabel=None)
    
    p1.set(yticklabels=[])
    p1.set(ylabel=None)
                
    #### plot cqcc pca
    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = cqcc_pca[:,0]
    df["comp-2"] = cqcc_pca[:,1]
    
    p2 = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 2),
                data=df, s=2, ax=axs[0,1], legend=False)
                
    p2.set(xticklabels=[])
    p2.set_title('(b) PCA (CQCC)')
    p2.set(xlabel=None)
    
    p2.set(yticklabels=[])
    p2.set(ylabel=None)
    
    #### plot 2d_ilrcc tsne
    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = ilrcc_tsne[:,0]
    df["comp-2"] = ilrcc_tsne[:,1]
    
    p3 = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 2),
                data=df, s=2, ax=axs[1,0], legend=False)
    
    p3.set(xticklabels=[])
    p3.set_title('(c) t-SNE (2D-ILRCC)')
    p3.set(xlabel=None)
    
    p3.set(yticklabels=[])
    p3.set(ylabel=None)
    
    
    #### plot cqcc tsne
    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = cqcc_tsne[:,0]
    df["comp-2"] = cqcc_tsne[:,1]
    
    p4 = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 2),
                data=df, s=2, ax=axs[1,1], legend=False)
                
    p4.set(xticklabels=[])
    p4.set_title('(d) t-SNE (CQCC)')
    p4.set(xlabel=None)
    
    p4.set(yticklabels=[])
    p4.set(ylabel=None)
    
    
    
    # a desired name to the plot.
    fig.savefig('tsne_data_CQCC/tsne_pca_plot_2D_ILRCC_CQCC_50.png', bbox_inches='tight')
    
def main():

    # load the necessary data
    
    data_cqcc, data_2d_ilrcc, labels = load_data()
    
    # do pca for cqcc and 2d-ilrcc
    cqcc_pca = pca_transform(data_cqcc, 2)
    ilrcc_pca = pca_transform(data_2d_ilrcc, 2)
    
    # do tsne for cqcc and 2d-ilrcc
    cqcc_tsne = tsne_transform(data_cqcc, 2)
    ilrcc_tsne = tsne_transform(data_2d_ilrcc, 2)
    
    
    np.save('tsne_data_CQCC/cqcc_pca.npy',cqcc_pca)
    np.save('tsne_data_CQCC/ilrcc_pca.npy',ilrcc_pca)
    np.save('tsne_data_CQCC/cqcc_tsne.npy',cqcc_tsne)
    np.save('tsne_data_CQCC/ilrcc_tsne.npy',ilrcc_tsne)
    
    cqcc_pca = np.load('tsne_data_CQCC/cqcc_pca.npy')
    ilrcc_pca = np.load('tsne_data_CQCC/ilrcc_pca.npy')
    cqcc_tsne = np.load('tsne_data_CQCC/cqcc_tsne.npy')
    ilrcc_tsne = np.load('tsne_data_CQCC/ilrcc_tsne.npy')
    
    # call plot function
    plot_tsne_pca_subplots(cqcc_pca, cqcc_tsne, ilrcc_pca, ilrcc_tsne, labels)
     
    
if __name__ == "__main__":
    main()



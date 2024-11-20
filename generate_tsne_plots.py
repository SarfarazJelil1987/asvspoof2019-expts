from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

# load the numpy file
data = np.load('tsne_data_CQCC/embeddings.npy')
data = data[0:10000,:]
print(data.shape)

labels = np.load('tsne_data_CQCC/labels.npy')
labels = labels[0:10000]
print(labels.shape)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(data)

# transfomr using tsne
#tsne = TSNE(n_components=2, verbose=1, perplexity=10, random_state=123, n_iter=1000)
#z = tsne.fit_transform(pca_result)

z = pca_result

#y_gen = np.ones(2000)
#y_spoof = np.zeros(20000)
#y_train = np.concatenate((y_gen,y_spoof), axis=0)

#print(y_train.shape)
#print(y_train)

df = pd.DataFrame()
df["y"] = labels
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]


tsne_plot = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 2),
                data=df, s=20)#.set(title="T-SNE projection")
                
tsne_plot.set(xticklabels=[])
tsne_plot.set(xlabel=None)
tsne_plot.set(yticklabels=[])
tsne_plot.set(ylabel=None)

#plt.legend(labels=["Genuine", "Spoof"])
                
tsne_fig = tsne_plot.get_figure()
 
# use savefig function to save the plot and give
# a desired name to the plot.
tsne_fig.savefig('tsne_data_2D_ILRCC/tsne_plot_2D_ILRCC.png')



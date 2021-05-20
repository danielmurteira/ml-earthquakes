from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.npyio import save
import pandas
import math
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

def plot_classes(labels,lon,lat, alpha=0.5, edge = 'k'):
    """Plot seismic events using Mollweide projection.
    Arguments are the cluster labels and the longitude and latitude
    vectors of the events"""
    img = imread("data/mollweide_projection_sw.jpg")        
    plt.figure(figsize=(10,5),frameon=False)    
    x = lon/180*np.pi
    y = lat/180*np.pi
    ax = plt.subplot(111, projection="mollweide")
    print(ax.get_xlim(), ax.get_ylim())
    t = ax.transData.transform(np.vstack((x,y)).T)
    print(np.min(np.vstack((x,y)).T,axis=0))
    print(np.min(t,axis=0))
    clims = np.array([(-np.pi,0),(np.pi,0),(0,-np.pi/2),(0,np.pi/2)])
    lims = ax.transData.transform(clims)
    plt.close()
    plt.figure(figsize=(10,5),frameon=False)    
    plt.subplot(111)
    plt.imshow(img,zorder=0,extent=[lims[0,0],lims[1,0],lims[2,1],lims[3,1]],aspect=1)        
    x = t[:,0]
    y= t[:,1]
    nots = np.zeros(len(labels)).astype(bool)
    diffs = np.unique(labels)    
    ix = 0   
    for lab in diffs[diffs>=0]:        
        mask = labels==lab
        nots = np.logical_or(nots,mask)        
        plt.plot(x[mask], y[mask],'o', markersize=4, mew=1,zorder=1,alpha=alpha, markeredgecolor=edge)
        ix = ix+1                    
    mask = np.logical_not(nots)    
    if np.sum(mask)>0:
        plt.plot(x[mask], y[mask], '.', markersize=1, mew=1,markerfacecolor='w', markeredgecolor=edge)
    plt.axis('off')

def confusion_matrix(predictions):
    tp = fp = tn = fn = 0
    for i in range(len(predictions)-1):
        clusters_match = predictions[i] == predictions[i+1:]
        groups_match = faults[i] == faults[i+1:]
        tp += np.sum(np.logical_and(clusters_match,groups_match))
        tn += np.sum(np.logical_and(np.logical_not(clusters_match),np.logical_not(groups_match)))
        fp += np.sum(np.logical_and(clusters_match,np.logical_not(groups_match)))
        fn += np.sum(np.logical_and(np.logical_not(clusters_match),groups_match))
    return tp, tn, fp, fn

def calc_kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters).fit(X)
    predictions = kmeans.predict(X)
    return predictions

def plot_elbow(X):
    kneigh = KNeighborsClassifier().fit(X,np.zeros(X.shape[0]))
    distaux, _ = kneigh.kneighbors(n_neighbors = 4)
    dist = np.max(distaux, axis = 1)
    dist = np.sort(dist)
    dist = dist[::-1]
    plt.figure(4)
    plt.title('Elbow')
    plt.plot(dist[:], '-r')
    plt.savefig('report/elbow.png',dpi=300)
    plt.show()

def calc_dbscan(X, eps, min_pts):
    dbscan = DBSCAN(eps, min_samples=min_pts).fit(X)
    predictions = dbscan.labels_
    return predictions

def calc_gaussian_mixture(X, n_components):
    gm = GaussianMixture(n_components).fit(X)
    predictions = gm.predict(X)
    return predictions

def run_algorithm(alg_name, matrix, file):
    precision = []
    recall = []
    randindex = []
    arandindex = []
    f1 = []
    silhouette = []
    scale = []
    best_results = np.zeros(6)
    best_indexes = np.zeros(6)

    start=0
    end=0
    if alg_name == "kmeans":
        start = 2
        end = 101
    elif alg_name == "dbscan":
        start = 100
        end = 401
    elif alg_name == "gmm":
        start = 2
        end = 51

    for i in range(start,end):
        print('i:',i)
        if alg_name == "kmeans":
            predictions = calc_kmeans(matrix, i)
        elif alg_name == "dbscan":
            predictions = calc_dbscan(matrix, i, 4)
        elif alg_name == "gmm":
            predictions = calc_gaussian_mixture(matrix, i)
        tp, tn, fp, fn = confusion_matrix(predictions)
        pre = tp/(tp+fp)
        precision.append(pre)
        rec = tp/(tp+fn)
        recall.append(rec)
        ri = (tp+tn)/((len(matrix)*len(matrix)-1)/2)
        randindex.append(ri)
        ari = adjusted_rand_score(faults, predictions)
        arandindex.append(ari)
        f = 2*(pre * rec)/(pre + rec)
        f1.append(f)
        sil = silhouette_score(matrix, predictions)
        silhouette.append(sil)
        scale.append(i)
        if(pre>best_results[0]):
            best_results[0] = pre
            best_indexes[0] = i
        if(rec>best_results[1]):
            best_results[1] = rec
            best_indexes[1] = i
        if(ri>best_results[2]):
            best_results[2] = ri
            best_indexes[2] = i
        if(ari>best_results[3]):
            best_results[3] = ari
            best_indexes[3] = i
        if(f>best_results[4]):
            best_results[4] = f
            best_indexes[4] = i
        if(sil>best_results[5]):
            best_results[5] = sil
            best_indexes[5] = i

    if alg_name == "kmeans":
        plt.figure(0)
        plt.xlim(left=2)
        plt.xlim(right=100)
        file.write('\nBEST K:\n')
    elif alg_name == "dbscan":
        plt.figure(1)
        plt.xlim(left=100)
        plt.xlim(right=400)
        file.write('\nBEST EPS:\n')
    elif alg_name == "gmm": 
        plt.figure(2)
        plt.xlim(left=2)
        plt.xlim(right=50)
        file.write('\nBEST N:\n')
    plt.title(alg_name)
    plt.plot(scale, precision[:], '-r', label='Precision')
    plt.plot(scale, recall[:], '-g', label='Recall')
    plt.plot(scale, randindex[:], '-b', label='RI')
    plt.plot(scale, arandindex[:], '-m', label='ARI')
    plt.plot(scale, f1[:], '-c', label='F1')
    plt.plot(scale, silhouette[:], '-y', label='Silhouette')
    plt.legend(ncol=2)
    plt.savefig('report/err_'+alg_name+'.png',dpi=300)
    plt.show()

    file.write(' precision [' + str(best_indexes[0]) + '] = ' + str(round(best_results[0],2)) + '\n')
    file.write('    recall [' + str(best_indexes[1]) + '] = ' + str(round(best_results[1],2)) + '\n')
    file.write(' randindex [' + str(best_indexes[2]) + '] = ' + str(round(best_results[2],2)) + '\n')
    file.write('arandindex [' + str(best_indexes[3]) + '] = ' + str(round(best_results[3],2)) + '\n')
    file.write('        f1 [' + str(best_indexes[4]) + '] = ' + str(round(best_results[4],2)) + '\n')
    file.write('silhouette [' + str(best_indexes[5]) + '] = ' + str(round(best_results[5],2)) + '\n')
    best_i = int(best_indexes[3])
    best_i_err = round(best_results[3],2)
    return best_i, best_i_err

def save_map(alg_name, classes, matrix_latlng):
    lats = matrix_latlng[:,0]
    lngs = matrix_latlng[:,1]
    plot_classes(classes,lngs,lats)
    plt.savefig('report/map_'+alg_name+'.png',dpi=300)

#load data
data = pandas.read_csv("data/dataset.csv")
matrix = np.zeros((len(data),4))
faults = data.fault.values
lat = data.latitude.values
lng = data.longitude.values
matrix_latlng = np.zeros((len(data),3)) #to know lats and lngs without -1 faults to run plot_classes

RADIUS = 6371
x = []
y = []
z = []

for i in range(len(data)): 
    x.append(RADIUS * math.cos(lat[i] * math.pi/180) * math.cos(lng[i] * math.pi/180))
    y.append(RADIUS * math.cos(lat[i] * math.pi/180) * math.sin(lng[i] * math.pi/180))
    z.append(RADIUS * math.sin(lat[i] * math.pi/180))

matrix[:,0] = x
matrix[:,1] = y
matrix[:,2] = z   
matrix[:,3] = faults

matrix_latlng[:,0] = lat
matrix_latlng[:,1] = lng
matrix_latlng[:,2] = faults

#filter -1 fault values from matrix and faults vector; also, delete fault values column from attributes matrix
maux = matrix[matrix[:,3]!=-1,:]
matrix = np.delete(maux, 3, 1)
faults = np.asarray([a for a in faults if a not in set([-1])])
matrix_latlng = matrix_latlng[matrix_latlng[:,2]!=-1,:]

#the elbow region is the eps range for dbscan
plot_elbow(matrix)

#run algorithms
file = open("report/report.txt", "w")
best_k, best_k_err = run_algorithm('kmeans', matrix, file)
best_eps, best_eps_err = run_algorithm('dbscan', matrix, file)
best_n, best_n_err = run_algorithm('gmm', matrix, file)

#save maps with lowest ARI error value
file.write('\nbest k: ' + str(best_k) + ' (ARI error = ' + str(best_k_err) + ')\n')
kmeans_classes = calc_kmeans(matrix, best_k)
save_map('kmeans', kmeans_classes, matrix_latlng)

file.write('best eps: ' + str(best_eps) + ' (ARI error = ' + str(best_eps_err) + ')\n')
dbscan_classes = calc_dbscan(matrix, best_eps, 4)
save_map('dbscan', dbscan_classes, matrix_latlng)

file.write('best n: ' + str(best_n) + ' (ARI error = ' + str(best_n_err) + ')\n')
gmm_classes = calc_gaussian_mixture(matrix, best_n)
save_map('gmm', gmm_classes, matrix_latlng)
file.close()

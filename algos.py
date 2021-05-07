from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import ectkmeans
import CE3

def run_ectkmeans(data, labels):
    predicts = ectkmeans.fit(data)
    _nmi = nmi(labels, predicts, average_method='geometric')

    print('ECT-KMeans Algorithm Finished!')
    return _nmi, predicts


def run_kmeans(data, labels):
    predicts = KMeans().fit(X=data).labels_
    _nmi = nmi(labels, predicts, average_method='geometric')

    print('KMeans Algorithm Finished!')
    return _nmi


import numpy as np
from sklearn.datasets.samples_generator import make_blobs

########################################### define cluster - given by instructor
class cluster:

    def __init__(self):
        pass

    def fit(self, X):
        pass

#define kmeans - subclass of cluster
class kmeans(cluster):
    
    def __init__(self):
        super(kmeans, self).__init__()
        
        #set parameters
        self.k = 5
        self.max_iterations = 100
    
    def fit(self, X):
        
        super(kmeans, self).fit(X)
        
        # create a list of hypothesis outcomes
        hypothesis = np.zeros(X.shape[0], dtype = object)
        
        # randomly assign centroid values from X
        centroids = kmeans.rand_centroids(self.k)
        
        for current_iter in range(self.max_iterations):
            for idx in range(X.shape[0]):
                
                vec = X[idx,:]
                
                # create a new list of size k to hold the distances between
                # the value in X and each of the k centroids
                distances = np.zeros(self.k, dtype = object)
                
                for cluster in range(self.k):
                    cent_mean = centroids[cluster,:]
                    # calculate the distance between current X and centroid
                    distances[cluster] = np.linalg.norm(vec-cent_mean)
                    
                # assign the index of the smallest distance to be our 
                # cluster hypothesis for the current X
                hypothesis[idx] = np.argmin(distances)
                
            # recalculate centroids at k to be the mean of the new cluster 
            # values
            for i in range(self.k):
                whr = np.where(hypothesis == i)
            
                mean_k = np.mean(X[whr[0],:], axis = 0)
                centroids[i,:] = mean_k
            
        return(hypothesis, centroids)
            
    # fit with aproximately the same cluster size
    def fit_extended(self, X, balanced = False):
        
        # balanced = False, size of clusters can vary
        # similar to doing 'fit' on X
        if (balanced == False):
            print(3)
            kmeans.fit(self, X)
            
        # size of cluster is relatively consistent    
        else:
            super(kmeans, self).fit(X)
        
            # create a list of hypothesis outcomes and distance values
            hypothesis = np.zeros(X.shape[0], dtype = object)
            hypothesis_vals = np.zeros(X.shape[0], dtype = object)
            
            # randomly assign centroid values from X
            centroids = kmeans.rand_centroids(self.k)
            
            # get the integer size of each cluster 
            max_cap = round(X.shape[0]/self.k)
            
            # 
            for current_iter in range(self.max_iterations):
                for idx in range(X.shape[0]):
                    
                    vec = X[idx,:]
                    
                    # create a new list of size k to hold the distances between
                    # the value in X and each of the k centroids
                    distances = np.zeros(self.k, dtype = object)
                    
                    #create an array to hold the min, max and count of centroid
                    c_count = np.zeros(self.k, dtype = object)
                    c_max = np.zeros(self.k, dtype = object)
                    c_min = np.zeros(self.k, dtype = object)
                    max_value = 0
                    
                    # calculate the distance between the centroids and the 
                    # current X
                    for cluster in range(self.k):
                        cent_mean = centroids[cluster,:]
                        
                        # calculate the distance
                        distances[cluster] = np.linalg.norm(vec-cent_mean)
                        
                    # assign the distance and it's index to be our cluster 
                    #hypothesis for the current X
                    h_cluster = np.argmin(distances)
                    d_val = np.min(distances)
                    hypothesis_vals[idx] = d_val
                    
                    # recalculate the index of the min and max value index
                    c_max[h_cluster] = np.argmax(max(hypothesis_vals[c_max[h_cluster]], d_val))
                    c_min[h_cluster] = np.argmin(min(hypothesis_vals[c_min[h_cluster]], d_val))
                    max_value = np.max(c_max)
                    
                    # check if we have reached max_cap
                    # if we haven't, assign the cluster and update count
                    if (c_count[h_cluster] < max_cap):
                        hypothesis[idx] = h_cluster
                        c_count[h_cluster] += 1 
                        
                    # if this cluster has past max_cap, then push the last 
                    # value of the cluster to the next cluster
                    else:
                        c_count, c_max, c_min, hypothesis = kmeans.shift_cluster(h_cluster, d_val, max_value, c_count, c_max, c_min, hypothesis)
                    
                    
                # recalculate centroids at k to the mean of the new cluster values
                for i in range(self.k):
                    whr = np.where(hypothesis == i)
                    
                    mean_k = np.mean(X[whr[0],:], axis = 0)
                    centroids[i,:] = mean_k
            
            return (hypothesis, centroids)

    def shift_cluster(self, h_cluster, d_val, max_cap, idx, c_count, c_max, c_min, hypothesis, hypothesis_val, change = True):
            
        while (change == True and h_cluster < self.k-1):
            
            # get d_val into the cluster and push the max of that cluster into 
            # the next cluster
            hypothesis[idx] = h_cluster
            # move the max value of the current cluster to the next one
            hypothesis[c_max[h_cluster]] = h_cluster + 1
            
            # recalculate the max and min of both new clusters
            c_max[h_cluster] = np.argmax(max(c_max[h_cluster],d_val))
            c_min[h_cluster] = np.argmin(min(c_min[h_cluster], d_val))
            c_max[h_cluster+1] = np.argmax(max(c_max[h_cluster+1], hypothesis_val[max]))
            c_min[h_cluster+1] = np.argmin(min(c_min[h_cluster+1], d_val))
            
            # adjust count of these two clusters
            c_count[h_cluster] -= 1
            c_count[h_cluster+1] += 1
            
            if c_count[h_cluster + 1] <= max_cap:
                change = False
            else:
                c_count, c_max, c_min, hypothesis = kmeans.shift_cluster(h_cluster, d_val, max_cap, idx, c_count, c_max, c_min, hypothesis)
                change = False
                
        if (change == True and h_cluster == self.k-1):
            c_count, c_max, c_min, hypothesis = kmeans.shift_cluster_left(h_cluster, d_val, max_cap, idx, c_count, c_max, c_min, hypothesis)
            
        return (c_count, c_max, c_min, hypothesis)
        
    
    def shift_cluster_left(h_cluster, d_val, max_cap, idx, c_count, c_max, c_min, hypothesis, hypothesis_val, change = True):
        while (change == True):
            # get d_val into the cluster and push the max of that cluster into 
            # the next cluster
            hypothesis[idx] = h_cluster
            # move the min value of the current cluster to the next one
            hypothesis[c_min[h_cluster]] = h_cluster - 1
            # recalculate the max and min of both new clusters
            c_max[h_cluster] = np.argmax(max(c_min[h_cluster],d_val))
            c_min[h_cluster] = np.argmin(min(c_min[h_cluster], d_val))
            c_max[h_cluster-1] = np.argmax(max(c_max[h_cluster-1], hypothesis_val[max]))
            c_min[h_cluster-1] = np.argmin(min(c_min[h_cluster-1], d_val))
            
            #adjust count of these two clusters
            c_count[h_cluster] -= 1
            c_count[h_cluster-1] += 1
            
            # check that the new cluster is less than our cap
            if c_count[h_cluster - 1] <= max_cap:
                change = False
            else:
                c_count, c_max, c_min, hypothesis = kmeans.shift_cluster_left(h_cluster, d_val, max_cap, idx, c_count, c_max, c_min, hypothesis)
    
    # set a different number of clusters
    def set_k(self, new_k):
        self.k = new_k
     
    # set a different number of max_iteration    
    def set_max_iterations(self, new_iterations):
        self.max_iterations = new_iterations
        
    # get cluster amounts - k
    def get_k(self):
        return self.k
    
    # get maximum iteration limit
    def get_max_iterations(self):
        return self.max_iterations
    
    #randomly select k centroids from our array X
    def rand_centroids(k):
        pos_centroids = np.random.randint(0, X.shape[0], k)
        
        first_centroids = np.zeros((k, X.shape[1]))
        j = 0
        for i in pos_centroids:
            first_centroids[j] = X[i]
            j += 1
        return first_centroids
    
    # recalculate centroid values using the mean of the current clusters
    def recalculate_centroid(k, hypothesis):
        for cluster in range(k):
            centroids = np.zeros(k, dtype = object)
            whr = np.where(hypothesis == k)
            centroids[k,:] = np.mean(X[whr[0]],axis = 0)
    
X, cluster_assignments = make_blobs(n_samples=200, centers=4, cluster_std=0.60, random_state=0)   
test = kmeans()
test.set_k(4)
test.set_max_iterations(100)
#test_x, test_y, train_x, train_y
hypothesis, centroids = test.fit_extended(X, balanced = True)

# make balanced optional
# do test and train data
# return as a txt file

    
import numpy as np


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centelecture and notebook
    #     # Choose 1st center randomly and use Euclidean distance to calculate rs according to the other centers.
    centers = generator.choice(n, size=1)
    min_distance = np.sum(np.power((x - x[centers[0]]), 2), axis=1)
    centers = np.append(centers, np.argmax(min_distance))
    for i in range(2, n_cluster):
        this_distance = np.sum(np.power((x - np.expand_dims(x[centers[:i]], axis=1)), 2), axis=2)
        min_distance = np.min(this_distance, axis=0)
        centers = np.append(centers, np.argmax(min_distance))
    centers = list(centers)
    # DO NOT CHANGE CODE BELOW THIS LINE
    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers



def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)

class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        iteration = 0
        r = np.zeros((N, self.n_cluster), dtype=int)
        centroids = x[self.centers]

        new_distance = np.sum(np.power(x - np.expand_dims(centroids, axis=1), 2), axis=2)
        y = np.argmin(new_distance, axis=0)
        new_J = np.sum(np.min(new_distance, axis=0))
        for i in range(N):
            r[i, y[i]] = 1
        # for i in range(N):
        #     new_distance = np.sum(np.power((x[i] - centroids), 2), axis=1)
        #     membership = np.argmin(new_distance)
        #     r[i, membership] = 1
        #     y[i] = membership
        #     new_J = new_J + new_distance[membership]

        while iteration < self.max_iter:
            new_centroids = np.divide(np.dot(r.T, x).T, np.sum(r, axis=0)).T
            r0 = np.where(np.sum(r, axis=0) == 0)[0]
            if len(r0) != 0:
                new_centroids[r0[0]] = centroids[r0[0]]

            centroids = new_centroids
            iteration += 1
            J = new_J
            new_distance = np.sum(np.power(x - np.expand_dims(centroids, axis=1), 2), axis=2)
            y = np.argmin(new_distance, axis=0)
            new_J = np.sum(np.min(new_distance, axis=0))
            if np.abs((new_J - J)) <= self.e:
                break
            r = np.zeros((N, self.n_cluster), dtype=int)
            for i in range(N):
                r[i, y[i]] = 1
            # for i in range(N)
        # DO NOT CHANGE CODE BELOW THIS LINE
        return centroids, y, self.max_iter

        


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        # centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        # iteration = 0
        # J = 0
        # new_J = 0
        # r = np.zeros((N, self.n_cluster), dtype=int)
        # member = np.zeros(N)
        # centroid_labels = np.zeros(self.n_cluster, dtype=int)
        #
        # centroids = x[centers]
        # for i in range(N):
        #     new_distance = np.sum(np.power((x[i] - centroids), 2), axis=1)
        #     membership = np.argmin(new_distance)
        #     r[i, membership] = 1
        #     member[i] = membership
        #     new_J = new_J + new_distance[membership]
        #
        # while np.abs((new_J - J)) > self.e or iteration < self.max_iter:
        #     new_centroids = (np.dot(r.T, x).T / np.sum(r, axis=0)).T
        #     r0 = np.where(np.sum(r, axis=0) == 0)[0]
        #     if len(r0) != 0:
        #         new_centroids[r0[0]] = centroids[r0[0]]
        #     centroids = new_centroids
        #     iteration += 1
        #     J = new_J
        #     new_J = 0
        #     for i in range(N):
        #         new_distance = np.sum(np.power((x[i] - centroids), 2), axis=1)
        #         membership = np.argmin(new_distance)
        #         r[i] = 0
        #         r[i, membership] = 1
        #         member[i] = membership
        #         new_J = new_J + new_distance[membership]
        this_kmeans = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
        centroids, membership, _ = this_kmeans.fit(x, centroid_func)
        centroid_labels = np.zeros(self.n_cluster, dtype=int)
        for i in range(self.n_cluster):
            all_label = y[membership == i]
            if len(all_label) != 0:
                centroid_labels[i] = np.argmax(np.bincount(all_label))

        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        membership = np.argmin(np.sum(np.power(x - np.expand_dims(self.centroids, axis=1),2), axis=2), axis=0)
        labels = self.centroid_labels[membership]

        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)
        

def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE
    x, y, _ = image.shape
    new_im = np.zeros((x, y, 3))
    for i in range(x):
        for j in range(y):
            r_index = np.argmin(np.sum(np.power((image[i,j] - code_vectors),2), axis = 1))
            new_im[i,j] = code_vectors[r_index]
    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im


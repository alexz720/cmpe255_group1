#PCA 1 
#Custom PCA class which has methods for transforming, inverse transforming, and getting loadings (essentially components)
#Note: some normalization issues could be addressed but it all works
class MPCA:
    #Note: can change n_components = None for a selection based on variable per cent of variance explained
    def __init__(self, n_components=1):
        # Initialize class variables that will be used later
        self.n_components = n_components
        self.projection_matrix = None
        self.evr_ = None
    #Perform PCA on the input data and returns the transformed data
    def fit_transform(self, X):
        # Convert to mean = 0
        X_cent = X - X.mean(axis=0)

        # Compute the svd as described in the report
        u, s, vh = np.linalg.svd(X_cent, full_matrices=False)

        # Compute the projection matrix which we will use to transform the data matrix
        self.projection_matrix = vh.T

        # Project the data onto the principal components
        X_pca = X_cent.dot(self.projection_matrix)

        # NOTE: this is explained in great detail the report.
        # Ax X_pca = us, we can find the explained variance ratios by using
        # The normalized square norms of the data points along the principal components
        # This is one of the benefits of using the SVD method as it doesn't require finding the covariance matrix
        # Alternatively, we can think about it in the context of eigenvalues -- as the
        # singular values are related to these via square rooting, it makes sense that the distances should be squared
        # as X_pca = us is something like the square root of the eigendecomposition of X^TX
        # As I explain in the paper, this is a bit more efficient than the covariance one.
        evr = np.sum(X_pca**2, axis=0) / np.sum(X_cent**2)
        
        # Assign the value of evr_ attribute
        self.evr_ = evr

        # Compute the cumulative explained variance ratio [note: no longer used as class var*, 4/29]
        cum_evr = np.cumsum(evr)

        #NOTE: it currently defaults to 1 but could be useful in a future imnplementation
        if self.n_components is None:
            self.n_components = np.sum(cum_evr < 0.75) + 1

        return X_pca[:, :self.n_components]

    # function to project data onto the principal components after the model has been fit
    def transform(self, X):
        # Center the data
        X_cent = X - X.mean(axis=0)
        # Project the data onto the principal components
        X_pca = X_cent.dot(self.projection_matrix)
        return X_pca[:, :self.n_components]   

    # Note components typically refer to slightly different concepts,
    # with components being the principal components themselves, and loadings 
    # being the normalized versions of their coefficients,
    #  but here they're all normalized so I only left the loadings.
    # There may be some instability with non-normalized data
    def get_loadings(self):
        loadings = self.projection_matrix[:, :self.n_components]
        return loadings
    
    #Inverse transform function as the projection matrix is orthogonal
    #we can simply multiply by its transpose here.
    def inverse_transform(self, X_transformed):
        # Project the transformed data back onto the original feature space
        X_original = X_transformed.dot(self.projection_matrix.T)

        return X_original

#Basic Linear regression class that uses gradient descent. It is 
#then used by the PCA regression class we define later.
#This is basically following exactly what we had on homework #2
class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta = None

    def fit(self, X, y):
        # Initialize the model parameters
        self.theta = np.zeros(X.shape[1]).reshape(-1, 1)

        # Perform gradient descent
        for i in range(self.num_iterations):
            # Calculate the predicted values of y
            y_pred = np.dot(X, self.theta)
            # Calculate the error between the predicted values and the actual values of y
            error = y_pred - y
            # Calculate the gradient of the cost function with respect to theta
            gradient = np.dot(X.T, error) / len(X)
            # Update the values of theta
            self.theta -= self.learning_rate * np.dot(X.T, error) / len(X)

    def predict(self, X):
        # Calculate the predicted values of y using the learned parameters
        y_pred = np.dot(X, self.theta)

        return y_pred



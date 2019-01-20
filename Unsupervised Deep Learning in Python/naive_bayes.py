# Assumptions: p(x | y) ~ mvnorm(mu, var)
# Important: var is a diagonal matrix
# Note: p(y | x) = p(x | y) * p(y) / p(x) for each y
# Since p(x) is constant in y, only p(x | y) * p(y) can be calculated for hard classification
# If exact probability is required, p(x) should also be calculated, which is not done here

from numpy import mean, argmax, log, zeros
from scipy.stats import multivariate_normal

class gaussian_naive_bayes(object):
	def fit(self, X, Y, smoothing = 0.01):
		self.gaussians = dict()
		self.priors = dict()
		labels = set(Y)
		for label in labels:
			x = X[Y == label]
			self.gaussians[label] = {
				'mean': x.mean(axis = 0),
				'var': x.var(axis = 0) + smoothing
			}
			self.priors[label] = float(len(Y[Y == label]))/len(Y)
	
	def predict(self, X):
		rows, cols = X.shape
		classes = len(self.gaussians)
		preds = zeros((rows, classes))
		for clas, gauss in self.gaussians.items():
			mean, var = gauss['mean'], gauss['var']
			preds[:, clas] = multivariate_normal.logpdf(X, mean = mean, cov = var) + log(self.priors[clas])
		
		return argmax(preds, axis = 1)
	
	def score(self, X, Y):
		pred = self.predict(X)
		return mean(pred == Y)


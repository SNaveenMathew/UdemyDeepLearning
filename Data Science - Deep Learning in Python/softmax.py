from numpy import exp

def softmax(wtx):
	expo = exp(wtx)
	ans = expo/expo.sum(axis = 1, keepdims = True)
	return ans


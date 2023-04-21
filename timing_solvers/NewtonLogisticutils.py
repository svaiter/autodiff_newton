import numpy as np
import numpy.random as npr
import jax
import jax.numpy as jnp

import sklearn.preprocessing

def generateData(n1,n2,pDim):
	n = n1+n2
	# Labels
	y = np.ones((n,1))
	y[n2:n] = -1
	y = y.reshape((n,1))

	# Design matrix, standard gaussian, entries corresponding to label 1 are shifted.
	X = npr.normal(0,1,(n,pDim))
	X[n2:n,:] = X[n2:n,:] + 1/np.sqrt(pDim)
	X[0:n2,:] = X[0:n2,:] -1/np.sqrt(pDim)
	X = sklearn.preprocessing.scale(X)
	X = np.c_[ np.ones(n), X]
	return jnp.array(X), jnp.array(y)



def logisticLossjax(X,y,beta,lbda, w):
	# Evaluation of logistic loss, X design, y labels, beta parameters, lbda L2 regularization
	temp1 = jnp.dot(X,beta) * y
	return jnp.sum( jnp.log(1 + jnp.exp(-temp1)) * w )  + jnp.exp(lbda) * 0.5 * jnp.sum(beta[0:]**2)



logisticGradjax = jax.jacrev(logisticLossjax,2)
logisticHessjax = jax.jacfwd(logisticGradjax,2)
logisticDxw = jax.jacfwd(logisticGradjax,4)

def newtonExplicitStep(X,y,beta,lbda, w):
	## Newton direction, explicit inversion
	n = X.shape[0]
	pDim = X.shape[1]
	
	H = logisticHessjax(X,y,beta,lbda, w).reshape((pDim , pDim ))
	g = logisticGradjax(X,y,beta,lbda, w)
	return -jnp.linalg.solve(H,g), g
   

def newtonIncrement(X,y,beta,lbda, w,step = None):
	# Wolfe condition
	def WolfeLineSearchjax(f, g, d, x,  c1, c2, min_step, max_step, X,y,lbda, w, max_iterations=50):
		# We start by precomputing some values
		step = (min_step + max_step) / 2
		fx = f(x)
		m = np.sum(-g(x) * d)


		# Then we divide the step size by 2 as long as one of the conditions is met
		condition_1 = f(x + step * d) > fx + c1 * step * m
		condition_2 = np.sum(g(x + step * d) * d) < c2 * m
		n_iterations = 0

		while (condition_1 or condition_2) and n_iterations < max_iterations:
			if condition_1:
				max_step = step
			else:
				min_step = step
			step = (min_step + max_step) / 2

			condition_1 = f(x + step * d) > fx + c1 * step * m
			condition_2 = np.sum(g(x + step * d) * d) < c2 * m
			n_iterations += 1
		return step


	# Compute the descent direction
	direction, grad = newtonExplicitStep(X,y,beta,lbda, w)
	
	if step==None:
		# Find the optimal step size using Wolfe line search
		
		logisticLossBetajax = lambda beta: logisticLossjax(X,y,beta,lbda, w)
		logisticGradBetajax = jax.jacrev(logisticLossBetajax,0)
		step = WolfeLineSearchjax(logisticLossBetajax, logisticGradBetajax, direction, beta, c1=0.001, c2=0.7, min_step=0, max_step=1.0, X = X, y= y, lbda = lbda, w = w)
	
	# Move beta along the descent direction and memorise old beta
	return beta + step * direction, grad, step

def newton(X,y,beta0,lbda, w,step = None, N=50, stopGrad = True, verb = False):
	beta = beta0.copy()
	for k in range(N-1):
		# Stop gradient increments
		if stopGrad:
			beta, grad, step = newtonIncrement(
				jax.lax.stop_gradient(X),
				jax.lax.stop_gradient(y),
				jax.lax.stop_gradient(beta),
				jax.lax.stop_gradient(lbda), 
				jax.lax.stop_gradient(w), step 
			)
		else:
			beta, grad, step = newtonIncrement(X, y, beta, lbda, w, step)
		## Gradient norm
		subopt = jnp.linalg.norm(grad)
		if verb: ## print information on iterations
			print("It: ", end='')
			print(k, end='')
			print(", subopt: ", end='')
			print(np.format_float_scientific(subopt, precision=2), end='')
			print(", step: ", end='')
			print(np.format_float_scientific(step, precision = 2))
		if subopt < 1e-10:  ## stoping criterion							
			break
	## One last step
	beta,grad,step = newtonIncrement(X, y, beta, lbda, w, step)
	return beta
	
## Test
#betaInit = np.zeros((pDim,1)) 
#betaInit[0] = 0.5
#betaZero = jnp.array(betaInit) 
#beta = newton(X,y,betaZero,lbda, w, verb = True)


def diffImpNorm(X,y,beta0,lbda, w, solutionVariable = None):
	## Bacward mode, derivative of \|x(b)\|^2 / 2 where x(b) is solution to the 
	## logistic problem
	## Explicit matrix computation
	n = X.shape[0]
	pDim = X.shape[1]
	
	if solutionVariable is None:
		beta = newton(X,y,beta0,lbda, w)
	else:
		beta = solutionVariable.copy()
	vrhs = beta
	
	H = logisticHessjax(X,y,beta,lbda, w).reshape((pDim , pDim ))
	M = logisticDxw(X,y,beta,lbda, w).reshape((pDim , n ))
	
	return -M.T.dot(jnp.linalg.solve(H, vrhs))

def diffImpNorm2(X,y,beta0,lbda, w, solutionVariable = None):
	## Bacward mode, derivative of \|x(b)\|^2 / 2 where x(b) is solution to the 
	## logistic problem
	## Matrix vector product with autodiff
	n = X.shape[0]
	pDim = X.shape[1]
	
	if solutionVariable is None:
		beta = newton(X,y,beta0,lbda, w)
	else:
		beta = solutionVariable.copy()
	vrhs = beta

	H = logisticHessjax(X,y,beta,lbda, w).reshape((pDim , pDim ))
	M = logisticDxw(X,y,beta,lbda, w).reshape((pDim , n ))
	
	return -jax.jacrev(lambda w: jax.lax.stop_grad(jnp.linalg.solve(H, vrhs)) * logisticGradjax(X,y,beta,lbda, w), 4)(w) 


def NewtonNorm(X,y,beta0,lbda,w,  stopGrad = True):
	## Evaluate norm of the solution
	return jnp.sum(newton(X,y,beta0,lbda, w, stopGrad = stopGrad)**2)/2


def diffNewtonNorm(X,y,beta0,lbda,w,  stopGrad = True):
	## Derivative of NewtonNorm
	return jax.jacrev(NewtonNorm, 4)(X,y,beta0,lbda, w, stopGrad = stopGrad)




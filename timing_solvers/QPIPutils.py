import numpy as np
import numpy.random as npr
import jax
import jax.numpy as jnp

def generateRandomQPdata(n, m, p):
	## Generate random data for the quadratic program
	Q = jnp.array(npr.uniform(size = (n,n*2)) * 2 - 1)
	Q = Q.dot(Q.T)
	A = jnp.array(npr.uniform(size = (m,n)) * 2 - 1)
	G = jnp.array(npr.uniform(size = (p,n)) * 2 - 1)
	q = jnp.array(npr.uniform(size = (n,1)) * 2 - 1)
	h = jnp.array(npr.uniform(size = (p,1)) * 2 - 1)
	b = jnp.array(npr.uniform(size = (m,1)) * 2 - 1)
	return Q, A, G, q, b, h

def IPupdate(x, s, z, y, Q, A, G, q, b, h):
	## Performs one interior point update
	## x, s, z, y: primal slack and dual variables to be updated
	## Q, A, G, q, b, h of appropriate size define the optimization problem 
	
	n = Q.shape[0]
	p = G.shape[0]
	m = A.shape[0] 
	
	## Affine scaling direction and step size
	M = jnp.vstack((
		jnp.hstack((Q, np.zeros((n,p)), G.T,A.T)),
		jnp.hstack((np.zeros((p,n)),jnp.diagflat(z),jnp.diagflat(s), np.zeros((p,m)))),
		jnp.hstack((G,np.eye(p),np.zeros((p,m+p)))),
		jnp.hstack((A,np.zeros((m,m+p+p))))
	))
	rhs1 = jnp.vstack((
		-(A.T.dot(y) + G.T.dot(z) + Q.dot(x) + q),
		-s*z,
		-(G.dot(x) + s - h),
		-(A.dot(x) - b)
	))

	deltaAff = jnp.linalg.solve(M,rhs1)
	dxaff = deltaAff[:n]
	dsaff = deltaAff[n:(n+p)]
	dzaff = deltaAff[(n+p):(n+p+p)]
	dyaff = deltaAff[(n+p+p):]

	if jnp.sum(dsaff < 0) > 0:
		alphas = jnp.min((-s / dsaff)[dsaff < 0])
	else:
		alphas = 1

	if jnp.sum(dzaff < 0) > 0:
		alphaz = jnp.min((-z / dzaff)[dzaff < 0])
	else:
		alphaz = 1
	alpha = jnp.min(jnp.array((alphas, alphaz,1)))
	
	## Centering plus corrector direction
	sz = jnp.sum(s * z)
	sigma = (  jnp.sum( (s + alpha * dsaff) * (z + alpha * dzaff) ) / (sz) )**3
	mu = sz / p
	rhs2 = jnp.vstack((
		q * 0,
		-dsaff*dzaff + sigma * mu,
		h * 0,
		b * 0
	))
	

	deltaCC = jnp.linalg.solve(M,rhs2)
	dxcc = deltaCC[:n]
	dscc = deltaCC[n:(n+p)]
	dzcc = deltaCC[(n+p):(n+p+p)]
	dycc = deltaCC[(n+p+p):]

	ds = dsaff + dscc
	dz = dzaff + dzcc
	
	## Final step size
	if jnp.sum(ds < 0) > 0:
		alphas = jnp.min((-s / ds)[ds < 0])
	else:
		alphas = 2

	if jnp.sum(dz < 0) > 0:
		alphaz = jnp.min((-z / dz)[dz < 0])
	else:
		alphaz = 2

	alpha = jnp.min(jnp.array((0.99999 * alphas, 0.99999 * alphaz,1)))

	## Return updated variables
	return x + alpha * (dxaff + dxcc), s + alpha * (dsaff + dscc), z + alpha * (dzaff + dzcc), y + alpha * (dyaff + dycc), rhs2, alpha
	
def IP(Q, A, G, q, b, h, verb = False, stopGrad = True, returnx = False):
	## Implement the primal dual interior point solver for QP
	## Q, A, G, q, b, h of appropriate size define the optimization problem 
	## verb: verbosity, print diagnostic information on running iteration ("suboptimality" and chosen step-size)
	## stopGrad: if true only propagate gradient through the last iteration
	## returnx: if ture only returns primal variable, otherwise all variables plus diagnostic information
	
	n = Q.shape[0]
	p = G.shape[0]
	m = A.shape[0] 
	
	## Initialize 
	M = np.vstack((
		np.hstack((Q,G.T,A.T)),
		np.hstack((G,-np.eye(p),np.zeros((p,m)))),
		np.hstack((A,np.zeros((m,p)),np.zeros((m,m))))
	))

	rhs = jnp.vstack((-q,h,b))
	if stopGrad:
		temp = jnp.linalg.solve(jax.lax.stop_gradient(M), jax.lax.stop_gradient(rhs))
	else:
		temp = jnp.linalg.solve(M, rhs)
	x = temp[:n]
	z = temp[n:(n+p)]
	y = temp[(n+p):]

	x0 = x.copy()
	y0 = y.copy()
	z = jnp.dot(G,x) - h
	alphap = max(z)

	s0 = - z.copy()
	z0 = z.copy()
	if alphap >= 0:
		s0 += 1+alphap
		z0 += 1+alphap
	
	x = x0.copy()
	s = s0.copy()
	z = z0.copy()
	y = y0.copy()
	
	## Iterations
	for k in range(50):
		xp = x.copy()
		sp = s.copy()
		zp = z.copy()
		yp = y.copy()
		
		if stopGrad: ## Update with stopGrad
			x, s, z, y, rhs2, alpha = IPupdate(
					jax.lax.stop_gradient(xp),
					jax.lax.stop_gradient(sp),
					jax.lax.stop_gradient(zp),
					jax.lax.stop_gradient(yp), 
					jax.lax.stop_gradient(Q), 
					jax.lax.stop_gradient(A), 
					jax.lax.stop_gradient(G), 
					jax.lax.stop_gradient(q), 
					jax.lax.stop_gradient(b), 
					jax.lax.stop_gradient(h)
				)
		else: ## Update without stopGrad
			x, s, z, y, rhs2, alpha = IPupdate(xp,sp,zp,yp, Q, A, G, q, b, h)
		subopt = jnp.linalg.norm(rhs2)
		if verb: ## print information on iterations
			print("It: ", end='')
			print(k, end='')
			print(", subopt: ", end='')
			print(np.format_float_scientific(np.linalg.norm(rhs2), precision=2), end='')
			print(", step: ", end='')
			print(np.format_float_scientific(alpha, precision = 2))
		if subopt < 1e-10:  ## stoping criterion							
			break

	## One last upadte
	x, s, z, y, rhs2, alpha = IPupdate(x,s,z,y, Q, A, G, q, b, h)
	if returnx: ## Return only primal variable or all variables and additional info
		return x
	else:
		return x, s, z, y, rhs2, alpha

## Test
#x, s, z, y, _, _ = IP(Q, A, G, q, b, h, verb = True, stopGrad= True)



def diffImpNorm(Q, A, G, q, b, h, solutionVariable = None):
	## Bacward mode, derivative of \|x(b)\|^2 / 2 where x(b) is solution to the 
	## QP problem
	n = Q.shape[0]
	p = G.shape[0]
	m = A.shape[0] 
		
	if solutionVariable is None:
		x, s, z, y, _, _ = IP(Q, A, G, q, b, h)
	else:
		x = solutionVariable[0].copy()
		s = solutionVariable[1].copy()
		z = solutionVariable[2].copy()
		y = solutionVariable[3].copy()

	vrhs = jnp.vstack((
			x,
			jnp.zeros((p, 1)),
			jnp.zeros((p, 1)),
			jnp.zeros((m,1))
		))

	M = jnp.vstack((
			jnp.hstack((Q, np.zeros((n,p)), G.T,A.T)),
			jnp.hstack((np.zeros((p,n)),np.diagflat(z),np.diagflat(s), np.zeros((p,m)))),
			jnp.hstack((G,np.eye(p),np.zeros((p,m+p)))),
			jnp.hstack((A,np.zeros((m,m+p+p))))
		))

	return jnp.linalg.solve(M, vrhs)[(n+p+p):]

def diffImpB(Q, A, G, q, b, h, solutionVariable = None):
	## Implement the primal dual interior point solver for QP with implicit diff 
	## for evaluating jacobian of x with respect to b.
	##
	## Q, A, G, q, b, h of appropriate size define the optimization problem 
	## solutionVariable, primal dual solution in appropriate format. 
	## if provided, the solution is not recomputed.
	
	n = Q.shape[0]
	p = G.shape[0]
	m = A.shape[0] 
	
	if solutionVariable is None:
		x, s, z, y, _, _ = IP(Q, A, G, q, b, h)
	else:
		x = solutionVariable[0].copy()
		s = solutionVariable[1].copy()
		z = solutionVariable[2].copy()
		y = solutionVariable[3].copy()
	
	## derivative with respect to b right hand side
	db = jnp.eye(m)
	
	## Jacobian of equation right hand side.
	drhs = jnp.vstack((
			jnp.zeros((n, m)),
			jnp.zeros((p, m)),
			jnp.zeros((p, m)),
			db
		))
	
	## Jacobian of left hand side
	M = jnp.vstack((
			jnp.hstack((Q, np.zeros((n,p)), G.T,A.T)),
			jnp.hstack((np.zeros((p,n)),np.diagflat(z),np.diagflat(s), np.zeros((p,m)))),
			jnp.hstack((G,np.eye(p),np.zeros((p,m+p)))),
			jnp.hstack((A,np.zeros((m,m+p+p))))
		))
	
	## Return implicit diff 
	return jnp.linalg.inv(M).dot(drhs)[:n,:]

def diffOneStep(x, s, z, y, Q, A, G, q, b, h):
	return jax.jacfwd(IPupdate, 8)(x, s, z, y, Q, A, G, q, b, h)[0].reshape((n,m))

def diffIP(Q, A, G, q, b, h,  stopGrad = True, returnx = False):
	return jax.jacfwd(IP, 4)(Q, A, G, q, b, h,  stopGrad = stopGrad, returnx = returnx)[0]



def IPnorm(Q, A, G, q, b, h,  stopGrad = True):
	## Evaluate norm of the solution
	return jnp.sum(IP(Q, A, G, q, b, h,  stopGrad = stopGrad, returnx = True)**2)/2

def IPnormOneStep(x, s, z, y, Q, A, G, q, b, h,  stopGrad = True):
	## One step variant
	x, s, z, y, rhs2, alpha =IPupdate(x, s, z, y, Q, A, G, q, b, h)
	return jnp.sum(x**2)/2

def diffIPNorm(Q, A, G, q, b, h,  stopGrad = True):
	## Derivative of Ipnorm
	return jax.jacrev(IPnorm, 4)(Q, A, G, q, b, h,  stopGrad = stopGrad)

def diffNormOneStep(x, s, z, y, Q, A, G, q, b, h):
	## Derivative of one step norm
	return jax.jacrev(IPnormOneStep, 8)(x, s, z, y, Q, A, G, q, b, h)

import numpy as np

from icecream import ic


def to_hyperboloid_points(poincare_pts):
    """
    Post: result.shape[1] == poincare_pts.shape[1] + 1
    """
    norm_sqd = (poincare_pts ** 2).sum(axis=1)
    N = poincare_pts.shape[1]
    result = np.zeros((poincare_pts.shape[0], N + 1), dtype=np.float64)
    result[:,:N] = (2. / (1 - norm_sqd))[:,np.newaxis] * poincare_pts
    result[:,N] = (1 + norm_sqd) / (1 - norm_sqd)
    return result


def to_hyperboloid_point(poincare_pt):
    """
    Post: len(result) == len(poincare_pt) + 1
    """
    return to_hyperboloid_points(poincare_pt[np.newaxis,:])[0,:]


def to_poincare_ball_point(hyperboloid_pt):
    """
    Project the point of the hyperboloid onto the Poincaré ball.
    Post: len(result) == len(hyperboloid_pt) - 1
    
    The last coordinate is the special coordinate
    """
    N = len(hyperboloid_pt) - 1
    return hyperboloid_pt[:N] / (hyperboloid_pt[N] + 1)


def minkowski_dot(u, v):
    """
    `u` and `v` are vectors in Minkowski space.
    """
    rank = u.shape[-1] - 1
    euc_dp = u[:rank].dot(v[:rank])
    return euc_dp - u[rank] * v[rank]


def project_onto_tangent_space(hyperboloid_point, minkowski_tangent):
    return minkowski_tangent + minkowski_dot(hyperboloid_point, minkowski_tangent) * hyperboloid_point


def frechet_gradient(theta, points, weights=None):
    """
    Return the gradient of the weighted Frechet mean of the provided points at the hyperboloid point theta.
    This is tangent to theta in on the hyperboloid.
    Arguments are numpy arrays.  `points` is 2d, the others are 1d. They satisfy:
    len(weights) == len(points) and points.shape[1] == theta.shape[0].
    If weights is None, use uniform weighting.
    """
    if weights is None:
        weights = np.ones_like(points[:,0]) / points.shape[0]
    weights /= weights.sum()
        
    # compute minkowski dot products
    last = theta.shape[0] - 1
    mdps = points[:,:last].dot(theta[:last]) - points[:,last] * theta[last]
    
    # scale it
    max_mdp = -(1 + 1e-10)
    mdps[mdps > max_mdp] = max_mdp
    
    dists = np.arccosh(-mdps)
    scales = -dists * weights / np.sqrt(mdps ** 2 - 1)
    minkowski_tangent = (points * scales[:,np.newaxis]).sum(axis=0)
    return project_onto_tangent_space(theta, minkowski_tangent)


def exponential(base, tangent):
    """
    Compute the exponential of `tangent` from the point `base`.
    """
    tangent = tangent.copy()
    norm = np.sqrt(max(minkowski_dot(tangent, tangent), 0))
    if norm == 0:
        return base
    tangent /= norm
    return np.cosh(norm) * base + np.sinh(norm) * tangent


class FrechetMeanMan:
    """Compute the Frechet Mean manually with the hyperboloid model of geometry.
    Assume data is inputted as lying in the hyperboloid model of geometry."""
    
    def __init__(self, epsilon=1e-12, max_iter=32, init_point=None, lr=0.001):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.init_point = init_point
        self.lr = lr
        self.estimate_ = None
        self.mean_history = None
        self.grad_norm_history = None
        self.converged = True
            
    def frechet_mean(self, X, MIN_GRAD_NORM=1e-12, max_iter=32, return_history=False, init_point=None, lr=0.001, weights=None):
        """
        Given a 2d numpy array of X on the hyperboloid, keep making updates
        until the gradient norm falls below MIN_GRAD_NORM; re-inits if necessary.
        
        This function requires X be in the hyperboloid model of geometry and outputs the frechet
        mean also in the hyperboloid model of geometry.
        """

        # initial values
        theta = X[0] if init_point is None else init_point
        theta_history = [theta]
        grad_norm_history = []
        steps = 1
        
        while True:
            
            # compute hyperboloid gradient
            hyperboloid_gradient = frechet_gradient(theta, X, weights=weights)
            
            # break if gradient is below threshold
            dot = minkowski_dot(hyperboloid_gradient, hyperboloid_gradient)
            gradient_norm = np.sqrt(dot)
            grad_norm_history.append(gradient_norm)
            if gradient_norm < MIN_GRAD_NORM:
                break
            
            # update weights for next iteration 
            theta = exponential(theta, -1 * lr * hyperboloid_gradient)
            theta_history.append(theta)
            steps += 1
            
            # break if took too many iterations passed
            if steps >= max_iter:
                # print('WARNING: Maximum number of iterations {} reached. The mean may be inaccurate.'.format(steps))
                self.converged = False
                break
                
        if return_history:
            return theta, theta_history, grad_norm_history
        return theta

        
    def fit(self, X, weights=None):
            
        # compute the frechet mean 
        theta, theta_history, grad_norm_history = self.frechet_mean(
            X, 
            MIN_GRAD_NORM=self.epsilon,
            max_iter=self.max_iter,
            return_history=True,
            init_point=self.init_point,
            lr=self.lr,
            weights=weights
            )

        # record values
        self.estimate_ = theta
        self.mean_history = theta_history
        self.grad_norm_history = grad_norm_history
        
        return self
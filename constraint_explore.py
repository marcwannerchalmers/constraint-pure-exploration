import numpy as np
from scipy.optimize import linprog, minimize, LinearConstraint, OptimizeResult
import itertools
from copy import deepcopy

PRECISION = 1e-12


######## UTILS ########
def project_on_feasible(allocation, A, b):
    """
    Project allocation on feasible set
    :param allocation: allocation to project
    :param A: matrix of constraints
    :param b: vector of constraints
    """
    simplex = np.ones_like(allocation).reshape(1, -1)
    eye = np.eye(len(allocation))
    if A is not None:
        A = np.concatenate([A, eye, -eye, simplex, -simplex], axis=0)
        b = np.concatenate(
            [
                b,
                np.ones(len(allocation)),
                np.zeros(len(allocation)),
                np.array([1]),
                np.array([-1]),
            ],
            axis=0,
        )
    else:
        A = np.concatenate([eye, -eye, simplex, -simplex], axis=0)
        b = np.concatenate(
            [
                np.ones(len(allocation)),
                np.zeros(len(allocation)),
                np.array([1]),
                np.array([-1]),
            ],
            axis=0,
        )
    constraints = LinearConstraint(A=A, ub=b)
    x0 = np.ones_like(allocation) / len(allocation)
    fun = lambda x, y: np.linalg.norm(x - y) ** 2
    results = minimize(fun=fun, x0=x0, args=(allocation), constraints=constraints)
    if not results["success"]:
        raise "LP Solver failed"
    x = results["x"]
    if np.abs(np.sum(x) - 1) > 1e-5:
        raise "Allocation doesnt sum to 1"
    return x


def get_policy(mu, A, b):
    """
    Find optimal policy
    :param mu: Reward vector
    :param A: if None solve standard bandit problem without any constraints on policy
    :param b: if None solve standard bandit problem without any constraints on policy
    :return:
        - optimal policy
        - aux info from optimizer
    """
    simplex = np.ones_like(mu).reshape(1, -1)
    eye = np.eye(len(mu))
    one = np.array([1])
    if A is not None:
        A = np.concatenate([A, -eye, simplex, -simplex], axis=0)
        b = np.concatenate([b, np.zeros(len(mu)), one, -one], axis=0)
    else:
        A = np.concatenate([-eye, simplex, -simplex], axis=0)
        b = np.concatenate([np.zeros(len(mu)), one, -one], axis=0)

    results = linprog(
        -mu, A_ub=A, b_ub=b, A_eq=None, b_eq=None, method="highs-ds"
    )  # Use simplex method
    if not results["success"]:
        raise "LP Solver failed"
    # Get active constraints
    aux = {"A": A, "b": b, "slack": results["slack"]}
    return results["x"], aux


def arreqclose_in_list(myarr, list_arrays):
    """
    Test if np array is in list of np arrays
    """
    return next(
        (
            True
            for elem in list_arrays
            if elem.size == myarr.size and np.allclose(elem, myarr)
        ),
        False,
    )


def enumerate_all_policies(A, b):
    """
    Enumerate all policies in the polytope Ax <= b
    """
    # Compute all possible bases
    n_constraints = A.shape[0]
    n_arms = A.shape[1]
    bases = list(itertools.combinations(range(n_constraints), n_arms))
    policies = []
    for base in bases:
        base = np.array(base)
        B = A[base]
        # Check that the base is not degenerate
        if np.linalg.matrix_rank(B) == A.shape[1]:
            policy = np.linalg.solve(B, b[base])
            # Verify that policy is in the polytope
            if np.all(A.dot(policy) <= b + 1e-5) and not arreqclose_in_list(
                policy, policies
            ):
                policies.append(policy)
    return policies


def compute_neighbors(vertex, A, b, slack):
    """
    Compute all neighbors of vertex in the polytope Ax <= b
    :param vertex: vertex of the polytope
    :param A: matrix of constraints
    :param b: vector of constraints
    :param slack: vector of slack variables
    """
    active = slack == 0
    not_active = slack != 0
    n_constraints = np.arange(A.shape[0])
    active_constaints = n_constraints[active].tolist()
    inactive_constraints = n_constraints[not_active].tolist()
    neighbors = []

    # Compute all possible bases at the vertex
    bases = list(itertools.combinations(active_constaints, len(vertex)))
    # For each possible base swap one element with an inactive constraint to get a neighbor
    for base in bases:
        for constraint in inactive_constraints:
            # Swap constraint into each position of the base
            for i in range(len(base)):
                new_base = np.array(deepcopy(base))
                new_base[i] = constraint
                B = A[new_base]
                # Check that the base is not degenerate
                if np.linalg.matrix_rank(B) == len(vertex):
                    possible_neighbor = np.linalg.solve(B, b[new_base])
                    # Verify that neighbor is in the polytope
                    if np.all(
                        A.dot(possible_neighbor) <= b + 1e-5
                    ) and not arreqclose_in_list(possible_neighbor, neighbors):
                        neighbors.append(possible_neighbor)
    return neighbors


def binary_search(mu, interval, threshold, kl):
    """
    Find maximizer of KL(mu, x) in interval satysfiyng threshold using binary search
    :param mu: reward of arm
    :param interval: interval to search in
    :param threshold: threshold to satisfy (f(t) = log t)
    :param kl: KL divergence function

    """
    p = 0
    q = len(interval)
    done = False
    while not done:
        i = int((p + q) / 2)
        x = interval[i]
        loss = kl(mu, x)
        if loss < threshold:
            p = i
        else:
            q = i
        if p + 1 >= q:
            done = True

    return x, loss


def get_confidence_interval(mu, pulls, f_t, upper=6, lower=-1, kl=None):
    """
    Compute confidence interval for each arm
    :param mu: reward vector
    :param pulls: number of pulls for each arm
    :param f_t: threshold function f(t) = log t
    :param upper: upper bound for search
    :param lower: lower bound for search
    :param kl: KL divergence function
    :return:
        - lower bound for each arm
        - upper bound for each arm
    """
    if kl is None:
        kl = lambda m1, m2: ((m1 - m2) ** 2) / (2)
    ub = [
        binary_search(m, np.linspace(m, upper, 5000), threshold=f_t / n, kl=kl)[0]
        for m, n in zip(mu, pulls)
    ]
    lb = [
        binary_search(m, np.linspace(m, lower, 5000), threshold=f_t / n, kl=kl)[0]
        for m, n in zip(mu, pulls)
    ]

    return lb, ub

def binary_search_lipschitz(mu, pulls, interval, threshold, kl, lip_fun, arm, n_arms, mode=None):
    """
    Find maximizer of KL(mu, x) in interval satysfiyng threshold using binary search
    :param mu: reward of arm
    :param interval: interval to search in
    :param threshold: threshold to satisfy (f(t) = log t)
    :param kl: KL divergence function

    """
    p = 0
    q = len(interval)
    done = False
    while not done:
        i = int((p + q) / 2)
        x = interval[i]
        if mode=="upper":
            loss = np.sum([pulls*kl(mu, x - lip_fun[arm, k]) for k in range(n_arms)])
        else:
            loss = np.sum([pulls*kl(mu, x + lip_fun[arm, k]) for k in range(n_arms)])
        if loss < threshold:
            p = i
        else:
            q = i
        if p + 1 >= q:
            # make sure that x is projected to feasible
            x = interval[i]
            done = True

    # account for possible non-monotonicity and return the worst case interval if no feasible loss found
    if mode=="upper":
        loss = np.sum([pulls*kl(mu, x - lip_fun[arm, k]) for k in range(n_arms)])
    else:
        loss = np.sum([pulls*kl(mu, x + lip_fun[arm, k]) for k in range(n_arms)])
    if loss > threshold:
        x = interval[-1]

    return x, loss

def kl_bernoulli(mu, lam):
    return mu * np.log(mu / lam) + (1 - mu) * np.log((1 - mu) / (1 - lam))

def get_confidence_interval_lipschitz(mu, pulls, f_t, lip_fun, upper=6, lower=-1, kl=None):
    """
    Compute confidence interval for each arm
    :param mu: reward vector
    :param pulls: number of pulls for each arm
    :param f_t: threshold function f(t) = log t
    :param upper: upper bound for search
    :param lower: lower bound for search
    :param kl: KL divergence function
    :return:
        - lower bound for each arm
        - upper bound for each arm
    """
    if kl is None:
        kl = kl_bernoulli
    ub = [
        binary_search_lipschitz(mn[0], mn[1], np.linspace(mn[0], upper, 5000), threshold=f_t, kl=kl, lip_fun=lip_fun, arm=arm, n_arms=len(mu), mode="upper")[0]
        for arm, mn in enumerate(zip(mu, pulls))
    ]
    lb = [
        binary_search_lipschitz(mn[0], mn[1], np.linspace(mn[0], lower, 5000), threshold=f_t, kl=kl, lip_fun=lip_fun, arm=arm, n_arms=len(mu), mode="lower")[0]
        for arm, mn in enumerate(zip(mu, pulls))
    ]

    return lb, ub


def gaussian_projection(w, mu, pi1, pi2, sigma=1):
    """
    perform close-form projection onto the hyperplane lambda^T(pi1 - pi2) = 0 assuming Gaussian distribution
    :param w: weight vector
    :param mu: reward vector
    :param pi1: optimal policy
    :param pi2: suboptimal neighbor policy
    :param sigma: standard deviation of Gaussian distribution

    return:
        - lambda: projection
        - value of the projection

    """
    v = pi1 - pi2
    normalizer = ((v**2) / (w + PRECISION)).sum()
    lagrange = mu.dot(v) / normalizer
    lam = mu - lagrange * v / (w + PRECISION)
    var = sigma**2
    value = (w * ((mu - lam) ** 2)).sum() / (2 * var)
    return lam, value


def bernoulli_projection(w, mu, pi1, pi2, sigma=1):
    """
    Projection onto the hyperplane lambda^T(pi1 - pi2) = 0 assuming Bernoulli distribution using scipy minimize
    """
    mu = np.clip(mu, 1e-3, 1 - 1e-3)
    bounds = [(1e-3, 1 - 1e-3) for _ in range(len(mu))]
    v = pi1 - pi2
    constraint = LinearConstraint(v.reshape(1, -1), 0, 0)

    def objective(lam):
        kl_bernoulli = mu * np.log(mu / lam) + (1 - mu) * np.log((1 - mu) / (1 - lam))
        return (w * kl_bernoulli).sum()

    x0 = gaussian_projection(w, mu, pi1, pi2, sigma)[0]
    x0 = np.clip(x0, 1e-3, 1 - 1e-3)
    res = minimize(objective, x0, constraints=constraint, bounds=bounds)
    lam = res.x
    value = objective(lam)
    return lam, value


def best_response(w, mu, pi, neighbors, sigma=1, dist_type="Gaussian"):
    """
    Compute best response instance w.r.t. w by projecting onto neighbors
    :param w: weight vector
    :param mu: reward vector
    :param pi: optimal policy
    :param neighbors: list of neighbors
    :param sigma: standard deviation of Gaussian distribution
    :param dist_type: distribution type to use for projection

    return:
        - value of best response
        - best response instance
    """
    if dist_type == "Gaussian":
        projections = [
            gaussian_projection(w, mu, pi, neighbor, sigma) for neighbor in neighbors
        ]
    elif dist_type == "Bernoulli":
        projections = [
            bernoulli_projection(w, mu, pi, neighbor, sigma) for neighbor in neighbors
        ]
    else:
        raise NotImplementedError
    values = [p[1] for p in projections]
    instances = [p[0] for p in projections]
    return np.min(values), instances[np.argmin(values)]


def solve_game(
    mu,
    vertex,
    neighbors,
    sigma=1,
    dist_type="Gaussian",
    allocation_A=None,
    allocation_b=None,
    tol=None,
    x0=None,
):
    """
    Solve the game instance w.r.t. reward vector mu. Used for track-n-stop algorithms
    :param mu: reward vector
    :param vertex: vertex of the game
    :param neighbors: list of neighbors
    :param sigma: standard deviation of Gaussian distribution
    :param dist_type: distribution type to use for projection
    :param allocation_A: allocation constraint. If None allocations lies in simplex.
    :param tol: Default None for speed in TnS.
    :param x0: initial point
    """

    def game_objective(w):
        return -best_response(w, mu, vertex, neighbors, sigma, dist_type)[0]

    tol_sweep = [1e-16, 1e-12, 1e-6, 1e-4]  # Avoid tolerance issues in scipy
    if tol is not None and tol > tol_sweep[0]:
        tol_sweep = [tol] + tol_sweep
    else:
        tol_sweep = [None] + tol_sweep  # Auto tune tol via None
    if allocation_A is None:
        # Solve optimization problem over simplex
        simplex = np.ones_like(mu).reshape(1, -1)
        constraint = LinearConstraint(A=simplex, lb=1, ub=1)
        bounds = [(0, 1) for _ in range(len(mu))]
        count = 0
        done = False
        if x0 is None:
            while count < len(tol_sweep) and not done:
                x0 = np.random.uniform(0.3, 0.6, size=len(mu))
                x0 = x0 / x0.sum()
                tol = tol_sweep[count]
                res = minimize(
                    game_objective, x0, constraints=constraint, bounds=bounds, tol=tol
                )
                done = res["success"]
                count += 1
        else:
            res = minimize(
                game_objective, x0, constraints=constraint, bounds=bounds, tol=tol
            )
    else:
        # Solve optimization problem over allocation constraint
        constraint = LinearConstraint(A=allocation_A, ub=allocation_b)
        bounds = [(0, 1) for _ in range(len(mu))]
        count = 0
        done = False
        if x0 is None:
            while count < len(tol_sweep) and not done:
                tol = tol_sweep[count]
                x0 = np.random.uniform(0.3, 0.6, size=len(mu))
                x0 = project_on_feasible(x0, allocation_A, allocation_b)
                res = minimize(
                    game_objective, x0, constraints=constraint, bounds=bounds, tol=tol
                )
                done = res["success"]
                count += 1
        else:
            res = minimize(
                game_objective, x0, constraints=constraint, bounds=bounds, tol=tol
            )
    if res["success"] == False:
        raise ValueError("Optimization failed")
    return res.x, -res.fun


####### ALGORITHMS #######


class Explorer:
    """
    Abstract class for an explorer
    """

    def __init__(
        self,
        n_arms,
        A,
        b,
        delta,
        ini_phase=1,
        sigma=1,
        restricted_exploration=False,
        dist_type="Gaussian",
        seed=None,
        d_tracking=False,
    ):
        """
        Initialize the explorer
        :param n_arms: number of arms
        :param A: matrix constraints
        :param b: vector constraints
        :param delta: confidence parameter
        :param ini_phase: initial phase (how many times to play each arm before adaptive search starts). Default: 1
        :param sigma: standard deviation of Gaussian distribution
        :param restricted_exploration: whether to use restricted exploration or not
        :param dist_type: distribution type to use for projection
        :param seed: random seed
        """
        self.n_arms = n_arms
        self.A = A
        self.b = b
        self.delta = delta
        self.ini_phase = ini_phase
        self.sigma = sigma
        self.restricted_exploration = restricted_exploration
        self.dist_type = dist_type
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.d_tracking = d_tracking
        self.cumulative_weights = np.zeros(n_arms)
        self.D = 1
        self.alpha = 1

        self.t = 0
        self.neighbors = {}
        self.means = np.zeros(n_arms)
        self.n_pulls = np.zeros(n_arms)

        if dist_type == "Gaussian":
            # Set KL divergence and lower/upper bounds for binary search
            self.kl = lambda x, y: 1 / (2 * (sigma**2)) * ((x - y) ** 2)
            self.lower = -1
            self.upper = 10
        elif dist_type == "Bernoulli":
            # Set KL divergence for Bernoulli distribution and lower/upper bounds for binary search
            self.kl = lambda x, y: x * np.log(x / y) + (1 - x) * np.log(
                (1 - x) / (1 - y)
            )
            self.lower = 0 + 1e-4
            self.upper = 1 - 1e-4
            self.ini_phase = (
                10  # Take longer initial phase for Bernoulli to avoid all 0  or all 1
            )
        else:
            raise NotImplementedError

        if restricted_exploration:
            # Compute allocation constraint
            test = np.ones_like(self.means)
            _, aux = get_policy(test, A, b)
            self.allocation_A = aux["A"]
            self.allocation_b = aux["b"]
        else:
            self.allocation_A = None
            self.allocation_b = None

    def tracking(self, allocation):
        """
        Output arm based on either d-tracking or cumulative tracking
        """
        if self.d_tracking:
            return np.argmin(self.n_pulls - self.t * allocation)
        else:
            eps = 1 / (2 * np.sqrt(self.t + self.n_arms**2))
            eps_allocation = allocation + eps
            eps_allocation = eps_allocation / eps_allocation.sum()
            self.cumulative_weights += eps_allocation
            return np.argmin(self.n_pulls - self.cumulative_weights)

    def act():
        """
        Choose an arm to play
        """
        raise NotImplementedError

    def stopping_criterion(self, vertex):
        """
        Check stopping criterion. Stopping based on the generalized log-likelihood ratio test
        """

        hash_tuple = tuple(vertex.tolist())
        game_value, _ = lipschitz_best_response(
            w=self.empirical_allocation(),
            mu=self.means,
            pi=vertex,
            sigma=self.sigma,
            dist_type=self.dist_type,
            A=self.A,
            b=self.b
        )

        beta = np.log((1 + np.log(self.t)) / self.delta)

        return self.t * game_value > beta

    def empirical_allocation(self):
        """
        Compute empirical allocation
        """
        return self.n_pulls / self.t

    def update(self, arm, reward):
        """
        Update the explorer with the reward obtained from playing the arm
        :param arm: arm played
        :param reward: reward obtained
        """
        self.t += 1
        self.n_pulls[arm] += 1
        self.means[arm] = self.means[arm] + (1 / self.n_pulls[arm]) * (
            reward - self.means[arm]
        )

        if self.dist_type == "Bernoulli":
            self.means = np.clip(self.means, self.lower, self.upper)


class TnS(Explorer):
    """
    Track-n-stop style of algorithm for bandits with linear constraints
    """

    def __init__(
        self,
        n_arms,
        A,
        b,
        delta,
        ini_phase=1,
        sigma=1,
        restricted_exploration=False,
        dist_type="Gaussian",
        seed=None,
        d_tracking=True,
    ):
        """
        Initialize the explorer
        :param n_arms: number of arms
        :param A: matrix constraints
        :param b: vector constraints
        :param delta: confidence parameter
        :param ini_phase: initial phase (how many times to play each arm before adaptive search starts)
        :param sigma: standard deviation of Gaussian distribution
        :param restricted_exploration: whether to use restricted exploration or not
        :param dist_type: distribution type to use for projection
        :param seed: random seed
        :param d_tracking: D-tracking or C-tracking
        """
        super().__init__(
            n_arms,
            A,
            b,
            delta,
            ini_phase,
            sigma,
            restricted_exploration,
            dist_type,
            seed,
            d_tracking,
        )

    def act(self):
        """
        Choose an arm to play
        """
        if self.t < self.n_arms * self.ini_phase:
            # Initial phase play each arm once
            arm = self.t % self.n_arms
            return arm, False, None, None

        # Compute optimal policy w.r.t. current empirical means
        optimal_policy, aux = get_policy(mu=self.means, A=self.A, b=self.b)

        # Check if policy already visited. If yes retrieve neighbors otherwise compute neighbors
        hash_tuple = tuple(optimal_policy.tolist())  # npy not hashable
        if hash_tuple in self.neighbors:
            neighbors = self.neighbors[hash_tuple]
        else:
            neighbors = compute_neighbors(
                optimal_policy, aux["A"], aux["b"], slack=aux["slack"]
            )
            self.neighbors[hash_tuple] = neighbors

        # Solve game to get allocation
        allocation, game_value = solve_game(
            mu=self.means,
            vertex=optimal_policy,
            neighbors=neighbors,
            dist_type=self.dist_type,
            sigma=self.sigma,
            allocation_A=self.allocation_A,
            allocation_b=self.allocation_b,
        )

        # Check if forced exploration is needed else D-tracking
        not_saturated = self.n_pulls < (np.sqrt(self.t) - self.n_arms / 2)
        if not_saturated.any() and self.d_tracking:
            # Play smallest N below sqrt(t) - n_arms/2
            arm = np.argmin(self.n_pulls)

        else:
            # Play arm according to tracking rule
            arm = self.tracking(allocation)

        # Check stopping criterion
        stop = self.stopping_criterion(optimal_policy)

        misc = {
            "game_value": game_value,
            "allocation": allocation,
            "optimal_policy": optimal_policy,
        }

        return arm, stop, optimal_policy, misc
    
def kl_bernoulli(mu, lam):
    return mu * np.log(mu / lam) + (1 - mu) * np.log((1 - mu) / (1 - lam))
    
def lipschitz_best_response(w,
            mu,
            pi,
            sigma,
            dist_type,
            A, 
            b):
    A_pi = np.zeros((len(pi)-1, len(pi)))
    b_pi = np.zeros((len(pi)-1,))
    best_arm = np.argmax(pi)
    for k in range(len(b_pi)):
        if k == best_arm:
            continue
        A_pi[k, best_arm] = 1
        A_pi[k, k] = -1
        k += 1

    constraints = [LinearConstraint(A, ub=b), LinearConstraint(A_pi, lb=b_pi)]

    def objective(lam):
        return np.dot(w, kl_bernoulli(mu, lam))

    res = minimize(objective, x0=0.5*np.ones_like(mu), constraints=constraints, method='SLSQP')

    return res.fun, res.x
    


class CGE(Explorer):
    """
    Constrainted Game Explorer for bandits with linear constraints.

    Performs exploration by treating the lower bound as a zero-sum game.

    Allocation player is AdaHedge
    Instance player performs a best response w.r.t. the allocation

    """

    def __init__(
        self,
        n_arms,
        A,
        b,
        delta,
        ini_phase=1,
        sigma=1,
        restricted_exploration=False,
        dist_type="Gaussian",
        seed=None,
        d_tracking=True,
        use_adahedge=True,
        lambda_mat=None
    ):
        super().__init__(
            n_arms,
            A,
            b,
            delta,
            ini_phase,
            sigma,
            restricted_exploration,
            dist_type,
            seed,
            d_tracking,
        )
        # Initialize the allocation player
        if use_adahedge:
            simplex = np.ones_like(self.means).reshape(1, -1)
            eye = np.eye(len(self.means))
            one = np.array([1])
            if restricted_exploration:
                allocation_A = np.concatenate([A, -eye, simplex, -simplex], axis=0)
                allocation_b = np.concatenate([b, np.zeros(n_arms), one, -one], axis=0)
                self.ada = AdaGrad(A=allocation_A, b=allocation_b, loss_rescale=1)
            else:
                allocation_A = np.concatenate([-eye, simplex, -simplex], axis=0)
                allocation_b = np.concatenate([np.zeros(n_arms), one, -one], axis=0)
                self.ada = AdaGrad(
                    A=allocation_A, b=allocation_b, loss_rescale=1
                )  # AdaHedge(d=n_arms, loss_rescale=0.01)
        else:
            self.ada = OnlineGradientDescent(d=n_arms)

        self.lambda_mat = lambda_mat

    def act(self):
        """
        Choose an arm to play
        """
        if self.t < self.n_arms * self.ini_phase:
            # Initial phase play each arm once
            arm = self.t % self.n_arms
            return arm, False, None, None

        # Compute optimal policy w.r.t. current empirical means
        optimal_policy = np.zeros_like(self.means)
        optimal_policy[np.argmax(self.means)] = 1
        
        # Get allocation from AdaHedge
        allocation = self.ada.get_weights()

        # Perform best response
        #TODO: adapt correctly
        br_value, br_instance = lipschitz_best_response(
            w=allocation,
            mu=self.means,
            pi=optimal_policy,
            sigma=self.sigma,
            dist_type=self.dist_type,
            A=self.A,
            b=self.b
        )

        # Compute loss for allocation player
        ft = np.log(self.t)
        # Optimism
        """lb, ub = get_confidence_interval_lipschitz(
            self.means, self.n_pulls, ft, kl=self.kl, lower=self.lower, upper=self.upper, lip_fun=self.lambda_mat
        )"""
        lb, ub = get_confidence_interval(
            self.means, self.n_pulls, ft, kl=self.kl, lower=self.lower, upper=self.upper)
        loss = [
            np.max(
                [
                    ft / self.n_pulls[a],
                    self.kl(lb[a], br_instance[a]),
                    self.kl(ub[a], br_instance[a]),
                ]
            )
            for a in range(self.n_arms)
        ]
        # Update allocation player
        self.ada.update(-np.array(loss))

        # Perform D-tracking
        arm = self.tracking(allocation)

        # Check stopping criterion
        stop = self.stopping_criterion(optimal_policy)

        misc = {
            "br_value": br_value,
            "br_instance": br_instance,
            "allocation": allocation,
            "optimal_policy": optimal_policy,
        }

        return arm, stop, optimal_policy, misc


class ProjectedTnS(TnS):
    """
    Track-n-stop style of algorithm for bandits with linear constraints. That computes the allocation according to normal BAI problem
    """

    def __init__(
        self,
        n_arms,
        A,
        b,
        delta,
        ini_phase=1,
        sigma=1,
        restricted_exploration=False,
        dist_type="Gaussian",
        seed=None,
        d_tracking=True,
    ):
        """
        Initialize the explorer
        :param n_arms: number of arms
        :param A: matrix constraints
        :param b: vector constraints
        :param delta: confidence parameter
        :param ini_phase: initial phase (how many times to play each arm before adaptive search starts)
        :param sigma: standard deviation of Gaussian distribution
        :param restricted_exploration: whether to use restricted exploration or not
        :param dist_type: distribution type to use for projection
        :param seed: random seed
        :param d_tracking: D-tracking or C-tracking
        """
        super().__init__(
            n_arms,
            A,
            b,
            delta,
            ini_phase,
            sigma,
            restricted_exploration,
            dist_type,
            seed,
            d_tracking,
        )
        self.unconstrained_neighbors = {}

    def act(self):
        """
        Choose an arm to play
        """
        if self.t < self.n_arms * self.ini_phase:
            # Initial phase play each arm once
            arm = self.t % self.n_arms
            return arm, False, None, None

        # Compute optimal policy w.r.t. current empirical means
        optimal_policy, aux = get_policy(mu=self.means, A=self.A, b=self.b)
        optimal_arm, arm_aux = get_policy(mu=self.means, A=None, b=None)

        # Check if policy already visited. If yes retrieve neighbors otherwise compute neighbors
        hash_tuple = tuple(optimal_policy.tolist())  # npy not hashable
        if not hash_tuple in self.neighbors:
            neighbors = compute_neighbors(
                optimal_policy, aux["A"], aux["b"], slack=aux["slack"]
            )
            self.neighbors[hash_tuple] = neighbors

        # Check if optimal arm is already visited
        hash_tuple = tuple(optimal_arm.tolist())  # npy not hashable
        if hash_tuple in self.unconstrained_neighbors:
            neighbors = self.unconstrained_neighbors[hash_tuple]
        else:
            neighbors = compute_neighbors(
                optimal_arm, arm_aux["A"], arm_aux["b"], slack=arm_aux["slack"]
            )
            self.unconstrained_neighbors[hash_tuple] = neighbors

        # Solve game to get allocation
        allocation, game_value = solve_game(
            mu=self.means,
            vertex=optimal_arm,
            neighbors=neighbors,
            dist_type=self.dist_type,
            sigma=self.sigma,
            allocation_A=None,
            allocation_b=None,
        )

        if self.restricted_exploration:
            allocation = project_on_feasible(allocation, self.A, self.b)

        # Check if forced exploration is needed else D-tracking
        not_saturated = self.n_pulls < (np.sqrt(self.t) - self.n_arms / 2)
        if not_saturated.any() and self.d_tracking:
            # Play smallest N below sqrt(t) - n_arms/2
            arm = np.argmin(self.n_pulls)

        else:
            # Play arm according to tracking rule
            arm = self.tracking(allocation)

        # Check stopping criterion
        stop = self.stopping_criterion(optimal_policy)

        misc = {
            "game_value": game_value,
            "allocation": allocation,
            "optimal_policy": optimal_policy,
        }

        return arm, stop, optimal_policy, misc


class UniformExplorer(Explorer):
    """
    Uniform explorer for bandits with linear constraints. If restricted during exploration, the uniform policy is projected onto the feasible set.
    """

    def __init__(
        self,
        n_arms,
        A,
        b,
        delta,
        ini_phase=1,
        sigma=1,
        restricted_exploration=False,
        dist_type="Gaussian",
        seed=None,
        allocation=None,
    ):
        super().__init__(
            n_arms,
            A,
            b,
            delta,
            ini_phase,
            sigma,
            restricted_exploration,
            dist_type,
            seed,
        )

        if allocation is None:
            self.allocation = np.ones(n_arms) / n_arms
        else:
            self.allocation = allocation

        if self.restricted_exploration:
            self.allocation = project_on_feasible(self.allocation, self.A, self.b)

    def act(self):
        """
        Choose an arm to play
        """

        # Play each arm once
        if self.t < self.n_arms * self.ini_phase:
            arm = self.t % self.n_arms
            return arm, False, None, None

        # Get optimal policy
        optimal_policy, aux = get_policy(mu=self.means, A=self.A, b=self.b)

        # Check neighbors
        hash_tuple = tuple(optimal_policy.tolist())  # npy not hashable
        if hash_tuple in self.neighbors:
            neighbors = self.neighbors[hash_tuple]
        else:
            neighbors = compute_neighbors(
                optimal_policy, aux["A"], aux["b"], slack=aux["slack"]
            )
            self.neighbors[hash_tuple] = neighbors

        # Sample arm 'uniformly'
        arm = self.random_state.choice(self.n_arms, p=self.allocation)

        # Check stopping criterion
        stop = self.stopping_criterion(optimal_policy)

        misc = {"allocation": self.allocation, "optimal_policy": optimal_policy}

        return arm, stop, optimal_policy, misc


########### BANDIT ENVIRONMENTS ############ß


class Bandit:
    """
    Generic bandit class
    """

    def __init__(self, expected_rewards, seed=None):
        self.n_arms = len(expected_rewards)
        self.expected_rewards = expected_rewards
        self.seed = seed
        self.random_state = np.random.RandomState(seed)

    def sample(self):
        pass

    def get_means(self):
        return self.expected_rewards


class GaussianBandit(Bandit):
    """
    Bandit with gaussian rewards
    """

    def __init__(self, expected_rewards, seed=None, noise=1):
        super(GaussianBandit, self).__init__(expected_rewards, seed)
        self.noise = noise

    def sample(self):
        return self.random_state.normal(self.expected_rewards, self.noise)


class BernoulliBandit(Bandit):
    """
    Bandit with bernoulli rewards
    """

    def __init__(self, expected_rewards, seed=None):
        super(BernoulliBandit, self).__init__(expected_rewards, seed)

    def sample(self):
        return self.random_state.binomial(1, self.expected_rewards)


########### Regret Minimizers ############

from scipy.special import softmax


class AdaHedge:
    """
    AdaHedge algorithm from https://arxiv.org/pdf/1301.0534.pdf
    """

    def __init__(self, d, loss_rescale=1):
        """

        :param d: number of arms
        :param loss_rescale: rescale loss to avoid numerical issues
        """
        self.alpha = 4  # 4 #np.sqrt(np.log(d))
        self.w = np.ones(d) / d
        self.theta = np.zeros(d)
        self.t = 0
        self.gamma = 0
        self.loss_rescale = loss_rescale

    def random_weights(self):
        w = np.random.uniform(1, 2, size=len(self.w))
        self.w = w / np.sum(w)

    def get_weights(self):
        """
        Get weights
        """
        return self.w

    def update(self, loss):
        """
        Update weights in AdaHedge, see https://parameterfree.com/2020/05/03/adahedge/
        :param loss:
        :return:
        """
        self.t += 1
        loss = loss * self.loss_rescale
        self.theta = self.theta - loss
        total_loss = (self.w * loss).sum()
        if self.t == 1:
            delta = total_loss - loss.min() + 1e-5
        else:
            delta = (
                self.gamma * np.log(np.sum(self.w * np.exp(-loss / self.gamma)))
                + total_loss
            )
        self.gamma += delta / (self.alpha**2)
        logits = self.theta / self.gamma
        self.w = softmax(logits - logits.max())


from scipy.linalg import sqrtm


class AdaGrad:
    """
    AdaGrad Algorithm
    """

    def __init__(self, A, b, loss_rescale=1):
        self.eta = 1 / np.sqrt(2)
        self.d = A.shape[1]
        self.t = 0
        self.loss_rescale = loss_rescale
        self.w = np.ones(self.d) / self.d
        self.w = np.clip(project_on_feasible(self.w, A, b), 1e-12, 1 - 1e-12)
        self.H = 0
        self.A = A
        self.b = b
        self.loss_sequence = 0
        self.neg_entropy = lambda x: (x * np.log(x + PRECISION)).sum()
        self.delta = 0.1

    def get_weights(self):
        """
        Get weights
        """
        return self.w

    def update(self, loss):
        self.t += 1
        loss = loss * self.loss_rescale
        self.loss_sequence += loss
        self.H = self.H + np.outer(loss, loss)
        H = self.H + self.delta * np.eye(self.d)
        H_inv = np.linalg.pinv(sqrtm(H))

        def objective(x):
            return np.abs(x - self.w + self.eta * np.matmul(H_inv, loss)).sum()

        constraint = LinearConstraint(self.A, ub=self.b)
        res = minimize(objective, self.w, constraints=constraint)
        self.w = np.clip(res.x, 1e-12, 1 - 1e-12)


class OnlineGradientDescent:
    """
    Online Gradient Descent
    """

    def __init__(self, d, ini_lr=1):
        self.n_arms = d
        self.ini_lr = ini_lr
        self.allocation = np.ones(d) / d
        self.t = 0

    def get_weights(self):
        """
        Get weights
        """
        return self.allocation

    def update(self, loss):
        """
        Update weights
        """
        self.t += 1
        lr = self.ini_lr / np.sqrt(self.t)
        self.allocation -= lr * loss
        self.allocation = project_on_feasible(self.allocation, None, None)


########### EXPLORATION EXPERIMENT ############
import time


def run_exploration_experiment(bandit: Bandit, explorer: Explorer):
    """
    Run pure-exploration experiment for a given explorer and return stopping time and correctness
    """

    optimal_policy = np.zeros_like(bandit.expected_rewards)
    optimal_policy[np.argmax(bandit.expected_rewards)] = 1 
    done = False
    t = 0
    running_times = []
    while not done:
        t += 1
        if t % 100 == 0:
            print(t)
        # Act
        running_time = time.time()
        arm, done, policy, log = explorer.act()
        running_time = time.time() - running_time
        running_times.append(running_time)
        # Observe reward
        reward = bandit.sample()[arm]
        # Update explorer
        explorer.update(arm, reward)

    # Check correctness
    correct = np.array_equal(optimal_policy, policy)

    # Return stopping time, correctness, optimal policy and recommended policy
    return t, correct, optimal_policy, policy, np.mean(running_times)


def run_imdb_exp(bandit, explorer, A, b):
    """
    Run pure exploration experiment on IMDB dataset (or other env with gym-like interface)
    """
    running_times = []
    optimal_policy, _ = get_policy(bandit.get_means(), A, b)
    done = False
    t = 0

    while not done:
        t += 1
        # Act
        running_time = time.time()
        arm, done, policy, log = explorer.act()
        running_time = time.time() - running_time
        running_times.append(running_time)
        # Observe reward
        reward, _, _, _ = bandit.step(arm)
        # step in env
        explorer.update(arm, reward)

    # Check correctness
    correct = np.array_equal(optimal_policy, policy)

    # Return stopping time, correctness, optimal policy and recommended policy
    return t, correct, optimal_policy, policy, np.mean(running_times)


########### TESTING ############

if __name__ == "__main__":
    delta = 0.01
    mu = np.array([1.5, 1, 0.5, 0.4, 0.3, 0.2])
    A = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0]])
    b = np.array([0.5, 0.5])

    optimal_policy, _ = get_policy(mu, A, b)
    print(f" Optimal : {optimal_policy}")
    explorer = TnS(
        len(mu),
        A=A,
        b=b,
        delta=delta,
        restricted_exploration=True,
        dist_type="Gaussian",
    )
    bandit = GaussianBandit(mu)
    t, correct, _, policy, _ = run_exploration_experiment(bandit, explorer, A, b)
    print(f"Stopped at {t} with correct policy {correct}, Rec policy {policy}")

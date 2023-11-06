"""Optimizer algorithms."""

import logging
import math
import warnings
from time import time

# import botorch
import gpytorch
import torch


logger = logging.getLogger("optimizer")


# Optimizers


class Optimizer:
    """Optimizer base class."""

    def __init__(self):
        self.name = self.__class__.__name__
        logger.info(f"Init: {self.name}")

    def fit(self, df_train):
        """Fit the optimizer with training data."""
        return self

    def predict(self, df_test):
        """Predict the objective(s) with the optimizer."""
        raise NotImplementedError

    def evaluate(self, df_test):
        """Evaluate the predictive performance of the optimizer."""
        raise NotImplementedError

    def acquisition_function(self, df_test):
        """Evaluate the acquisition function."""
        raise NotImplementedError

    def select_candidate(self, df_test):
        """Select a candidate with the optimizer."""
        raise NotImplementedError


class RandomOptimizer(Optimizer):
    """Random optimizer."""

    def __init__(self):
        super().__init__()

    def select_candidate(self, df_test):
        """Select a random candidate."""
        return df_test.sample(n=1).squeeze()


class SingleTaskSingleObjectiveOptimizer(Optimizer):
    """Single-task single-objective optimizer."""

    def __init__(self, x_columns, y_column, maximize=True):
        super().__init__()
        self.x_columns = x_columns
        self.y_column = y_column
        self.maximize = maximize
        self.trained = False

    def fit(self, df_train, training_steps=200):
        """Fit the optimizer with training data."""
        # Prepare training data
        X_train = torch.tensor(df_train[self.x_columns].values)
        y_train = torch.tensor(df_train[self.y_column].values)
        # TODO: Standardize y and save the standardization parameters
        # Find best y_train to use in the acquisition function
        if self.maximize:
            self.best_y = y_train.max()
        else:
            self.best_y = y_train.min()
        # Setup
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = SingleTaskGP(X_train, y_train, self.likelihood)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        # Training loop
        logger.info(f"Start training with {list(X_train.shape)} data for {training_steps} steps")
        self.likelihood.train()
        self.model.train()
        # TODO: Progress bar
        start_time = time()
        self.losses = []
        for step in range(training_steps):
            optimizer.zero_grad()
            output = self.model(X_train)
            loss = -mll(output, y_train)
            loss.backward()
            self.losses.append(loss.item())
            logging.debug(f"Step {step + 1:3d}/{training_steps},  Loss: {loss.item():.3f}")
            optimizer.step()
        # Done training
        logger.info(f"Finished training in {time() - start_time:.2f} seconds")
        logger.debug(f"Losses: {self.losses}")
        self.trained = True
        return self

    def predict(self, df_test):
        """Predict the objective with the optimizer."""
        assert self.trained, "The optimizer must be trained first"
        # Prepare test data
        X_test = torch.tensor(df_test[self.x_columns].values)
        # Predict
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f_pred = self.model(X_test)  # model posterior distribution
            y_pred = self.likelihood(f_pred)  # posterior predictive distribution
            y_pred_mean = y_pred.mean
            y_pred_var = y_pred.variance
            y_pred_lower, y_pred_upper = y_pred.confidence_region()
        # TODO: De-standardize predictions
        # logger.warning("Predictions are not de-standardized")
        return y_pred_mean, y_pred_var, y_pred_lower, y_pred_upper

    def evaluate(self, df_test):
        """Evaluate the predictive performance of the optimizer."""
        assert self.trained, "The optimizer must be trained first"
        # Prepare test data
        y_test = torch.tensor(df_test[self.y_column].values)
        # TODO: standardize y_test if predictions are not de-standardized
        # Predict
        with warnings.catch_warnings():
            # Ignore warnings about predictig with the training data
            warnings.simplefilter("ignore")
            y_pred_mean, _, _, _ = self.predict(df_test)
        # Evaluate R^2
        r2 = compute_r2(y_test, y_pred_mean).item()
        logger.info(f"R^2: {r2:.4f}")
        return r2

    def acquisition_function(self, df_test):
        """Evaluate the acquisition function."""
        assert self.trained, "The optimizer must be trained first"
        # Prepare test data
        X_test = torch.tensor(df_test[self.x_columns].values)
        # Evaluate acquisition function
        self.model.eval()
        # acqf = botorch.acquisition.ExpectedImprovement(
        #     self.model, best_f=self.best_y, maximize=self.maximize
        # )
        acqf = ExpectedImprovement(self.model, best_f=self.best_y, maximize=self.maximize)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # acq = acqf(X_test.unsqueeze(1))  # Add a batch dimension required by botorch
            acq = acqf(X_test)
        return acq

    def select_candidate(self, df_test):
        """Select a candidate with the optimizer."""
        # Asserts are checked in acquisition_function
        acq = self.acquisition_function(df_test)
        index = torch.argmax(acq).item()
        return df_test.iloc[index].copy(), index


class MultiTaskSingleObjectiveOptimizer(Optimizer):
    """Multi-task single-objective optimizer."""

    def __init__(self, x_columns, t_column, y_column, maximize=True, separate_noise=False):
        super().__init__()
        self.x_columns = x_columns
        self.t_column = t_column
        self.y_column = y_column
        self.maximize = maximize
        self.trained = False
        self.separate_noise = separate_noise

    def fit(self, df_train, training_steps=200):
        """Fit the optimizer with training data."""
        assert df_train[self.t_column].dtype == "int64", "Task column must be integer"
        assert df_train[self.t_column].nunique() > 1, "There must be more than one task"
        self.tasks = df_train[self.t_column].unique().tolist()
        # Prepare training data
        X_train = torch.tensor(df_train[self.x_columns].values, dtype=torch.double)
        t_train = torch.tensor(df_train[self.t_column].values, dtype=torch.long)
        y_train = torch.tensor(df_train[self.y_column].values, dtype=torch.double)
        # TODO: Standardize y and save the standardization parameters
        # Find best y_train for each task to use in the acquisition function
        self.best_y = {}
        for i in self.tasks:
            if self.maximize:
                self.best_y[i] = y_train[t_train == i].max()
            else:
                self.best_y[i] = y_train[t_train == i].min()
        # Setup
        if self.separate_noise:
            self.likelihood = MultiTaskGaussianLikelihood(num_tasks=len(self.tasks))
        else:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = MultiTaskGP(X_train, t_train, y_train, self.likelihood)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        # Training loop
        logger.info(f"Start training with {list(X_train.shape)} data for {training_steps} steps")
        self.likelihood.train()
        self.model.train()
        # TODO: Progress bar
        start_time = time()
        self.losses = []
        for step in range(training_steps):
            optimizer.zero_grad()
            output = self.model(X_train, t_train)
            loss = -mll(output, y_train, t_train)
            loss.backward()
            self.losses.append(loss.item())
            logging.debug(f"Step {step + 1:3d}/{training_steps},  Loss: {loss.item():.3f}")
            optimizer.step()
        # Done training
        logger.info(f"Finished training in {time() - start_time:.2f} seconds")
        logger.info(f"Task correlation matrix:\n{self.model.task_correlation_matrix()}")
        logger.info(f"Likelihood noise:\n{[n.detach().sqrt() for n in list(self.likelihood.noise)]}")
        logger.debug(f"Losses: {self.losses}")
        self.trained = True
        return self

    def predict(self, df_test):
        """Predict the objective with the optimizer."""
        assert self.trained, "The optimizer must be trained first"
        assert df_test[self.t_column].dtype == "int64", "Task column must be integer"
        for i in df_test[self.t_column].unique():
            assert i in self.tasks, f"Task {i} not in known tasks {self.tasks}"
        # Prepare test data
        X_test = torch.tensor(df_test[self.x_columns].values, dtype=torch.double)
        t_test = torch.tensor(df_test[self.t_column].values, dtype=torch.long)
        # Predict
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f_pred = self.model(X_test, t_test)  # model posterior distribution
            y_pred = self.likelihood(f_pred, t_test)  # posterior predictive distribution
            y_pred_mean = y_pred.mean
            y_pred_var = y_pred.variance
            y_pred_lower, y_pred_upper = y_pred.confidence_region()
        # TODO: De-standardize predictions
        # logger.warning("Predictions are not de-standardized")
        return y_pred_mean, y_pred_var, y_pred_lower, y_pred_upper

    def evaluate(self, df_test):
        """Evaluate the predictive performance of the optimizer."""
        assert self.trained, "The optimizer must be trained first"
        assert df_test[self.t_column].dtype == "int64", "Task column must be integer"
        for i in df_test[self.t_column].unique():
            assert i in self.tasks, f"Task {i} not in known tasks {self.tasks}"
        # Prepare test data
        y_test = torch.tensor(df_test[self.y_column].values, dtype=torch.double)
        # TODO: standardize y_test if predictions are not de-standardized
        # Predict
        with warnings.catch_warnings():
            # Ignore warnings about predictig with the training data
            warnings.simplefilter("ignore")
            y_pred_mean, _, _, _ = self.predict(df_test)
        # Evaluate R^2
        r2 = compute_r2(y_test, y_pred_mean).item()
        logger.info(f"R^2: {r2:.4f}")
        return r2

    def acquisition_function(self, df_test):
        """Evaluate the acquisition function."""
        assert self.trained, "The optimizer must be trained first"
        assert df_test[self.t_column].dtype == "int64", "Task column must be integer"
        assert len(df_test[self.t_column].unique()) == 1, "Only works for a single task"
        i = df_test[self.t_column].unique().item()
        assert i in self.tasks, f"Task {i} not in known tasks {self.tasks}"
        # Prepare test data
        X_test = torch.tensor(df_test[self.x_columns].values, dtype=torch.double)
        t_test = torch.tensor(df_test[self.t_column].values, dtype=torch.long)
        # Evaluate acquisition function
        self.model.eval()
        # acqf = botorch.acquisition.ExpectedImprovement(
        #     self.model, best_f=self.best_y[t], maximize=self.maximize
        # )
        acqf = ExpectedImprovement(self.model, best_f=self.best_y[i], maximize=self.maximize)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # acq = acqf(Xt_test.unsqueeze(1))  # Add a batch dimension required by botorch
            acq = acqf(X_test, t_test)
        return acq

    def select_candidate(self, df_test):
        """Select a candidate with the optimizer."""
        # Asserts are checked in acquisition_function
        acq = self.acquisition_function(df_test)
        index = torch.argmax(acq).item()
        return df_test.iloc[index].copy(), index


# Gaussian process models


class SingleTaskGP(gpytorch.models.ExactGP):
    """Single-task GP model."""

    def __init__(self, X_train, y_train, likelihood, covar_module=None, mean_module=None):
        assert X_train.shape[0] == y_train.shape[0]
        assert X_train.ndim == 2
        super().__init__(X_train, y_train, likelihood)
        # self.num_outputs = 1  # Part of the botorch API
        self.mean_module = mean_module or gpytorch.means.ConstantMean()
        self.covar_module = covar_module or gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(ard_num_dims=X_train.shape[-1])
        )

    def forward(self, X):
        assert X.ndim == 2, "X is expected to be 2D"
        mean = self.mean_module(X)
        covar = self.covar_module(X)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    # def posterior(self, X, posterior_transform=None):
    #     # This method is part of the botorch API and is alled by the botorch acquisition function
    #     assert posterior_transform is None, "posterior_transform is not supported"
    #     mvn = self(X)
    #     # mvn = self.likelihood(self(X))
    #     posterior = botorch.posteriors.gpytorch.GPyTorchPosterior(distribution=mvn)
    #     return posterior


class MultiTaskGP(gpytorch.models.ExactGP):
    """Multi-task GP model.

    Based on:
    - https://docs.gpytorch.ai/en/latest/examples/03_Multitask_Exact_GPs/Hadamard_Multitask_GP_Regression.html  # noqa: E501
    - https://botorch.org/api/_modules/botorch/models/multitask.html#MultiTaskGP

    Note: The botorch model wraps the kernel in a scale kernel, but I think that may be
    overparameterizing the kernel, since it is also scaled by the task kernel (IndexKernel).

    The tasks share noise and lengthscale parameters in the likelihood and data kernel,
    but are scaled differently through the task kernel.
    """

    def __init__(self, X_train, t_train, y_train, likelihood, covar_module=None, mean_module=None):
        assert X_train.shape[0] == t_train.shape[0] == y_train.shape[0]
        assert X_train.ndim == 2
        super().__init__((X_train, t_train), y_train, likelihood)
        # self.num_outputs = 1  # Part of the botorch API
        self.mean_module = mean_module or gpytorch.means.ConstantMean()
        # Feature covariance
        self.covar_module = covar_module or gpytorch.kernels.MaternKernel(
            ard_num_dims=X_train.shape[-1]
        )
        # Task covariance
        self.tasks = t_train.unique().to(dtype=torch.long).tolist()
        num_tasks = len(self.tasks)
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=num_tasks)

    def forward(self, X, t):
        assert X.shape[0] == t.shape[0]
        assert X.ndim == 2, "X is expected to be 2D"
        for i in t.unique():
            assert i in self.tasks, f"Task {i} not in known tasks {self.tasks}"
        mean = self.mean_module(X)
        covar = self.covar_module(X) * self.task_covar_module(t)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def task_covariance_matrix(self):
        """Compute task covariance matrix."""
        return self.task_covar_module._eval_covar_matrix().detach()

    def task_correlation_matrix(self):
        """Compute task correlation matrix from task covariance matrix.

        The correlation matrix can be computed from the task covaraince matrix:
        - https://en.wikipedia.org/wiki/Correlation#Correlation_matrices
        - https://en.wikipedia.org/wiki/Covariance_matrix
        """
        covm = self.task_covariance_matrix()
        stds = covm.diagonal().sqrt().unsqueeze(0)
        corr = covm / (stds.t() @ stds)
        return corr

    # def posterior(self, X, posterior_transform=None):
    #     # This method is part of the botorch API and is alled by the botorch acquisition function
    #     assert posterior_transform is None, "posterior_transform is not supported"
    #     Xt = X  # Expects the last column of Xt to be the task column
    #     mvn = self(Xt)
    #     # mvn = self.likelihood(self(X))
    #     posterior = botorch.posteriors.gpytorch.GPyTorchPosterior(distribution=mvn)
    #     return posterior


# Likelihood functions


class MultiTaskGaussianLikelihood(gpytorch.likelihoods._GaussianLikelihoodBase):
    """Gaussian likelihood with a separate heteroskedastic noise model for each task.

    Must inherit _GaussianLikelihoodBase to be compatible with gpytorch ExactMarginalLogLikelihood.

    Tasks must be indexed by integers from 0 to num_tasks-1.
    """

    def __init__(self, num_tasks):
        super().__init__(noise_covar=None)
        # Create a separate GaussianLikelihood, i.e. a separate noise model, for each task
        self.task_likelihoods = torch.nn.ModuleList(
            [gpytorch.likelihoods.GaussianLikelihood() for _ in range(num_tasks)]
        )

    def _shaped_noise_covar(self, base_shape, *params, **kwargs):
        # From _GaussianLikelihoodBase
        raise NotImplementedError()

    def expected_log_prob(self, target, input, *params, **kwargs):
        # From _GaussianLikelihoodBase
        raise NotImplementedError()

    def forward(self, function_samples, t, *params, **kwargs):
        # From _GaussianLikelihoodBase
        raise NotImplementedError()
        # assert function_samples.shape == t.shape
        # noise = torch.zeros_like(function_samples)
        # for i in t.unique():
        #     noise[t == i] = self.task_likelihoods[t].noise
        # return gpytorch.distributions.Normal(function_samples, noise.sqrt())

    def log_marginal(self, observations, function_dist, *params, **kwargs):
        # From _GaussianLikelihoodBase
        raise NotImplementedError()

    @property
    def noise(self):
        # From GaussianLikelihood
        return [m.noise for m in self.task_likelihoods]

    @noise.setter
    def noise(self, value):
        # From GaussianLikelihood
        raise NotImplementedError

    @property
    def raw_noise(self):
        # From GaussianLikelihood
        raise NotImplementedError()

    @raw_noise.setter
    def raw_noise(self, value):
        # From GaussianLikelihood
        raise NotImplementedError

    def marginal(self, function_dist, params):
        """Analytic marginal.

        Overwrite _GaussianLikelihoodBase.marginal() to use separate noise models for each task.
        """
        # When marginal is called by prediction_strategy in ExactGP.__call__,
        # params is a list of the model training inputs.
        # Otherwise, params is assumed to be the task indices.
        if isinstance(params, list):
            assert self.training is False  # eval mode
            assert len(params) == 2, params
            t = params[1].squeeze()
        else:
            assert isinstance(params, torch.Tensor)
            assert len(params) == len(function_dist.mean)
            t = params.squeeze()
        # Prepare the noise covariance matrix
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        noise = torch.zeros_like(mean)
        for i in t.unique():
            noise[t == i] = self.task_likelihoods[int(i)].noise
        noise_covar = torch.diag(noise)
        # Compute the marginal distribution by adding the noise covar to the data covar
        full_covar = covar + noise_covar
        return function_dist.__class__(mean, full_covar)


# Acquisition functions


class ExpectedImprovement():
    """Expected improvement acquisition function.

    Simplified version of botorch.acquisition.ExpectedImprovement.
    """

    def __init__(self, model, best_f, maximize=True):
        self.model = model
        self.best_f = best_f
        self.maximize = maximize
        self.min_var = 1e-12

    def forward(self, X, *params, **kwargs):
        # Compute model posterior
        mvn = self.model(X, *params, **kwargs)
        # TODO: Apply the likelihood to include the noise?
        # mvn = self.model.likelihood(mvn)
        mean = mvn.mean
        sigma = mvn.variance.clamp_min(self.min_var).sqrt()
        # Compute expected improvement
        u = (mean - self.best_f) / sigma
        u = u if self.maximize else -u
        ei = phi(u) + u * Phi(u)
        return sigma * ei

    def __call__(self, X, *params, **kwargs):
        return self.forward(X, *params, **kwargs)


def phi(x):
    r"""Standard normal PDF."""
    inv_sqrt_2pi = 1 / math.sqrt(2 * math.pi)
    return inv_sqrt_2pi * (-0.5 * x.square()).exp()


def Phi(x):
    r"""Standard normal CDF."""
    inv_sqrt_2 = 1 / math.sqrt(2)
    return 0.5 * torch.erfc(-inv_sqrt_2 * x)


# Utility functions


def compute_r2(y_true, y_pred):
    """Compute R^2 (coefficient of determination)."""
    ssr = torch.sum((y_true - y_pred)**2)
    sst = torch.sum((y_true - torch.mean(y_true))**2)
    return 1 - (ssr / sst)

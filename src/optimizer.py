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

    def __init__(self, x_columns, t_column, y_column, maximize=True):
        super().__init__()
        self.x_columns = x_columns
        self.t_column = t_column
        self.y_column = y_column
        self.maximize = maximize
        self.trained = False

    def fit(self, df_train, training_steps=200):
        """Fit the optimizer with training data."""
        assert df_train[self.t_column].dtype == "int64", "Task column must be integer"
        assert df_train[self.t_column].nunique() > 1, "There must be more than one task"
        self.tasks = df_train[self.t_column].unique().tolist()
        # Prepare training data
        Xt_train = torch.tensor(
            df_train[self.x_columns + [self.t_column]].values, dtype=torch.double)
        y_train = torch.tensor(df_train[self.y_column].values, dtype=torch.double)
        # TODO: Standardize y and save the standardization parameters
        # Find best y_train for each task to use in the acquisition function
        self.best_y = {}
        for t in self.tasks:
            if self.maximize:
                self.best_y[t] = y_train[Xt_train[:, -1] == t].max()
            else:
                self.best_y[t] = y_train[Xt_train[:, -1] == t].min()
        # Setup
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = MultiTaskGP(Xt_train, y_train, self.likelihood)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        # Training loop
        logger.info(f"Start training with {list(Xt_train.shape)} data for {training_steps} steps")
        self.likelihood.train()
        self.model.train()
        # TODO: Progress bar
        start_time = time()
        self.losses = []
        for step in range(training_steps):
            optimizer.zero_grad()
            output = self.model(Xt_train)
            loss = -mll(output, y_train)
            loss.backward()
            self.losses.append(loss.item())
            logging.debug(f"Step {step + 1:3d}/{training_steps},  Loss: {loss.item():.3f}")
            optimizer.step()
        # Done training
        logger.info(f"Finished training in {time() - start_time:.2f} seconds")
        logger.info(f"Task correlation matrix:\n{self.model.task_correlation_matrix()}")
        logger.debug(f"Losses: {self.losses}")
        self.trained = True
        return self

    def predict(self, df_test):
        """Predict the objective with the optimizer."""
        assert self.trained, "The optimizer must be trained first"
        assert df_test[self.t_column].dtype == "int64", "Task column must be integer"
        for t in df_test[self.t_column].unique():
            assert t in self.tasks, f"Task {t} not in known tasks {self.tasks}"
        # Prepare test data
        Xt_test = torch.tensor(df_test[self.x_columns + [self.t_column]].values, dtype=torch.double)
        # Predict
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f_pred = self.model(Xt_test)  # model posterior distribution
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
        assert df_test[self.t_column].dtype == "int64", "Task column must be integer"
        for t in df_test[self.t_column].unique():
            assert t in self.tasks, f"Task {t} not in known tasks {self.tasks}"
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
        t = df_test[self.t_column].unique().item()
        assert t in self.tasks, f"Task {t} not in known tasks {self.tasks}"
        # Prepare test data
        Xt_test = torch.tensor(df_test[self.x_columns + [self.t_column]].values, dtype=torch.double)
        # Evaluate acquisition function
        self.model.eval()
        # acqf = botorch.acquisition.ExpectedImprovement(
        #     self.model, best_f=self.best_y[t], maximize=self.maximize
        # )
        acqf = ExpectedImprovement(self.model, best_f=self.best_y[t], maximize=self.maximize)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # acq = acqf(Xt_test.unsqueeze(1))  # Add a batch dimension required by botorch
            acq = acqf(Xt_test)
        return acq

    def select_candidate(self, df_test):
        """Select a candidate with the optimizer."""
        # Asserts are checked in acquisition_function
        acq = self.acquisition_function(df_test)
        index = torch.argmax(acq).item()
        return df_test.iloc[index].copy(), index


# GP models


class SingleTaskGP(gpytorch.models.ExactGP):
    """Single-task GP model."""

    def __init__(self, X_train, y_train, likelihood, covar_module=None, mean_module=None):
        super().__init__(X_train, y_train, likelihood)
        # self.num_outputs = 1  # Part of the botorch API
        self.mean_module = mean_module or gpytorch.means.ConstantMean()
        self.covar_module = covar_module or gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(ard_num_dims=X_train.shape[-1])
        )

    def forward(self, X):
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

    Based on: https://botorch.org/api/_modules/botorch/models/multitask.html#MultiTaskGP

    The botorch model wraps the kernel in a scale kernel, but I think that may be overparameterizing
    the kernel, since it is also scaled by the task kernel (IndexKernel).

    The tasks share noise and lengthscale parameters in the likelihood and data kernel,
    but are scaled differently through the task kernel.
    """

    def __init__(self, Xt_train, y_train, likelihood, covar_module=None, mean_module=None):
        # Expects the last column of Xt_train to be the task identifier
        super().__init__(Xt_train, y_train, likelihood)
        # self.num_outputs = 1  # Part of the botorch API
        self.mean_module = mean_module or gpytorch.means.ConstantMean()
        # Feature covariance
        self.covar_module = covar_module or gpytorch.kernels.MaternKernel(
            ard_num_dims=Xt_train.shape[-1] - 1  # Exclude the task column
        )
        # Task covariance
        self.tasks = Xt_train[:, -1].unique().to(dtype=torch.long).tolist()
        num_tasks = len(self.tasks)
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=num_tasks)

    def forward(self, Xt):
        # Expects the last column of Xt to be the task column
        assert Xt.ndim == 2, "Xt is expected to be 2D"
        X = Xt[:, :-1]  # Feature columns
        t = Xt[:, -1].to(dtype=torch.long)   # Task column
        for task in t.unique():
            assert task in self.tasks, f"Task {task} not in known tasks {self.tasks}"
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

    def forward(self, X):
        # Compute model posterior
        mvn = self.model(X)
        # TODO: Apply the likelihood to include the noise?
        # mvn = self.model.likelihood(mvn)
        mean = mvn.mean
        sigma = mvn.variance.clamp_min(self.min_var).sqrt()
        # Compute expected improvement
        u = (mean - self.best_f) / sigma
        u = u if self.maximize else -u
        ei = phi(u) + u * Phi(u)
        return sigma * ei

    def __call__(self, X):
        return self.forward(X)


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

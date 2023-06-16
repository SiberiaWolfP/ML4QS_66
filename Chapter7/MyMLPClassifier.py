from util.common import GPU
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, ExponentialLR
import numpy as np
import pandas as pd
if GPU:
    import cupy as cp
    import cudf as cd


# Define the MLP network
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes, output_dim, activation):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = {
            'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'logistic': nn.Sigmoid(), 'identity': nn.Identity()
        }

        layer_dims = [input_dim, *hidden_layer_sizes]

        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                self.layers.append(self.activations[activation])

        self.layers.append(nn.Linear(layer_dims[-1], output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MyMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate='constant', learning_rate_init=0.001, max_iter=200,
                 early_stopping=False, power_t=0.5,
                 validation_fraction=0.1, n_iter_no_change=10, tol=0.0001, random_state=None, verbose=False,
                 warm_start=False):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.power_t = power_t
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.classes_ = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

    def _init_model(self, input_dim, output_dim):
        if self.model is None or not self.warm_start:
            self.model = MLP(input_dim, self.hidden_layer_sizes, output_dim, self.activation)
        if self.solver == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha)
        elif self.solver == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha)
        else:
            raise ValueError("Only 'adam' and 'sgd' solvers are supported")

        if self.learning_rate == 'constant':
            self.scheduler = StepLR(self.optimizer, step_size=self.max_iter, gamma=1)
        elif self.learning_rate == 'invscaling':
            self.scheduler = ExponentialLR(self.optimizer, gamma=self.power_t)
        elif self.learning_rate == 'adaptive':
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=self.n_iter_no_change,
                                               threshold=self.tol)

        self.model = self.model.to(self.device)

    def fit(self, X, y):
        X = self._to_tensor(X)
        y = self._to_tensor(y)
        # X = torch.FloatTensor(X).to(self.device)
        # y = torch.LongTensor(y).to(self.device)
        n_samples, n_features = X.shape
        self.classes_ = torch.unique(y)
        n_classes = len(self.classes_)

        # Handle batch size
        if self.batch_size == 'auto':
            batch_size = min(200, n_samples)
        else:
            batch_size = self.batch_size

        # Initialize model
        if not self.warm_start or self.model is None:
            self._init_model(n_features, n_classes)

        best_val_loss = float('inf')
        no_improvement_count = 0
        X, X_val, y, y_val = train_test_split(X, y, test_size=self.validation_fraction,
                                              random_state=self.random_state, stratify=y.cpu().numpy())

        for epoch in range(self.max_iter):
            permutation = torch.randperm(X.size()[0])
            for i in range(0, X.size()[0], batch_size):
                self.optimizer.zero_grad()
                indices = permutation[i:i + batch_size]
                batch_x, batch_y = X[indices], y[indices]
                outputs = self.model(batch_x)
                loss = self.loss_fn(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

            if self.validation_fraction > 0:
                val_outputs = self.model(X_val)
                val_loss = self.loss_fn(val_outputs, y_val)
                if self.verbose:
                    print(f'Epoch: {epoch + 1}, Val loss: {val_loss.item():.4f}')
                if val_loss.item() < best_val_loss - self.tol:
                    best_val_loss = val_loss.item()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= self.n_iter_no_change:
                    if self.learning_rate == 'adaptive':
                        self.scheduler.step(val_loss)
                        if self.verbose:
                            print(f"Reduced learning rate to {self.optimizer.param_groups[0]['lr']}")
                        if self.optimizer.param_groups[0]['lr'] < self.learning_rate_init * 1e-3:
                            if self.verbose:
                                print("Learning rate too small. Stopping.")
                            break
                    else:
                        if self.verbose:
                            print(
                                f"Validation loss did not improve more than tol={self.tol} for {self.n_iter_no_change}"
                                f" consecutive epochs. Stopping.")
                        break
            # else:
                # self.scheduler.step()

            if self.verbose:
                print(f'Epoch: {epoch + 1}, Loss: {loss.item():.4f}')

        return self

    def predict_proba(self, X):
        X = self._to_tensor(X)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)].cpu().numpy()

    def score(self, X, y):
        X = self._to_tensor(X)
        y = self._to_tensor(y)
        y_pred = self.predict(X)
        return accuracy_score(y.cpu().numpy(), y_pred)

    def get_params(self, deep=True):
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': self.activation,
            'solver': self.solver,
            'alpha': self.alpha,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'learning_rate_init': self.learning_rate_init,
            'max_iter': self.max_iter,
            'early_stopping': self.early_stopping,
            'power_t': self.power_t,
            'validation_fraction': self.validation_fraction,
            'n_iter_no_change': self.n_iter_no_change,
            'tol': self.tol,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'warm_start': self.warm_start
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _to_tensor(self, X):
        # if isinstance(X, pd.DataFrame):
        if isinstance(X, cd.DataFrame):
            return torch.from_numpy(cp.asnumpy(X.values)).to(self.device)
        elif isinstance(X, cp.ndarray):
            return torch.from_numpy(cp.asnumpy(X.squeeze())).to(self.device)
        elif isinstance(X, np.ndarray):
            return torch.from_numpy(X.squeeze()).to(self.device)
        # elif X.shape[1] == 1:
        #     if dtype == torch.float:
        #         X = X.flatten().astype(np.float32)
        #     elif dtype == torch.int8:
        #         X = X.flatten().astype(np.int32)
        # return torch.tensor(X, dtype=dtype).to(self.device)

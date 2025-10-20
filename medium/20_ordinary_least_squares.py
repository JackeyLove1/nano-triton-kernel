'''
## OLS Regression on GPU — Problem Statement

Given a feature matrix (X \in \mathbb{R}^{n_{\text{samples}} \times n_{\text{features}}}) and a target vector (y \in \mathbb{R}^{n_{\text{samples}}}), compute the coefficient vector (\beta) that minimizes the sum of squared residuals:

$$\min_{\beta}\ \lVert X\beta - y \rVert^2$$

**Closed-form solution:**
$$\beta ;=; (X^{\mathsf T}X)^{-1} X^{\mathsf T} y$$

## Implementation Requirements
* External libraries are not permitted.
* The `solve` function signature must remain unchanged.
* The final coefficients must be stored in the **`beta`** vector.
* Assume the feature matrix (X) is full rank (i.e., (X^{\mathsf T}X) is invertible).

'''

import torch
import triton
import triton.language as tl

def solve(X: torch.Tensor, y: torch.Tensor, beta: torch.Tensor, n_samples: int, n_features: int):
    X = X.view(n_samples, n_features)
    X_T = torch.t(X)
    beta.copy_(torch.linalg.solve(X_T @ X, X_T @ y))


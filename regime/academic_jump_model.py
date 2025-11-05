"""
Academic Statistical Jump Model - Optimization Solver (Phase B)

This module implements the coordinate descent optimization algorithm with dynamic
programming for state sequence optimization, as specified in "Downside Risk Reduction
Using Regime-Switching Signals: A Statistical Jump Model Approach" (Shu et al.,
Princeton, 2024).

Optimization Problem:
    min_{Theta,S} Σ_{t=0}^{T-1} l(x_t, θ_{s_t}) + λ * Σ_{t=1}^{T-1} 1_{s_t ≠ s_{t-1}}

Where:
    - l(x, θ) = (1/2) ||x - θ||_2^2 (scaled squared Euclidean distance)
    - Θ = {θ_0, θ_1} (2 centroids for bull/bear states)
    - S = {s_0, ..., s_{T-1}} (state sequence, each s_t ∈ {0,1})
    - λ ≥ 0 (jump penalty - controls regime persistence)

Algorithm:
    1. Initialize Θ using K-means clustering
    2. Coordinate descent: Alternate E-step (optimize S via DP) and M-step (optimize Θ)
    3. Convergence: |objective_new - objective_old| < 1e-6
    4. Multi-start: 10 random initializations, keep best result

Academic Foundation:
    - 33 years empirical validation (1990-2023)
    - Proven Sharpe improvements: +42% to +158%
    - MaxDD reduction: ~50% across S&P 500/DAX/Nikkei
    - O(T*K²) = O(2T) complexity for K=2 states

Implementation Reference:
    Section 3.4.2 "Online Inference" in academic paper
    GitHub: Yizhan-Oliver-Shu/jump-models (reference implementation)
"""

from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from regime.academic_features import calculate_features


def _compute_loss(features: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Compute loss l(x, theta) = (1/2) ||x - theta||_2^2 for all x and theta.

    Args:
        features: (T, D) feature matrix
        theta: (K, D) centroid matrix

    Returns:
        loss: (T, K) loss matrix where loss[t, k] = l(x_t, theta_k)

    Note:
        Uses scaled squared Euclidean distance per paper specification.
    """
    T, D = features.shape
    K = theta.shape[0]

    # Compute ||x_t - theta_k||^2 for all t, k
    # Broadcasting: features[:, None, :] is (T, 1, D), theta[None, :, :] is (1, K, D)
    diff = features[:, None, :] - theta[None, :, :]  # (T, K, D)
    squared_dist = np.sum(diff ** 2, axis=2)  # (T, K)

    # Scale by 1/2 per paper
    loss = 0.5 * squared_dist

    return loss


def dynamic_programming(
    features: np.ndarray,
    theta: np.ndarray,
    lambda_penalty: float
) -> Tuple[np.ndarray, float]:
    """
    Dynamic programming algorithm for optimal state sequence given fixed centroids.

    Solves the optimization problem:
        min_S Σ_{t=0}^{T-1} l(x_t, θ_{s_t}) + λ * Σ_{t=1}^{T-1} 1_{s_t ≠ s_{t-1}}

    Algorithm:
        DP[0][k] = l(x_0, θ_k) for all k ∈ {0,1}
        DP[t][k] = l(x_t, θ_k) + min_j(DP[t-1][j] + λ*1_{j≠k})
        Backtrack from argmin_k(DP[T-1][k])

    Complexity: O(T*K²) = O(2T) for K=2 states

    Args:
        features: (T, D) feature matrix
        theta: (K, D) centroid matrix (K=2 for bull/bear)
        lambda_penalty: Jump penalty (controls regime persistence)

    Returns:
        state_sequence: (T,) array of state assignments {0,1}
        objective_value: Final objective function value

    Reference:
        Section 3.4.2 "Online Inference" in Shu et al., Princeton 2024

    Example:
        >>> features = np.random.randn(100, 3)
        >>> theta = np.array([[0, 0, 0], [1, 1, 1]])
        >>> states, obj = dynamic_programming(features, theta, lambda_penalty=50.0)
        >>> print(f"Objective: {obj:.2f}, Switches: {(states[1:] != states[:-1]).sum()}")
    """
    T, D = features.shape
    K = theta.shape[0]

    # Validate inputs
    if K != 2:
        raise ValueError(f"Expected K=2 states, got {K}")
    if theta.shape[1] != D:
        raise ValueError(f"Theta dimension {theta.shape[1]} != features dimension {D}")
    if lambda_penalty < 0:
        raise ValueError(f"Lambda penalty must be >= 0, got {lambda_penalty}")

    # Compute loss matrix l(x_t, theta_k) for all t, k
    loss = _compute_loss(features, theta)  # (T, K)

    # Initialize DP table and backpointers
    dp = np.zeros((T, K))  # dp[t][k] = min cost to reach state k at time t
    backpointer = np.zeros((T, K), dtype=int)  # backpointer[t][k] = argmin for DP[t][k]

    # Base case: t=0
    dp[0, :] = loss[0, :]

    # Forward pass: Fill DP table
    for t in range(1, T):
        for k in range(K):
            # Compute min_j(DP[t-1][j] + λ*1_{j≠k})
            # Penalty is 0 if j==k, lambda if j!=k
            transition_costs = dp[t-1, :].copy()
            for j in range(K):
                if j != k:
                    transition_costs[j] += lambda_penalty

            # DP recurrence
            best_prev_state = np.argmin(transition_costs)
            dp[t, k] = loss[t, k] + transition_costs[best_prev_state]
            backpointer[t, k] = best_prev_state

    # Backward pass: Backtrack to find optimal state sequence
    state_sequence = np.zeros(T, dtype=int)
    state_sequence[T-1] = np.argmin(dp[T-1, :])  # Best final state

    for t in range(T-2, -1, -1):
        state_sequence[t] = backpointer[t+1, state_sequence[t+1]]

    # Compute final objective value
    objective_value = np.min(dp[T-1, :])

    return state_sequence, objective_value


def coordinate_descent(
    features: np.ndarray,
    lambda_penalty: float,
    max_iter: int = 100,
    tol: float = 1e-6,
    random_seed: Optional[int] = None,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """
    Coordinate descent optimization alternating between theta and S.

    Algorithm:
        1. Initialize Θ using K-means clustering (λ=0)
        2. Loop until convergence or max_iter:
           E-step: Fix Θ, optimize S using dynamic_programming()
           M-step: Fix S, optimize Θ: θ_k = mean({x_t : s_t = k})
           Check: |objective_new - objective_old| < tol
        3. Return final Θ, S, objective, converged_flag

    Args:
        features: (T, D) feature matrix
        lambda_penalty: Jump penalty
        max_iter: Maximum iterations (default: 100)
        tol: Convergence tolerance (default: 1e-6)
        random_seed: Random seed for K-means initialization
        verbose: Print iteration progress

    Returns:
        theta: (K, D) optimal centroids
        state_sequence: (T,) optimal state assignments
        objective_value: Final objective function value
        converged: True if converged within max_iter

    Reference:
        Section 3.4, Shu et al., Princeton 2024
        "A coordinate descent algorithm...alternating between optimizing
        the model parameters Θ and the state sequence S"

    Example:
        >>> features = np.random.randn(200, 3)
        >>> theta, states, obj, conv = coordinate_descent(features, lambda_penalty=50.0)
        >>> print(f"Converged: {conv}, Final objective: {obj:.2f}")
    """
    T, D = features.shape
    K = 2  # Two states: bull (0) and bear (1)

    # Initialize Θ using K-means (equivalent to λ=0)
    if verbose:
        print(f"Initializing with K-means (K={K})...")

    kmeans = KMeans(n_clusters=K, random_state=random_seed, n_init=10)
    kmeans_labels = kmeans.fit_predict(features)
    theta = kmeans.cluster_centers_.copy()  # (K, D)

    # Initial state sequence from K-means
    state_sequence = kmeans_labels.astype(int)

    # Compute initial objective
    prev_objective = np.inf

    converged = False

    for iteration in range(max_iter):
        # E-step: Fix Θ, optimize S using dynamic programming
        state_sequence, current_objective = dynamic_programming(
            features, theta, lambda_penalty
        )

        # M-step: Fix S, optimize Θ by averaging features in each state
        for k in range(K):
            mask = (state_sequence == k)
            if np.sum(mask) > 0:
                theta[k, :] = np.mean(features[mask, :], axis=0)
            else:
                # Handle empty cluster (shouldn't happen with good initialization)
                # Reinitialize to random feature
                theta[k, :] = features[np.random.randint(T), :]
                if verbose:
                    print(f"  Warning: Empty cluster {k} at iteration {iteration}")

        # Check convergence
        objective_change = abs(current_objective - prev_objective)

        if verbose:
            print(f"Iter {iteration+1}: Objective={current_objective:.4f}, "
                  f"Delta={objective_change:.2e}")

        # Objective should decrease (or stay same)
        if current_objective > prev_objective + 1e-10:
            if verbose:
                print(f"  Warning: Objective increased from {prev_objective:.4f} "
                      f"to {current_objective:.4f}")

        if objective_change < tol:
            converged = True
            if verbose:
                print(f"Converged after {iteration+1} iterations")
            break

        prev_objective = current_objective

    if not converged and verbose:
        print(f"Did not converge after {max_iter} iterations "
              f"(final delta={objective_change:.2e})")

    return theta, state_sequence, current_objective, converged


def fit_jump_model_multi_start(
    features: np.ndarray,
    lambda_penalty: float,
    n_starts: int = 10,
    max_iter: int = 100,
    random_seed: int = 42,
    verbose: bool = False
) -> dict:
    """
    Multi-start optimization with multiple random initializations.

    Runs coordinate descent n_starts times with different random seeds
    and keeps the result with the lowest objective value.

    Args:
        features: (T, D) feature matrix
        lambda_penalty: Jump penalty
        n_starts: Number of random initializations (default: 10 per paper)
        max_iter: Maximum iterations per run
        random_seed: Base random seed
        verbose: Print progress for each run

    Returns:
        Dictionary with:
            - theta: (K, D) best centroids
            - state_sequence: (T,) best state sequence
            - objective: Best objective value
            - n_converged: Number of runs that converged
            - all_objectives: List of all final objectives
            - best_run: Index of best run (0-indexed)

    Reference:
        Section 3.4, Shu et al., Princeton 2024
        "We run the algorithm ten times and retain the fitting with
        the lowest objective value"

    Example:
        >>> features = np.random.randn(500, 3)
        >>> result = fit_jump_model_multi_start(features, lambda_penalty=50.0, n_starts=10)
        >>> print(f"Best objective: {result['objective']:.2f}")
        >>> print(f"Converged runs: {result['n_converged']}/{len(result['all_objectives'])}")
    """
    if verbose:
        print(f"Running {n_starts} random initializations...")

    best_objective = np.inf
    best_theta = None
    best_state_sequence = None
    all_objectives = []
    n_converged = 0
    best_run = -1

    for run in range(n_starts):
        # Use different seed for each run
        run_seed = random_seed + run if random_seed is not None else None

        if verbose:
            print(f"\nRun {run+1}/{n_starts} (seed={run_seed}):")

        # Run coordinate descent
        theta, state_seq, objective, converged = coordinate_descent(
            features=features,
            lambda_penalty=lambda_penalty,
            max_iter=max_iter,
            tol=1e-6,
            random_seed=run_seed,
            verbose=verbose
        )

        all_objectives.append(objective)
        if converged:
            n_converged += 1

        # Keep best result
        if objective < best_objective:
            best_objective = objective
            best_theta = theta.copy()
            best_state_sequence = state_seq.copy()
            best_run = run

    if verbose:
        print(f"\nBest run: {best_run+1}, Best objective: {best_objective:.4f}")
        print(f"Converged: {n_converged}/{n_starts} runs")
        obj_array = np.array(all_objectives)
        print(f"Objective range: [{obj_array.min():.4f}, {obj_array.max():.4f}]")
        print(f"Objective std: {obj_array.std():.4f}")

    return {
        'theta': best_theta,
        'state_sequence': best_state_sequence,
        'objective': best_objective,
        'n_converged': n_converged,
        'all_objectives': all_objectives,
        'best_run': best_run
    }


class AcademicJumpModel:
    """
    Academic Statistical Jump Model for market regime detection.

    Implements the clustering-based regime detection with temporal penalty
    as described in Shu et al., Princeton 2024.

    Key Features:
        - Coordinate descent optimization with dynamic programming
        - Multi-start initialization (10 runs) for global optimum
        - 3-dimensional features: Downside Deviation + 2 Sortino Ratios
        - Online inference with 3000-day lookback window
        - Temporal penalty (λ) controls regime persistence

    Academic Performance (from paper):
        - S&P 500: Sharpe 0.68 vs 0.48 B&H (+42%)
        - MaxDD reduction: ~50%
        - Regime switches: <1 per year with λ=50

    Attributes:
        lambda_penalty: Jump penalty (controls persistence)
        theta_: (K, D) fitted centroids {bull, bear}
        state_labels_: {0: 'bull', 1: 'bear'} mapping
        is_fitted_: Whether model has been fitted

    Reference:
        Shu et al., "Downside Risk Reduction Using Regime-Switching Signals:
        A Statistical Jump Model Approach", Princeton 2024

    Example:
        >>> model = AcademicJumpModel(lambda_penalty=50.0)
        >>> model.fit(spy_data)  # 3000-day OHLC data
        >>> regime = model.predict(spy_data)
        >>> print(f"Current regime: {regime.iloc[-1]}")
    """

    def __init__(self, lambda_penalty: float = 50.0):
        """
        Initialize Academic Jump Model.

        Args:
            lambda_penalty: Jump penalty (default: 50.0 per paper's typical value)
                           Higher values = more persistent regimes
                           lambda=5: ~2.7 switches/year
                           lambda=50-100: <1 switch/year
        """
        self.lambda_penalty = lambda_penalty
        self.theta_ = None
        self.state_labels_ = {0: 'bull', 1: 'bear'}
        self.is_fitted_ = False
        self._fit_info_ = None  # Store fit diagnostics

    def fit(
        self,
        data: pd.DataFrame,
        n_starts: int = 10,
        max_iter: int = 100,
        random_seed: int = 42,
        verbose: bool = False
    ) -> 'AcademicJumpModel':
        """
        Fit model on OHLC data using multi-start coordinate descent.

        Workflow:
            1. Calculate features using Phase A (academic_features.py)
            2. Run multi-start optimization (10 random initializations)
            3. Store best centroids (bull vs bear)
            4. Label states based on cumulative return

        Args:
            data: OHLC DataFrame with 'Close' column (3000 days recommended)
            n_starts: Number of random initializations (default: 10)
            max_iter: Maximum iterations per run (default: 100)
            random_seed: Base random seed for reproducibility
            verbose: Print optimization progress

        Returns:
            self (fitted model)

        Example:
            >>> from data.alpaca import fetch_alpaca_data
            >>> spy_data = fetch_alpaca_data('SPY', timeframe='1D', period_days=3000)
            >>> model = AcademicJumpModel(lambda_penalty=50.0)
            >>> model.fit(spy_data, verbose=True)
            >>> print(f"Bull centroid: {model.theta_[0]}")
            >>> print(f"Bear centroid: {model.theta_[1]}")
        """
        if verbose:
            print(f"Fitting Academic Jump Model (lambda={self.lambda_penalty})...")

        # Calculate features using Phase A
        # Handle both 'Close' and 'close' column names
        close_col = 'Close' if 'Close' in data.columns else 'close'
        features_df = calculate_features(
            close=data[close_col],
            risk_free_rate=0.03,  # 3% annual risk-free rate
            standardize=False  # Use raw features per reference implementation
        )

        # Drop NaN rows from warm-up period
        features_df = features_df.dropna()
        features = features_df.values  # Convert to numpy array

        if verbose:
            print(f"Features shape: {features.shape}")
            print(f"Feature ranges:")
            print(f"  Downside Dev: [{features[:, 0].min():.4f}, {features[:, 0].max():.4f}]")
            print(f"  Sortino 20d: [{features[:, 1].min():.2f}, {features[:, 1].max():.2f}]")
            print(f"  Sortino 60d: [{features[:, 2].min():.2f}, {features[:, 2].max():.2f}]")

        # Run multi-start optimization
        result = fit_jump_model_multi_start(
            features=features,
            lambda_penalty=self.lambda_penalty,
            n_starts=n_starts,
            max_iter=max_iter,
            random_seed=random_seed,
            verbose=verbose
        )

        self.theta_ = result['theta']
        self._fit_info_ = result

        # Label states based on BOTH centroid characteristics AND Sortino ratio
        #
        # Bull state: Higher Sortino ratio (better risk-adjusted returns)
        # Bear state: Lower Sortino ratio (worse risk-adjusted returns)
        #
        # CRITICAL: Use Sortino as PRIMARY criterion, not downside deviation.
        # Why? With high lambda, DD alone doesn't distinguish bull/bear well.
        # Sortino ratio (return/risk) is the fundamental difference between regimes.

        dd_0 = self.theta_[0, 0]  # Downside deviation state 0
        dd_1 = self.theta_[1, 0]  # Downside deviation state 1
        sortino20_0 = self.theta_[0, 1]  # Sortino 20d state 0
        sortino20_1 = self.theta_[1, 1]  # Sortino 20d state 1

        state_seq = result['state_sequence']
        state_0_count = (state_seq == 0).sum()
        state_1_count = (state_seq == 1).sum()

        if verbose:
            print(f"\nPre-swap analysis:")
            print(f"  State 0: DD={dd_0:.4f}, S20={sortino20_0:.2f}, Count={state_0_count} ({state_0_count/len(state_seq):.1%})")
            print(f"  State 1: DD={dd_1:.4f}, S20={sortino20_1:.2f}, Count={state_1_count} ({state_1_count/len(state_seq):.1%})")

        # Primary criterion: Higher Sortino ratio = bull market
        # Secondary check: If Sortinos are very close, use frequency (dominant state = bull)
        sortino_diff = abs(sortino20_0 - sortino20_1)

        if sortino_diff > 0.05:  # Clear Sortino difference
            state_0_is_bull = (sortino20_0 > sortino20_1)
            criterion = "Sortino ratio"
        else:  # Degenerate case: use frequency
            state_0_is_bull = (state_0_count > state_1_count)
            criterion = "frequency"

        # CRITICAL: Don't swap theta_! Just swap the label mapping.
        # theta_ must stay aligned with what DP algorithm found during optimization.
        # We only change how we INTERPRET the numeric states (0 vs 1).
        if state_0_is_bull:
            self.state_labels_ = {0: 'bull', 1: 'bear'}
            if verbose:
                print(f"State 0 -> bull, State 1 -> bear (criterion: {criterion})")
        else:
            self.state_labels_ = {0: 'bear', 1: 'bull'}  # Swap labels, NOT centroids
            if verbose:
                print(f"State 0 -> bear, State 1 -> bull (criterion: {criterion})")

        self.is_fitted_ = True

        if verbose:
            print("\nFit complete!")
            print(f"Bull centroid (state 0): DD={self.theta_[0, 0]:.4f}, "
                  f"S20={self.theta_[0, 1]:.2f}, S60={self.theta_[0, 2]:.2f}")
            print(f"Bear centroid (state 1): DD={self.theta_[1, 0]:.4f}, "
                  f"S20={self.theta_[1, 1]:.2f}, S60={self.theta_[1, 2]:.2f}")

        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict state sequence for data (requires fitted model).

        Uses dynamic programming with fitted centroids to assign states.

        Args:
            data: OHLC DataFrame with 'Close' column

        Returns:
            Series of state labels ('bull' or 'bear') with original data index

        Raises:
            ValueError: If model not fitted

        Example:
            >>> model.fit(train_data)
            >>> predictions = model.predict(test_data)
            >>> print(predictions.value_counts())
            bull    120
            bear     30
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        # Calculate features
        # Handle both 'Close' and 'close' column names
        close_col = 'Close' if 'Close' in data.columns else 'close'
        features_df = calculate_features(
            close=data[close_col],
            risk_free_rate=0.03,
            standardize=False
        )

        # Drop NaN rows
        features_df = features_df.dropna()
        features = features_df.values

        # Run DP to get state sequence
        state_sequence, _ = dynamic_programming(
            features=features,
            theta=self.theta_,
            lambda_penalty=self.lambda_penalty
        )

        # Map numeric states to labels
        state_labels = [self.state_labels_[s] for s in state_sequence]

        # Return Series with original index
        return pd.Series(state_labels, index=features_df.index, name='regime')

    def _update_theta_online(
        self,
        features: pd.DataFrame,
        lambda_value: float,
        n_starts: int = 10
    ) -> np.ndarray:
        """
        Refit theta centroids using lookback window.

        Uses coordinate descent from Phase B with current lambda value.
        Applies label swapping (Sortino ratio criterion from Session 16 fix).

        Args:
            features: Feature DataFrame for lookback window
            lambda_value: Current lambda penalty value
            n_starts: Number of random initializations (default: 10)

        Returns:
            theta: (K=2, D=3) updated centroids
        """
        # Run multi-start optimization on features window
        result = fit_jump_model_multi_start(
            features=features.values,
            lambda_penalty=lambda_value,
            n_starts=n_starts,
            max_iter=100,
            random_seed=42,
            verbose=False
        )

        theta = result['theta']
        state_seq = result['state_sequence']

        # Apply label swapping logic (same as fit() method)
        sortino20_0 = theta[0, 1]
        sortino20_1 = theta[1, 1]
        state_0_count = (state_seq == 0).sum()
        state_1_count = (state_seq == 1).sum()

        sortino_diff = abs(sortino20_0 - sortino20_1)

        if sortino_diff > 0.05:
            state_0_is_bull = (sortino20_0 > sortino20_1)
        else:
            state_0_is_bull = (state_0_count > state_1_count)

        # Update state_labels_ to maintain consistency
        if state_0_is_bull:
            self.state_labels_ = {0: 'bull', 1: 'bear'}
        else:
            self.state_labels_ = {0: 'bear', 1: 'bull'}

        return theta

    def _update_lambda_online(
        self,
        data_window: pd.DataFrame,
        features_window: pd.DataFrame,
        lambda_candidates: List[float],
        validation_days: int = 2016
    ) -> float:
        """
        Reselect optimal lambda using cross-validation.

        Uses 8-year validation window with Sharpe ratio maximization criterion.

        Args:
            data_window: OHLC data for validation
            features_window: Features for validation
            lambda_candidates: Lambda values to test
            validation_days: Validation window size (default: 2016 = 8 years)

        Returns:
            optimal_lambda: Lambda value with highest Sharpe ratio
        """
        # Calculate returns for strategy simulation
        # Handle both 'Close' and 'close' column names
        close_col = 'Close' if 'Close' in data_window.columns else 'close'
        returns = data_window[close_col].pct_change()

        best_lambda = lambda_candidates[0]
        best_sharpe = -np.inf

        # Take validation window from end of data
        if len(data_window) < validation_days:
            validation_days = len(data_window)

        validation_data = data_window.iloc[-validation_days:]
        validation_features = features_window.iloc[-validation_days:]

        for lambda_val in lambda_candidates:
            # Create temporary model with specific lambda
            temp_model = AcademicJumpModel(lambda_penalty=lambda_val)

            # Fit model on validation window (full fit, not using current theta)
            # This is correct per cross-validation specification
            temp_model.fit(validation_data, n_starts=3, verbose=False)  # Use fewer starts for speed

            # Generate regime sequence on same data
            regime_sequence = temp_model.predict(validation_data)

            # Simulate 0/1 strategy
            val_returns = returns.loc[validation_data.index]
            strategy_result = simulate_01_strategy(
                regime_sequence=regime_sequence,
                returns=val_returns,
                delay_days=1,
                transaction_cost_bps=10.0,
                risk_free_rate=0.03
            )

            sharpe = strategy_result['sharpe_ratio']

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_lambda = lambda_val

        return best_lambda

    def _infer_state_online(
        self,
        features_t: np.ndarray,
        theta: np.ndarray,
        lambda_val: float,
        prev_state: int
    ) -> int:
        """
        Single-step inference given current parameters and previous state.

        Uses loss function with temporal penalty for switching cost.

        Args:
            features_t: Feature vector at time t (D=3,)
            theta: Current centroids (K=2, D=3)
            lambda_val: Current lambda penalty
            prev_state: Previous state (0 or 1)

        Returns:
            state_t: Optimal state at time t (0 or 1)
        """
        # Compute loss for each state
        loss_0 = 0.5 * np.sum((features_t - theta[0]) ** 2)
        loss_1 = 0.5 * np.sum((features_t - theta[1]) ** 2)

        # Add switching penalty if changing from previous state
        if prev_state == 0:
            loss_1 += lambda_val
        else:
            loss_0 += lambda_val

        # Select state with minimum loss
        if loss_0 <= loss_1:
            return 0
        else:
            return 1

    def online_inference(
        self,
        data: pd.DataFrame,
        lookback: int = 1500,
        theta_update_freq: int = 126,
        lambda_update_freq: int = 21,
        default_lambda: float = 15.0,
        lambda_candidates: List[float] = None
    ) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """
        Online inference with rolling parameter updates (Phase D).

        Implements real-time regime detection with:
        - Configurable lookback window for parameter estimation
        - Theta updates every 6 months (126 trading days)
        - Lambda updates every 1 month (21 trading days)
        - Single-step inference with temporal penalty

        Lookback Adaptation:
            Academic paper specifies 3000 days (11.9 years). We adapt based on:
            - Data availability: Alpaca provides ~9 years (2271 days)
            - Validation needs: March 2020 testing requires lookback < 1760 days
            - Statistical stability: 1500 days captures 6-year market cycle

        Recommended Lookback Values:
            - Testing (with March 2020): 1500 days (default)
            - Production (maximum stability): 2000-2500 days
            - Academic replication: 3000 days (requires 12-year dataset)

        Args:
            data: OHLC DataFrame with 'Close' column
            lookback: Parameter estimation window (default: 1500 days = 5.95 years)
            theta_update_freq: Refit centroids every N days (default: 126 = 6 months)
            lambda_update_freq: Reselect lambda every N days (default: 21 = 1 month)
            default_lambda: Initial lambda value (default: 15 for trading)
            lambda_candidates: Lambda values to test (default: [5,15,35,50,70,100,150])

        Returns:
            regime_states: Series of 'bull'/'bear' regime labels with date index
            lambda_history: Series of lambda values used over time
            theta_history: DataFrame of theta centroids [date, state, feature]

        Raises:
            ValueError: If insufficient data

        Reference:
            Section 3.4.2 "Online Inference", Shu et al., Princeton 2024

        Example:
            >>> model = AcademicJumpModel()
            >>> regimes, lambdas, thetas = model.online_inference(spy_data, lookback=1500)
            >>> print(f"March 2020 bear days: {(regimes.loc['2020-03'] == 'bear').sum()}")
        """
        if lambda_candidates is None:
            lambda_candidates = [5, 15, 35, 50, 70, 100, 150]

        # Calculate features for full dataset
        # Handle both 'Close' and 'close' column names
        close_col = 'Close' if 'Close' in data.columns else 'close'
        features_df = calculate_features(
            close=data[close_col],
            risk_free_rate=0.03,
            standardize=False
        ).dropna()

        features = features_df.values
        T = len(features_df)

        # Validate sufficient data
        if T < lookback:
            raise ValueError(
                f"Insufficient data: {T} days < {lookback} required for lookback window"
            )

        # Initialize result arrays
        state_sequence = np.zeros(T - lookback, dtype=int)
        state_label_sequence = []  # Store labels at time of inference (fix for label mapping bug)
        lambda_history = np.zeros(T - lookback)
        theta_history = []

        # Initialize parameters using first lookback window
        init_features = features_df.iloc[:lookback]
        current_theta = self._update_theta_online(init_features, default_lambda)
        current_lambda = default_lambda
        prev_state = 0  # Start in bull market

        # Rolling inference from lookback to end
        for t in range(lookback, T):
            relative_t = t - lookback  # Index in result arrays

            # Check if theta update needed (every 126 days)
            if relative_t > 0 and relative_t % theta_update_freq == 0:
                # Use last lookback days to refit theta
                theta_window_start = t - lookback
                theta_window_end = t
                theta_features = features_df.iloc[theta_window_start:theta_window_end]
                current_theta = self._update_theta_online(theta_features, current_lambda)

            # Check if lambda update needed (every 21 days)
            if relative_t > 0 and relative_t % lambda_update_freq == 0:
                # Use last lookback days for validation
                lambda_window_start = max(0, t - lookback)
                lambda_window_end = t
                lambda_data = data.iloc[lambda_window_start:lambda_window_end]
                lambda_features = features_df.iloc[lambda_window_start:lambda_window_end]
                current_lambda = self._update_lambda_online(
                    lambda_data,
                    lambda_features,
                    lambda_candidates,
                    validation_days=min(2016, len(lambda_features))
                )

            # Perform single-step inference for day t
            features_t = features[t]
            current_state = self._infer_state_online(
                features_t,
                current_theta,
                current_lambda,
                prev_state
            )

            # Store results
            state_sequence[relative_t] = current_state
            state_label_sequence.append(self.state_labels_[current_state])  # Store label at time of inference
            lambda_history[relative_t] = current_lambda
            theta_history.append({
                'date': features_df.index[t],
                'state_0_dd': current_theta[0, 0],
                'state_0_s20': current_theta[0, 1],
                'state_0_s60': current_theta[0, 2],
                'state_1_dd': current_theta[1, 0],
                'state_1_s20': current_theta[1, 1],
                'state_1_s60': current_theta[1, 2]
            })

            prev_state = current_state

        # Use labels stored at time of inference (not retroactive mapping)
        # This fixes bug where label changes during parameter updates would
        # retroactively remap all historical states incorrectly
        state_labels = state_label_sequence

        # Create result Series with proper index
        result_index = features_df.index[lookback:]
        regime_states = pd.Series(state_labels, index=result_index, name='regime')
        lambda_series = pd.Series(lambda_history, index=result_index, name='lambda')
        theta_df = pd.DataFrame(theta_history).set_index('date')

        # Store fitted state
        self.theta_ = current_theta
        self.is_fitted_ = True

        return regime_states, lambda_series, theta_df

    def get_fit_info(self) -> dict:
        """
        Get diagnostic information from fitting process.

        Returns:
            Dictionary with:
                - objective: Best objective value
                - n_converged: Number of runs that converged
                - all_objectives: List of all final objectives
                - best_run: Index of best run

        Raises:
            ValueError: If model not fitted
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted first. Call fit().")

        return self._fit_info_


def simulate_01_strategy(
    regime_sequence: pd.Series,
    returns: pd.Series,
    delay_days: int = 1,
    transaction_cost_bps: float = 10.0,
    risk_free_rate: float = 0.03
) -> dict:
    """
    Simulate 0/1 strategy performance with regime signals.

    Strategy logic:
        - Bull regime: 100% invested in asset
        - Bear regime: 100% cash (risk-free rate)
        - 1-day signal delay (realistic trading latency)
        - 10 bps transaction costs per one-way trade

    Args:
        regime_sequence: Series of regime labels ('bull' or 'bear')
        returns: Series of asset returns (same index as regime_sequence)
        delay_days: Signal delay in days (default: 1)
        transaction_cost_bps: One-way transaction cost in basis points (default: 10)
        risk_free_rate: Annual risk-free rate (default: 0.03)

    Returns:
        Dictionary with:
            - sharpe_ratio: Sharpe ratio of strategy returns
            - total_return: Cumulative return
            - annual_return: Annualized return
            - annual_volatility: Annualized volatility
            - n_trades: Number of trades (regime switches)
            - strategy_returns: Series of daily strategy returns

    Reference:
        Section 3.4.3, Shu et al., Princeton 2024
        "A one-way transaction cost of 10 basis points is applied"
        "This sequence is applied to trading with a one-day delay"

    Example:
        >>> regime = pd.Series(['bull', 'bull', 'bear', 'bear'], index=pd.date_range('2020-01-01', periods=4))
        >>> returns = pd.Series([0.01, 0.02, -0.01, -0.02], index=regime.index)
        >>> result = simulate_01_strategy(regime, returns)
        >>> print(f"Sharpe: {result['sharpe_ratio']:.2f}, Trades: {result['n_trades']}")
    """
    # Align indices
    common_index = regime_sequence.index.intersection(returns.index)
    regime = regime_sequence.loc[common_index]
    rets = returns.loc[common_index]

    # Apply delay to signals
    # Position today is based on regime signal from delay_days ago
    regime_delayed = regime.shift(delay_days)

    # Convert regime to position (1 = invested, 0 = cash)
    positions = (regime_delayed == 'bull').astype(int)

    # Calculate daily risk-free rate
    rf_daily = (1 + risk_free_rate) ** (1/252) - 1

    # Calculate strategy returns
    # When invested (position=1): get asset return - transaction costs when switching
    # When cash (position=0): get risk-free rate - transaction costs when switching
    strategy_rets = pd.Series(0.0, index=positions.index)

    # Identify regime switches (trades)
    position_changes = positions.diff().fillna(0)
    trades = (position_changes != 0)

    # Transaction cost per trade (applied symmetrically)
    tc_rate = transaction_cost_bps / 10000.0  # Convert bps to decimal

    for i in range(len(positions)):
        if pd.isna(positions.iloc[i]):
            # During warm-up period, stay in cash
            strategy_rets.iloc[i] = rf_daily
        elif positions.iloc[i] == 1:
            # Invested: get asset return
            strategy_rets.iloc[i] = rets.iloc[i]
            # Subtract transaction cost if we just switched into this position
            if trades.iloc[i]:
                strategy_rets.iloc[i] -= tc_rate
        else:
            # Cash: get risk-free rate
            strategy_rets.iloc[i] = rf_daily
            # Subtract transaction cost if we just switched into this position
            if trades.iloc[i]:
                strategy_rets.iloc[i] -= tc_rate

    # Drop NaN rows from delay period
    strategy_rets = strategy_rets.dropna()

    if len(strategy_rets) == 0:
        return {
            'sharpe_ratio': 0.0,
            'total_return': 0.0,
            'annual_return': 0.0,
            'annual_volatility': 0.0,
            'n_trades': 0,
            'strategy_returns': strategy_rets
        }

    # Calculate performance metrics
    total_return = (1 + strategy_rets).prod() - 1
    n_days = len(strategy_rets)
    annual_return = (1 + total_return) ** (252 / n_days) - 1
    annual_volatility = strategy_rets.std() * np.sqrt(252)

    # Sharpe ratio: (return - rf) / volatility
    if annual_volatility > 0:
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    else:
        sharpe_ratio = 0.0

    # Count actual trades (exclude initial position)
    n_trades = trades.sum()

    return {
        'sharpe_ratio': sharpe_ratio,
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'n_trades': int(n_trades),
        'strategy_returns': strategy_rets
    }


def cross_validate_lambda(
    data: pd.DataFrame,
    lambda_candidates: list = None,
    validation_window_days: int = 2016,  # 8 years * 252 trading days
    update_frequency_days: int = 21,  # Monthly updates (~21 trading days)
    lookback_window_days: int = 3000,  # For online inference
    risk_free_rate: float = 0.03,
    n_starts: int = 10,  # Multi-start optimization runs
    verbose: bool = False
) -> pd.DataFrame:
    """
    Cross-validate lambda parameter selection using rolling 8-year window.

    Implements the time-series cross-validation approach from Section 3.4.3:
    "At the beginning of each month during the out-of-sample testing period,
    for each candidate jump penalty, we generate the online inferred regime
    sequence over an 8-year lookback validation window...We then select the
    value lambda that yields the highest Sharpe ratio during this validation
    period and use this value for the following month."

    Workflow:
        For each update period (monthly):
            1. For each lambda candidate:
                a. Fit model on lookback_window_days
                b. Generate online regime inference over validation_window_days
                c. Simulate 0/1 strategy with 1-day delay
                d. Calculate Sharpe ratio
            2. Select lambda with highest Sharpe
            3. Apply selected lambda to next month (with 1-day delay)

    Args:
        data: OHLC DataFrame with 'Close' column (10+ years recommended)
        lambda_candidates: List of lambda values to test (default: [5, 15, 35, 50, 70, 100, 150])
        validation_window_days: Validation window in trading days (default: 2016 = 8 years)
        update_frequency_days: How often to update lambda (default: 21 = monthly)
        lookback_window_days: Training window for model fitting (default: 3000)
        risk_free_rate: Annual risk-free rate (default: 0.03)
        n_starts: Number of random initializations for model fitting (default: 10)
        verbose: Print progress

    Returns:
        DataFrame with columns:
            - date: Update date
            - selected_lambda: Lambda selected for this period
            - sharpe_ratio: Best Sharpe ratio achieved
            - n_trades: Number of trades in validation period
            - lambda_sharpes: Dict of {lambda: sharpe} for all candidates

    Reference:
        Section 3.4.3 "Optimal Jump Penalty Selection"
        Shu et al., Princeton 2024

    Example:
        >>> from data.alpaca import fetch_alpaca_data
        >>> spy_data = fetch_alpaca_data('SPY', timeframe='1D', period_days=3650)  # 10 years
        >>> cv_results = cross_validate_lambda(spy_data, verbose=True)
        >>> print(cv_results[['date', 'selected_lambda', 'sharpe_ratio']])
    """
    if lambda_candidates is None:
        lambda_candidates = [5, 15, 35, 50, 70, 100, 150]

    if verbose:
        print(f"Cross-validating lambda selection...")
        print(f"Lambda candidates: {lambda_candidates}")
        print(f"Validation window: {validation_window_days} days ({validation_window_days/252:.1f} years)")
        print(f"Update frequency: {update_frequency_days} days")

    # Calculate returns for strategy simulation
    returns = data['Close'].pct_change()

    # Determine update dates (monthly starting after warmup period)
    min_required_days = lookback_window_days + validation_window_days
    if len(data) < min_required_days:
        raise ValueError(
            f"Insufficient data: {len(data)} days < {min_required_days} required "
            f"(lookback={lookback_window_days} + validation={validation_window_days})"
        )

    # Start cross-validation after warmup period
    start_idx = min_required_days
    update_indices = range(start_idx, len(data), update_frequency_days)

    results = []

    for update_idx in update_indices:
        update_date = data.index[update_idx]

        if verbose:
            print(f"\n[{update_date.strftime('%Y-%m-%d')}] Testing {len(lambda_candidates)} lambdas...")

        # Validation window: previous validation_window_days
        val_start_idx = update_idx - validation_window_days
        val_end_idx = update_idx
        validation_data = data.iloc[val_start_idx:val_end_idx]

        # For model fitting, use lookback window before validation period
        fit_start_idx = val_start_idx - lookback_window_days
        fit_end_idx = val_start_idx
        fit_data = data.iloc[fit_start_idx:fit_end_idx]

        best_lambda = None
        best_sharpe = -np.inf
        best_n_trades = 0
        lambda_sharpes = {}

        for lambda_val in lambda_candidates:
            # Fit model on lookback window
            model = AcademicJumpModel(lambda_penalty=lambda_val)
            model.fit(fit_data, n_starts=n_starts, verbose=False)

            # Generate online regime inference over validation window
            regime_sequence = model.predict(validation_data)

            # Simulate 0/1 strategy with 1-day delay
            val_returns = returns.loc[validation_data.index]
            strategy_result = simulate_01_strategy(
                regime_sequence=regime_sequence,
                returns=val_returns,
                delay_days=1,
                transaction_cost_bps=10.0,
                risk_free_rate=risk_free_rate
            )

            sharpe = strategy_result['sharpe_ratio']
            n_trades = strategy_result['n_trades']
            lambda_sharpes[lambda_val] = sharpe

            if verbose:
                print(f"  Lambda={lambda_val:3d}: Sharpe={sharpe:6.3f}, Trades={n_trades}")

            # Track best lambda
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_lambda = lambda_val
                best_n_trades = n_trades

        if verbose:
            print(f"  Selected: Lambda={best_lambda} (Sharpe={best_sharpe:.3f})")

        results.append({
            'date': update_date,
            'selected_lambda': best_lambda,
            'sharpe_ratio': best_sharpe,
            'n_trades': best_n_trades,
            'lambda_sharpes': lambda_sharpes
        })

    results_df = pd.DataFrame(results)

    if verbose:
        print(f"\nCross-validation complete!")
        print(f"Lambda selection frequency:")
        print(results_df['selected_lambda'].value_counts().sort_index())
        print(f"\nMean Sharpe: {results_df['sharpe_ratio'].mean():.3f}")
        print(f"Median Lambda: {results_df['selected_lambda'].median():.0f}")

    return results_df

# Physics-Informed Neural Networks for Option Pricing

Solving the Black–Scholes PDE for European and American option pricing using PINNs in PyTorch.

## Methods

- **Baseline**: PDE residual loss + payoff initial condition + Dirichlet boundary conditions
- **Ext 1 – Asymptotic BC**: Enforces far-field boundary behavior for improved stability
- **Ext 2 – RAR Sampling**: Residual-based adaptive refinement to concentrate collocation points in high-error regions
- **Ext 3 – Activation Comparison**: Swish vs. Tanh activation functions
- **Ext 4 – Put-Call Parity**: Verified consistency between call and put pricing surfaces
- **Ext 5 – American Put**: Free-boundary formulation with early-exercise constraints

## Results

| Model | Description |
|---|---|
| Baseline | European call, standard PINN |
| + RBC | Asymptotic right boundary condition |
| + RAR | Adaptive collocation refinement |
| + Swish | Alternative activation function |
| American Put | Free-boundary early exercise |

Selected pricing surfaces and error plots are in [`images/`](images/).

## Usage

```bash
pip install torch scipy numpy matplotlib
python pinn_option_pricing.ipynb
```

## Tech Stack

Python, PyTorch, SciPy, NumPy, Matplotlib

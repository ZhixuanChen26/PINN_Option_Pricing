import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import norm

torch.manual_seed(42)
np.random.seed(42)

# Parameters from the paper
Strike  = 50.0
T       = 1.0
sigma   = 0.2
r       = 0.05
t0      = 0.0
S0      = 10.0
SMax    = 10 * S0
delta_S = 10.0
delta_t = 1 / 52
epochs  = 10000
n_col   = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def payoff(s):
    return np.maximum(s - Strike, 0.0)

# Terminal condition at t = T
S1 = np.arange(S0, SMax + 1e-9, delta_S)
u1 = payoff(S1)
t1 = np.full_like(S1, T)

# Lower boundary at S = 0
t2 = np.arange(t0, T, delta_t)
S2 = np.zeros_like(t2)
u2 = np.zeros_like(t2)

S_obs = np.concatenate([S2, S1]).reshape(-1, 1).astype(np.float32)
t_obs = np.concatenate([t2, t1]).reshape(-1, 1).astype(np.float32)
u_obs = np.concatenate([u2, u1]).reshape(-1, 1).astype(np.float32)

S_obs_t = torch.tensor(S_obs, device=device)
t_obs_t = torch.tensor(t_obs, device=device)
u_obs_t = torch.tensor(u_obs, device=device)


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1),
        )

    def forward(self, S, t):
        return self.net(torch.cat([S, t], dim=1))


def pde_loss(model, S, t):
    S = S.requires_grad_(True)
    t = t.requires_grad_(True)
    u = model(S, t)

    u_t  = torch.autograd.grad(u,   t,  grad_outputs=torch.ones_like(u),   create_graph=True)[0]
    u_S  = torch.autograd.grad(u,   S,  grad_outputs=torch.ones_like(u),   create_graph=True)[0]
    u_SS = torch.autograd.grad(u_S, S,  grad_outputs=torch.ones_like(u_S), create_graph=True)[0]

    residual = u_t + 0.5 * sigma**2 * S**2 * u_SS + r * S * u_S - r * u
    return torch.mean(residual ** 2)


def data_loss(model, S_obs, t_obs, u_obs):
    return torch.mean((model(S_obs, t_obs) - u_obs) ** 2)


model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"Training on {device} for {epochs} epochs")

for epoch in range(epochs):
    optimizer.zero_grad()

    S_col = torch.FloatTensor(n_col, 1).uniform_(0, SMax).to(device)
    t_col = torch.FloatTensor(n_col, 1).uniform_(t0, T).to(device)

    loss = pde_loss(model, S_col, t_col) + data_loss(model, S_obs_t, t_obs_t, u_obs_t)
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d}  loss: {loss.item():.5f}")


def bs_call(S, K, tau, r, sigma):
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        price = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    return np.where(S > 0, price, 0.0)


model.eval()
S_test = np.linspace(0, 100, 200).reshape(-1, 1).astype(np.float32)
t_test = np.zeros_like(S_test)  # evaluate at t=0

with torch.no_grad():
    u_pred = model(torch.tensor(S_test, device=device),
                   torch.tensor(t_test, device=device)).cpu().numpy().flatten()

bs_prices = bs_call(S_test.flatten(), Strike, T, r, sigma)
abs_err   = np.abs(u_pred - bs_prices)

print(f"\nMax absolute error:  {abs_err.max():.4f}")
print(f"Mean absolute error: {abs_err.mean():.4f}")
print(f"Relative L2 error:   {np.linalg.norm(u_pred - bs_prices) / np.linalg.norm(bs_prices):.4f}")

# Plot 1: pricing comparison at t=0
plt.figure()
plt.plot(S_test, bs_prices, 'b-',  label='Black-Scholes')
plt.plot(S_test, u_pred,    'r--', label='PINN')
plt.title('European Call at t=0: Black-Scholes vs PINN')
plt.xlabel('Asset price S')
plt.ylabel('Option value V')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('pricing_comparison.png', dpi=150)
plt.show()

# Plot 2: absolute error at t=0
plt.figure()
plt.plot(S_test, abs_err, 'r-')
plt.title('Absolute Error |BS - PINN| at t=0')
plt.xlabel('Asset price S')
plt.ylabel('Absolute error')
plt.grid(True, alpha=0.3)
plt.savefig('absolute_error.png', dpi=150)
plt.show()
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import AdamW
from optim import SM3

# Rosenbrock function
class Rosenbrock:
    def __init__(self):
        self.params = torch.nn.Parameter(torch.tensor([1.5, 1.5], requires_grad=True))
    
    def forward(self):
        x, y = self.params
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    def parameters(self):
        return [self.params]

def train_optimizer(problem, optimizer, optimizer_name, lr, steps=1000):
    logs = {'x': [], 'y': [], 'loss': []}
    for step in range(steps):
        optimizer.zero_grad()
        loss = problem.forward()
        loss.backward()
        optimizer.step()
        
        x, y = problem.params.detach().numpy()
        logs['x'].append(x)
        logs['y'].append(y)
        logs['loss'].append(loss.item())
    return logs

def plot_trajectories(logs, learning_rates):
    plt.figure(figsize=(10, 6))
    for opt_name, lr_data in logs.items():
        for lr, data in lr_data.items():
            plt.plot(data['x'], data['y'], label=f"{opt_name}, lr={lr}")
    
    plt.title("Optimization Trajectories on Rosenbrock Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()

learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
optimizers = {
    "AdamW": AdamW,
    "SM3": SM3
}

logs = {name: {} for name in optimizers}
for name, optimizer_cls in optimizers.items():
    for lr in learning_rates:
        problem = Rosenbrock()
        optimizer = optimizer_cls(problem.parameters(), lr=lr)
        logs[name][lr] = train_optimizer(problem, optimizer, name, lr)

plot_trajectories(logs, learning_rates)

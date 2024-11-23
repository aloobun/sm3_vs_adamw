import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import AdamW
from typing import List, Tuple, Dict
from optim import SM3

class QuadraticProblem:
    #f(x) = x^2
    def __init__(self, init_value: float = 2.0):
        self.x = torch.nn.Parameter(torch.tensor([init_value], dtype=torch.float32))
        
    def forward(self) -> torch.Tensor:
        return self.x ** 2
    
    def parameters(self) -> List[torch.nn.Parameter]:
        return [self.x]

class MetricsTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.logs = {
            'loss': [],
            'param_value': [],
            'grad_norm': [],
            'step_size': [],
            'adaptive_lr': [],
            'second_moment': [],
            'effective_lr': [],
            'param_distance_from_opt': [],
            'oscillation': []
        }
        
        self.prev_param = None
        self.running_mean_grad_sq = 0
        self.step_count = 0
        
    def update(self, problem: QuadraticProblem, loss: torch.Tensor, old_param_value: torch.Tensor, optimizer_name: str):
        param = problem.x.data
        grad = problem.x.grad.data
        self.logs['loss'].append(loss.item())
        self.logs['param_value'].append(param.item())
        self.logs['grad_norm'].append(torch.norm(grad).item())
        step_size = torch.norm(param - old_param_value).item()
        self.logs['step_size'].append(step_size)
        
        self.step_count += 1
        beta2 = 0.999
        self.running_mean_grad_sq = beta2 * self.running_mean_grad_sq + (1 - beta2) * (grad ** 2)
        
        self.logs['adaptive_lr'].append(step_size / (torch.sqrt(self.running_mean_grad_sq) + 1e-8).item())
        self.logs['second_moment'].append(self.running_mean_grad_sq.item())
        self.logs['effective_lr'].append(step_size / torch.norm(grad).item() if torch.norm(grad).item() > 0 else 0)
        
        self.logs['param_distance_from_opt'].append(abs(param.item()))
        if len(self.logs['param_value']) >= 3:
            recent_params = self.logs['param_value'][-3:]
            self.logs['oscillation'].append(np.var(recent_params))
        else:
            self.logs['oscillation'].append(0.0)
        
        self.prev_param = param.clone()
        return self.logs

def train_and_log(
    problem: QuadraticProblem,
    optimizer: torch.optim.Optimizer,
    optimizer_name: str,
    n_steps: int = 100
) -> Dict[str, List[float]]:
    metrics = MetricsTracker()
    
    for i in range(n_steps):
        loss = problem.forward()
        optimizer.zero_grad()
        loss.backward()
        old_param_value = problem.x.data.clone()
        optimizer.step()
        metrics.update(problem, loss, old_param_value, optimizer_name)
    
    return metrics.logs

def plot_comparison(all_logs: Dict[str, Dict[str, List[float]]]):
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    metrics_to_plot = [
        ('loss', 'Loss vs Iterations', 'log'),
        ('adaptive_lr', 'Adaptive Learning Rate', 'log'),
        ('param_distance_from_opt', 'Distance from Optimum', 'log'),
        ('effective_lr', 'Effective Learning Rate', 'log')
    ]
    
    for idx, (metric, title, scale) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        for opt_name, logs in all_logs.items():
            iterations = range(len(logs[metric]))
            ax.plot(iterations, logs[metric], label=f'{opt_name}')
        
        ax.set_title(title)
        if scale == 'log':
            ax.set_yscale('log')
        ax.set_xlabel('Iterations')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    return fig

def run_optimizer_comparison(n_steps: int = 100):
    optimizers = {
        'AdamW': (QuadraticProblem(), AdamW, {'lr': 0.1, 'weight_decay': 0.0}),
        'SM3': (QuadraticProblem(), SM3, {'lr': 0.1})
    }
    
    all_logs = {}
    for opt_name, (problem, opt_class, opt_kwargs) in optimizers.items():
        optimizer = opt_class(problem.parameters(), **opt_kwargs)
        all_logs[opt_name] = train_and_log(problem, optimizer, opt_name, n_steps)
    
    fig = plot_comparison(all_logs)
    return all_logs, fig

logs, fig = run_optimizer_comparison(n_steps=100)

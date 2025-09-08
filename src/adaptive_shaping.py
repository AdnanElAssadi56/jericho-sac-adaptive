"""
Adaptive Reward Shaping Module for SAC in Text-based Games
Author: Adnan El Assadi
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from collections import deque
import math


class AdaptiveShapingScheduler(ABC):
    """Base class for adaptive reward shaping schedulers."""
    
    def __init__(self, initial_alpha=1.0):
        self.initial_alpha = initial_alpha
        self.current_alpha = initial_alpha
        self.step_count = 0
        
    @abstractmethod
    def update(self, **kwargs):
        """Update the shaping coefficient based on training progress."""
        pass
    
    def get_alpha(self):
        """Get current shaping coefficient."""
        return self.current_alpha
    
    def reset(self):
        """Reset scheduler state."""
        self.current_alpha = self.initial_alpha
        self.step_count = 0


class TimeDecayScheduler(AdaptiveShapingScheduler):
    """Time-based decay scheduler for reward shaping."""
    
    def __init__(self, initial_alpha=1.0, decay_type='exponential', 
                 decay_rate=0.001, total_steps=1000000):
        super().__init__(initial_alpha)
        self.decay_type = decay_type
        self.decay_rate = decay_rate
        self.total_steps = total_steps
        
    def update(self, step=None, **kwargs):
        """Update alpha based on time decay."""
        if step is not None:
            self.step_count = step
        else:
            self.step_count += 1
            
        if self.decay_type == 'exponential':
            self.current_alpha = self.initial_alpha * math.exp(-self.decay_rate * self.step_count)
        elif self.decay_type == 'linear':
            progress = min(self.step_count / self.total_steps, 1.0)
            self.current_alpha = self.initial_alpha * (1.0 - progress)
        elif self.decay_type == 'cosine':
            progress = min(self.step_count / self.total_steps, 1.0)
            self.current_alpha = self.initial_alpha * (1 + math.cos(math.pi * progress)) / 2
        
        return self.current_alpha


class SparsityTriggeredScheduler(AdaptiveShapingScheduler):
    """Sparsity-triggered scheduler that boosts shaping during reward-sparse periods."""
    
    def __init__(self, initial_alpha=1.0, sparsity_threshold=50, 
                 boost_factor=2.0, window_size=100):
        super().__init__(initial_alpha)
        self.sparsity_threshold = sparsity_threshold
        self.boost_factor = boost_factor
        self.window_size = window_size
        self.reward_history = deque(maxlen=window_size)
        self.steps_since_reward = 0
        
    def update(self, reward=0, **kwargs):
        """Update alpha based on reward sparsity."""
        self.reward_history.append(reward)
        
        if reward != 0:
            self.steps_since_reward = 0
        else:
            self.steps_since_reward += 1
            
        # Check if we're in a sparse reward period
        if self.steps_since_reward > self.sparsity_threshold:
            self.current_alpha = self.initial_alpha * self.boost_factor
        else:
            # Gradually decay back to initial alpha
            decay_factor = max(0.1, 1.0 - self.steps_since_reward / self.sparsity_threshold)
            self.current_alpha = self.initial_alpha * decay_factor
            
        return self.current_alpha


class UncertaintyInformedScheduler(AdaptiveShapingScheduler):
    """Uncertainty-informed scheduler based on policy entropy and Q-value variance."""
    
    def __init__(self, initial_alpha=1.0, entropy_threshold=0.5, 
                 uncertainty_window=50, min_alpha=0.1, max_alpha=2.0):
        super().__init__(initial_alpha)
        self.entropy_threshold = entropy_threshold
        self.uncertainty_window = uncertainty_window
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.entropy_history = deque(maxlen=uncertainty_window)
        self.q_variance_history = deque(maxlen=uncertainty_window)
        
    def update(self, policy_entropy=None, q_variance=None, **kwargs):
        """Update alpha based on policy uncertainty."""
        if policy_entropy is not None:
            self.entropy_history.append(policy_entropy)
            
        if q_variance is not None:
            self.q_variance_history.append(q_variance)
            
        # Calculate uncertainty metrics
        if len(self.entropy_history) > 0:
            avg_entropy = np.mean(self.entropy_history)
            
            # High entropy = high uncertainty = need more shaping
            if avg_entropy > self.entropy_threshold:
                uncertainty_factor = min(2.0, avg_entropy / self.entropy_threshold)
            else:
                uncertainty_factor = max(0.5, avg_entropy / self.entropy_threshold)
                
            self.current_alpha = np.clip(
                self.initial_alpha * uncertainty_factor,
                self.min_alpha, self.max_alpha
            )
            
        return self.current_alpha


class AdaptiveRewardShaper:
    """Main adaptive reward shaping class that integrates with SAC."""
    
    def __init__(self, scheduler_type='time_decay', scheduler_params=None, 
                 rs_discount=0.99, base_weight=0.1):
        self.rs_discount = rs_discount
        self.base_weight = base_weight
        
        # Initialize scheduler
        if scheduler_params is None:
            scheduler_params = {}
            
        if scheduler_type == 'time_decay':
            self.scheduler = TimeDecayScheduler(**scheduler_params)
        elif scheduler_type == 'sparsity_triggered':
            self.scheduler = SparsityTriggeredScheduler(**scheduler_params)
        elif scheduler_type == 'uncertainty_informed':
            self.scheduler = UncertaintyInformedScheduler(**scheduler_params)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
            
        self.scheduler_type = scheduler_type
        
    def compute_shaped_reward(self, batch, current_V, target_V, step=None, **kwargs):
        """Compute adaptively shaped rewards."""
        # Update scheduler
        if self.scheduler_type == 'time_decay':
            self.scheduler.update(step=step)
        elif self.scheduler_type == 'sparsity_triggered':
            # Use average reward from batch
            avg_reward = np.mean(batch.rew) if hasattr(batch, 'rew') else 0
            self.scheduler.update(reward=avg_reward)
        elif self.scheduler_type == 'uncertainty_informed':
            self.scheduler.update(**kwargs)
            
        # Get current shaping coefficient
        alpha = self.scheduler.get_alpha()
        
        # Compute potential-based shaping with adaptive coefficient
        device = current_V.device if hasattr(current_V, 'device') else 'cpu'
        reward = torch.tensor(batch.rew, dtype=torch.float, device=device)
        
        # Adaptive potential-based shaping
        current_V_shaped = [(1-self.base_weight)*c_v + self.base_weight*(rew + self.rs_discount*t_v) 
                           for rew, c_v, t_v in zip(batch.rew, current_V, target_V)]
        current_V_shaped = torch.stack(current_V_shaped)
        
        # Apply adaptive scaling
        reward_shaping = alpha * (self.rs_discount * target_V - current_V_shaped)
        shaped_rewards = reward_shaping + reward
        
        return shaped_rewards, alpha
    
    def get_scheduler_info(self):
        """Get current scheduler information for logging."""
        return {
            'scheduler_type': self.scheduler_type,
            'current_alpha': self.scheduler.get_alpha(),
            'step_count': self.scheduler.step_count
        }
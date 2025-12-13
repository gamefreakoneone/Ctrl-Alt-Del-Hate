import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set Times New Roman as the default font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

# Data for each alpha
alpha_64 = {
    'training': [0.0909, 0.0762, 0.0703, 0.0497, 0.045],
    'validation': [0.087996, 0.087968, 0.091451, 0.091451, 0.095723]
}

alpha_16 = {
    'training': [0.0897, 0.077, 0.072, 0.0531, 0.0459],
    'validation': [0.086747, 0.086527, 0.089584, 0.096257, 0.102843]
}

alpha_32 = {
    'training': [0.0896, 0.0781, 0.0756, 0.0604, 0.0561],
    'validation': [0.086543, 0.085821, 0.087731, 0.091924, 0.096100]
}

# Epochs (x-axis)
epochs = np.arange(1, 6)

# Create figure with 3 subplots side by side
fig, axes = plt.subplots(1, 3, figsize=(12, 6))

# Alpha values and their data (in order: 16, 32, 64)
alphas = [
    (16, alpha_16),
    (32, alpha_32),
    (64, alpha_64)
]

# Colors for training and validation
training_color = '#b17c82'
validation_color = '#6b8e5a'

# Plot each alpha
for idx, (alpha, data) in enumerate(alphas):
    ax = axes[idx]
    
    # Plot training and validation loss
    ax.plot(epochs, data['training'], marker='o', linewidth=3, markersize=8,
            label='Training Loss', color=training_color)
    ax.plot(epochs, data['validation'], marker='s', linewidth=3, markersize=8,
            label='Validation Loss', color=validation_color)
    
    # Customize each subplot
    ax.set_xlabel('Epoch', fontsize=20, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=20, fontweight='bold')
    ax.set_title(f'Alpha = {alpha}', fontsize=24, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=16, loc='lower left', frameon=True, shadow=True)
    ax.tick_params(axis='both', labelsize=16)
    
    # Set x-axis to show only integer epochs
    ax.set_xticks(epochs)
    ax.set_xticklabels(epochs)

# Add overall title
fig.suptitle('Training vs Validation Loss', fontsize=28, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('overfitting_graphs.png', dpi=300, bbox_inches='tight')

print("Overfitting line graphs created successfully!")
print(f"Saved as 'overfitting_graphs.png'")


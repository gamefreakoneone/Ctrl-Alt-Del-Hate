import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set Times New Roman as the default font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

# Load data from summary_stats.json
with open('summary_stats.json', 'r') as f:
    data = json.load(f)

targets = data['targets']

# Extract target names and their "true" counts
target_counts = []
for target_name, counts in targets.items():
    true_count = counts.get('true', 0)
    # Clean up the target name for display
    display_name = target_name.replace('target_', '').replace('_', ' ').title()
    
    # Custom name mappings
    name_fixes = {
        'Disability Physical': 'Physical Disability',
        'Disability Hearing Impaired': 'Hearing Impaired',
        'Disability Visually Impaired': 'Visually Impaired',
        'Disability Other': 'Other Disability',
        'Origin Specific Country': 'Origin Country'
    }
    display_name = name_fixes.get(display_name, display_name)
    
    target_counts.append((display_name, true_count))

# Sort by count
target_counts.sort(key=lambda x: x[1], reverse=True)

# Get top 5 and bottom 5
top_5 = target_counts[:5]
bottom_5 = target_counts[-5:]

# Combine them
selected_targets = top_5 + bottom_5
names = [name for name, count in selected_targets]
counts = [count for name, count in selected_targets]

# Create color scheme - darker for top 5, darker green for bottom 5
colors = ['#b17c82'] * 5 + ['#6b8e5a'] * 5

# Create the bar chart (square shape)
fig, ax = plt.subplots(figsize=(12, 10))

# Create bars
x_pos = np.arange(len(names))
bars = ax.bar(x_pos, counts, width=0.7, color=colors, edgecolor='black', linewidth=1.5)

# Add count labels on each bar
for i, (bar, count) in enumerate(zip(bars, counts)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 100,
           f'{count:,}',
           ha='center', va='bottom', 
           fontsize=16, fontweight='bold')

# Customize the chart
ax.set_xlabel('Target Categories', fontsize=24, fontweight='bold')
ax.set_ylabel('Count (True)', fontsize=24, fontweight='bold')
ax.set_title('Top 5 and Bottom 5 Target Categories by Count', fontsize=28, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=18)
ax.tick_params(axis='y', labelsize=20)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add commas to y-axis tick labels for better readability
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

# Add a horizontal line to separate top 5 and bottom 5
ax.axvline(x=4.5, color='gray', linestyle='--', linewidth=2, alpha=0.5)

# Add text labels for sections
ax.text(2, max(counts) * 0.95, 'Top 5', ha='center', fontsize=20, 
        fontweight='bold', color='#b17c82', 
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#b17c82', linewidth=2))
ax.text(7, max(counts) * 0.95, 'Bottom 5', ha='center', fontsize=20, 
        fontweight='bold', color='#6b8e5a',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#6b8e5a', linewidth=2))

plt.tight_layout()
plt.savefig('targets_bargraph.png', dpi=300, bbox_inches='tight')

print("Target bar graph created successfully!")
print(f"Saved as 'targets_bargraph.png'")
print(f"\nTop 5 targets:")
for name, count in top_5:
    print(f"  {name}: {count:,}")
print(f"\nBottom 5 targets:")
for name, count in bottom_5:
    print(f"  {name}: {count:,}")


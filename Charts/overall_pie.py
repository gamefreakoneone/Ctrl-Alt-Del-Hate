import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set Times New Roman as the default font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 28

# Load data from summary_stats.json
with open('summary_stats.json', 'r') as f:
    data = json.load(f)

overall = data['overall']

# Extract labels and counts
labels = list(overall.keys())
counts = list(overall.values())
total = sum(counts)

# Calculate percentages
percentages = [(count / total) * 100 for count in counts]

# Capitalize labels for display
display_labels = [label.capitalize() for label in labels]

# Create color scheme
colors = ['#a4bd84', '#fdf1a8', '#b17c82']  # Green for supportive, yellow for neutral, red for hateful

# Create the pie chart
fig, ax = plt.subplots(figsize=(12, 12))

# Custom function to display percentage and count
def make_autopct(counts):
    def my_autopct(pct):
        total = sum(counts)
        val = int(round(pct*total/100.0))
        return f'{pct:.1f}%\n({val:,})'
    return my_autopct

# Create pie chart with custom styling
wedges, texts, autotexts = ax.pie(counts, 
                                    labels=display_labels,
                                    colors=colors,
                                    autopct=make_autopct(counts),
                                    startangle=90,
                                    textprops={'fontsize': 32, 'fontweight': 'bold'},
                                    pctdistance=0.75,
                                    wedgeprops={'edgecolor': 'white', 'linewidth': 3})

# Customize the percentage and count text
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontsize(28)
    autotext.set_fontweight('bold')

# Add title
ax.set_title('Overall Label Distribution', fontsize=36, fontweight='bold', pad=5)

plt.tight_layout()
plt.savefig('overall_pie.png', dpi=300, bbox_inches='tight')

print("Overall pie chart created successfully!")
print(f"Saved as 'overall_pie.png'")
print(f"\nLabel distribution:")
for label, count, pct in zip(display_labels, counts, percentages):
    print(f"  {label}: {count:,} ({pct:.1f}%)")
print(f"  Total: {total:,}")


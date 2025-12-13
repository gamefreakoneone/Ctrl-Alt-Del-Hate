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

facets = data['facets']

# Prepare data for stacked bar chart
features = list(facets.keys())

# Switch genocide and attack_defend positions
if 'genocide' in features and 'attack_defend' in features:
    genocide_idx = features.index('genocide')
    attack_defend_idx = features.index('attack_defend')
    features[genocide_idx], features[attack_defend_idx] = features[attack_defend_idx], features[genocide_idx]

rating_values = ['0', '1', '2', '3', '4']

# Create a dictionary to hold counts for each rating across features
rating_counts = {rating: [] for rating in rating_values}

# Populate counts for each feature and rating
for feature in features:
    for rating in rating_values:
        count = facets[feature].get(rating, 0)
        rating_counts[rating].append(count)

# Create the stacked bar chart (larger for poster)
fig, ax = plt.subplots(figsize=(14, 10))

# Set up the bar positions
x_pos = np.arange(len(features))
bar_width = 0.6

# Colors for each rating level (from light to dark for better visual progression)
colors = ['#a4bd84', '#d3dc92', '#fdf1a8', '#fcab92', '#b17c82']

# Create the stacked bars and add count labels
rating_labels = ['0 = Absent', '1 = Mild', '2 = Clear', '3 = Severe', '4 = Extreme']
bottom = np.zeros(len(features))
for i, rating in enumerate(rating_values):
    bars = ax.bar(x_pos, rating_counts[rating], bar_width, 
                   bottom=bottom, label=rating_labels[i], color=colors[i])
    
    # Add count labels on each bar segment
    for j, (bar, count) in enumerate(zip(bars, rating_counts[rating])):
        if count > 0:  # Only show label if count is greater than 0
            feature_name = features[j]
            # Skip labels for specific feature-rating combinations
            if (feature_name == 'genocide' and rating in ['3', '4']) or \
               (feature_name == 'status' and rating == '0'):
                continue
            
            height = bar.get_height()
            # Position label in the middle of each segment
            y_position = bottom[j] + height / 2
            
            # Special adjustment: print hatespeech rating 0 higher (at top of segment)
            if feature_name == 'hatespeech' and rating == '0':
                y_position = bottom[j] + height * 0.85  # Position near top of segment
            
            ax.text(bar.get_x() + bar.get_width() / 2, y_position, 
                   f'{count:,}',
                   ha='center', va='center', 
                   fontsize=14, fontweight='bold',
                   color='white' if i in [0, 4] else 'black')  # White text for dark colors
    
    bottom += np.array(rating_counts[rating])

# Customize the chart
ax.set_xlabel('Features', fontsize=24, fontweight='bold')
ax.set_ylabel('Counts', fontsize=24, fontweight='bold')
ax.set_title('Facet Feature Counts', fontsize=28, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(features, rotation=45, ha='right', fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.legend(title='Rating', loc='lower right', fontsize=18, title_fontsize=20)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add commas to y-axis tick labels for better readability
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

plt.tight_layout()
plt.savefig('facets_bargraph.png', dpi=300, bbox_inches='tight')

print("Stacked bar graph created successfully!")
print(f"Saved as 'facets_bargraph.png'")


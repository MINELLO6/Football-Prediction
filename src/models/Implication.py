import matplotlib.pyplot as plt

# Sample data for the bar chart
categories = ['SDC', 'DDC', 'SBM', 'DBM', 'DWM', 'MLR']
values = [0.38, 0.21, 0.25, 0.236, 0.8602, 0.3836]

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 7))

# Create the bar chart
ax.bar(categories, values, color='skyblue')

# Increase the font size for axis labels
label_fontsize = 12  # Increased font size for axis labels
ax.set_xlabel('Models', fontsize=label_fontsize)
ax.set_ylabel('Mean RPS', fontsize=label_fontsize)
ax.set_title('Model Comparison')

# Add grid lines
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Define footnote font size (slightly smaller than axis labels)
footnote_font_size = label_fontsize * 0.85

# Create the footnote text with the correct model descriptions
footnote = "SDC: Static Dixon & Coles  DDC: Dynamic Dixon & Coles  SBM: Static Bayesian Model DBM: Dynamic Bayesian Model  DWM: Discrete Weibull Model with \nGumbel copula  MLR: Multinomial Logistic Regression"

# Position the text closer to the bottom of the chart
fig.text(0.05, 0.04, footnote, ha='left', fontsize=footnote_font_size)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Reduced bottom margin to move footnote closer

plt.show()
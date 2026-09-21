import matplotlib.pyplot as plt

# Replace these values with the actual microsecond outputs from your terminal
labels = ['cpu no branch', 'cpu branch', 'gpu no branch', 'gpu branch']
times = [414, 234, 38599, 23]

plt.bar(labels, times, color=['#1f77b4', '#1f77b4', '#2ca02c', '#2ca02c'])
plt.ylabel('Execution Time (us)')
plt.title('CPU vs GPU Performance (1,000,000 Elements)')
plt.tight_layout()
plt.savefig('performance_comparison.png')
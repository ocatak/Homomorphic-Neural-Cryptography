import pandas as pd
import matplotlib.pyplot as plt

# Initialize a figure
plt.figure(figsize=(10, 6))
plt.rc('font', size=12)  # controls default text size
plt.rc('axes', titlesize=16)  # fontsize of the axes title
plt.rc('axes', labelsize=25)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=25)  # fontsize of the tick labels
plt.rc('ytick', labelsize=25)  # fontsize of the tick labels
plt.rc('legend', fontsize=25)  # fontsize of the legend
plt.ylim(0, 10)

# x is the same for all datasets
x = list(range(0, 14599))

df = pd.read_csv(f'dataset/ma-rate-0.3-curve-secp224r1-extra-out.csv')

# ABloss
y_ab = df['ABloss']

y_bob = df['Bobloss']

y_eve = df['Eveloss']

plt.plot(x, y_ab, color='blue', linewidth=1, label='AB')
plt.plot(x, y_bob, color='green', linewidth=1,
         label='Bob')
plt.plot(x, y_eve, color='orange', linewidth=1,
         label='Eve')

plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Customization and saving the figure
plt.xlabel('Iterations') 
plt.ylabel('Loss') 
plt.legend()
plt.savefig(f"pdf-figures/training_loss.pdf", bbox_inches='tight')

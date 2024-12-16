import matplotlib.pyplot as plt
import numpy as np

conditions = ['Marche Normale (nm)', 'Sac (bg)', 'Manteau (cl)']
subjects = ['Sujet 1', 'Sujet 2', 'Sujet 3', 'Sujet 4']

performance_data = np.array([
    [0.94, 0.89, 0.83],  # Sujet 1
    [0.94, 0.94, 0.82],  # Sujet 2
    [0.94, 0.84, 0.84],  # Sujet 3
    [0.88, 0.89, 0.82],  # Sujet 4
])

angles = ['0°', '90°', '144°', '180°']
angle_performance = [0.86, 0.94, 0.82, 0.94]

plt.figure(figsize=(12, 12))

plt.subplot(2, 1, 1)
ax1 = plt.gca()
x = np.arange(len(conditions))
width = 0.18  

colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f']

for i in range(len(subjects)):
    bars = ax1.bar(x + i*width - width*1.5, performance_data[i], width, 
                   label=subjects[i], color=colors[i], alpha=0.8)

plt.ylabel('Taux de Confiance')
plt.title('Performance par Condition et Sujet', pad=20)
plt.xticks(x, conditions)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.2)
plt.ylim(0.6, 1.0)

for i in range(len(subjects)):
    for j in range(len(conditions)):
        plt.text(x[j] + i*width - width*1.5, performance_data[i][j], 
                f'{performance_data[i][j]:.2f}', 
                ha='center', va='bottom', fontsize=8)

plt.subplot(2, 1, 2)
ax2 = plt.gca()
bars = ax2.bar(angles, angle_performance, width=0.5, color='#3498db', alpha=0.8)
plt.ylabel('Taux de Confiance Moyen')
plt.title('Performance par Angle de Vue', pad=20)
plt.ylim(0.6, 1.0)
plt.grid(True, alpha=0.2)

# Add value labels on the bars
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom')

# Adjust layout
plt.suptitle('Analyse des Performances de Reconnaissance de la Démarche', 
            fontsize=14, y=0.98)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

plt.figure(figsize=(10, 6))
heatmap = plt.imshow(performance_data, cmap='RdYlGn', aspect='auto')
plt.colorbar(heatmap, label='Taux de Confiance')
plt.title('Heatmap des Performances par Sujet et Condition')
plt.xticks(np.arange(len(conditions)), conditions, rotation=45)
plt.yticks(np.arange(len(subjects)), subjects)

for i in range(len(subjects)):
    for j in range(len(conditions)):
        plt.text(j, i, f'{performance_data[i][j]:.2f}', 
                ha='center', va='center', color='black')

plt.tight_layout()
plt.show()
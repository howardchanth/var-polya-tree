import numpy as np
from matplotlib import pyplot as plt

# https://arxiv.org/pdf/2002.02798v3
#https://proceedings.mlr.press/v80/oliva18a/oliva18a.pdf

# #Mnist
# plt.figure(figsize=(10,8))
#
# results = [2.02, 1.41, 1.52, 1.19, 0.97, 1.43, 0.94, 6.42, 4.64]
# #methods = [
# #           'Real NVP','MADE MoG','MAF MoG', 'TAN', 'RHODE', 'VPT (l=2)', 'VPT (l=4)', 'NICE (Gaussian)', 'NICE (Logistic)']
# colors = ['lightgrey', 'lightgrey','lightgrey',
#           'lightgrey','lightgrey', 'grey', 'grey', 'black', 'black']
# title = 'Likelihood Estimation on MNIST'
#
# width = 5
# for i in range(len(results)):
#     bar = plt.barh(i + width*i, results[i], width, color=colors[i])
#
# plt.yticks([])
# plt.xticks(fontsize=25)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
#
# # Adding labels and title
# #plt.ylabel('Categories')
# #plt.xlabel('Bits Per Dim')
# #plt.title(title)
#
# # Show the plot
# plt.tight_layout()  # Adjust layout
# plt.savefig('mnist.png', dpi=300)


plt.figure(figsize=(10,8))

results = [2.83, 2.65, 2.55, 3.38, 3.98, 4.37, 5.93, 4.54, 2.53, 2.06, 4.27, 4.18,]
#methods = ['ScoreFlow', 'VDM', 'MuLAN', 'RHODE', TAN','MAF MoG', 'MADE MoG','Real NVP','VPT (l=2)',
# 'VPT (l=4)', 'NICE (Gaussian)', 'NICE (Logistic)']
colors = ['lightgrey', 'lightgrey','lightgrey','lightgrey','lightgrey','lightgrey',
          'lightgrey','lightgrey', 'grey', 'grey', 'black', 'black']
title = 'Likelihood Estimation on CIFAR-10'

width = 5
for i in range(len(results)):
    bar = plt.barh(i + width*i, results[i], width, color=colors[i])

plt.yticks([])
plt.xticks(fontsize=25)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Adding labels and title
#plt.ylabel('Categories')
#plt.xlabel('Bits Per Dim')
#plt.title(title)

# Show the plot
plt.tight_layout()  # Adjust layout
plt.savefig('cifar.png', dpi=300)

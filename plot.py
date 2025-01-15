import matplotlib.pyplot as plt

depths = [2, 3, 4, 5, 6]

# training accuracy at each depth:
mean_acc = [1.0, 0.71828125, 0.9114882813, 0.04561328125, 0.02914]
max_acc  = [1.0, 1.0, 1.0,  0.27, 0.2046023438]
sum_acc  = [1.0, 1.0, 0.68, 0.3491, 0.1254277]
max_sum_acc = [1.0, 1.0, 1.0, .3176, .2045]

# epochs at each depth (how many epochs it took):
mean_epoch = [20000, 50000, 50000, 4600,  3500]
max_epoch  = [2000,  500,   5700,  38400, 30300]
sum_epoch  = [19000, 11000, 50000, 50000, 50000]
max_sum_epoch = [200, 600, 4200, 50000, 22900]

#PLOT 1: Accuracy vs. Depth
plt.figure(figsize=(6, 4))
plt.plot(depths, mean_acc, marker='o', label='Mean Aggregator')
plt.plot(depths, max_acc,  marker='s', label='Max Aggregator')
plt.plot(depths, sum_acc,  marker='^', label='Sum Aggregator')

plt.xticks(depths)  
plt.xlabel('Depth (r)')
plt.ylabel('Training Accuracy')
plt.title('Training Accuracy vs. Depth')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('training_accuracy_vs_depth.png')   
plt.show()

#PLOT 2: Epochs vs. Depth
plt.figure(figsize=(6, 4))
plt.plot(depths, mean_epoch, marker='o', label='Mean Aggregator')
plt.plot(depths, max_epoch,  marker='s', label='Max Aggregator')
plt.plot(depths, sum_epoch,  marker='^', label='Sum Aggregator')

plt.xticks(depths)   
plt.xlabel('Depth (r)')
plt.ylabel('Epochs')
plt.title('Epochs vs. Depth')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('epochs_vs_depth.png')   
plt.show()

#PLOT 3: Accuracy vs. Depth (MAXSUM)
plt.figure(figsize=(6, 4))
plt.plot(depths, max_sum_acc, marker='o', label='MaxSum Aggregator')
plt.plot(depths, max_acc,  marker='s', label='Max Aggregator')
plt.plot(depths, sum_acc,  marker='^', label='Sum Aggregator')

plt.xticks(depths)   
plt.xlabel('Depth (r)')
plt.ylabel('Training Accuracy')
plt.title('Training Accuracy vs. Depth')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('training_accuracy_vs_depth_max_sum.png')   
plt.show()

#PLOT 4: Epochs vs. Depth (MAXSUM)
plt.figure(figsize=(6, 4))
plt.plot(depths, max_sum_epoch, marker='o', label='MaxSum Aggregator')
plt.plot(depths, max_epoch,  marker='s', label='Max Aggregator')
plt.plot(depths, sum_epoch,  marker='^', label='Sum Aggregator')

plt.xticks(depths)   
plt.xlabel('Depth (r)')
plt.ylabel('Epochs')
plt.title('Epochs vs. Depth')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('epochs_vs_depth_max_sum.png')   
plt.show()
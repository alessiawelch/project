import matplotlib.pyplot as plt

#-------------------------------------------------
# Data from your table (adjust as needed):
# model   depth   training rate (accuracy)   epoch
# mean    2       1.0                        20000
# mean    3       0.71828125                 50000
# mean    4       0.9114882813               50000
# mean    5       0.04561328125              4600
# mean    6       0.02914                    3500
# max     2       1.0                        2000
# max     3       1.0                        500
# max     4       1.0                        5700
# max     5       0.27                       38400
# max     6       0.2046023438               30300
# sum     2       1.0                        19000
# sum     3       1.0                        11000
# sum     4       0.68                       50000
# sum     5       0.3491                     50000
# sum     6       0.1254277                  50000
#-------------------------------------------------

# Depth values:
depths = [2, 3, 4, 5, 6]

# Training accuracy at each depth:
mean_acc = [1.0, 0.71828125, 0.9114882813, 0.04561328125, 0.02914]
max_acc  = [1.0, 1.0,        1.0,          0.27,           0.2046023438]
sum_acc  = [1.0, 1.0,        0.68,         0.3491,         0.1254277]

# Epochs at each depth (how many epochs it took / was run):
mean_epoch = [20000, 50000, 50000, 4600,  3500]
max_epoch  = [2000,  500,   5700,  38400, 30300]
sum_epoch  = [19000, 11000, 50000, 50000, 50000]

#------------------- PLOT 1: Accuracy vs. Depth ---------------------#
plt.figure(figsize=(6, 4))
plt.plot(depths, mean_acc, marker='o', label='Mean Aggregator')
plt.plot(depths, max_acc,  marker='s', label='Max Aggregator')
plt.plot(depths, sum_acc,  marker='^', label='Sum Aggregator')

plt.xlabel('Depth (r)')
plt.ylabel('Training Accuracy')
plt.title('Training Accuracy vs. Depth')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#------------------- PLOT 2: Epochs vs. Depth -----------------------#
plt.figure(figsize=(6, 4))
plt.plot(depths, mean_epoch, marker='o', label='Mean Aggregator')
plt.plot(depths, max_epoch,  marker='s', label='Max Aggregator')
plt.plot(depths, sum_epoch,  marker='^', label='Sum Aggregator')

plt.xlabel('Depth (r)')
plt.ylabel('Epochs')
plt.title('Epochs vs. Depth')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

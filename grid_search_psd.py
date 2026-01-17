import matplotlib.pyplot as plt
import numpy as np
import scipy
import pickle
import os


#with open("dump_training_physics_smooth_16_stftx.pkl","rb") as f:
with open("psd_report.pkl", "rb") as f:
    data = pickle.load(f)

all_loss, all_pos, all_gt = data[0], data[1], data[2]

all_loss = np.array(all_loss)
all_pos = np.array(all_pos)
# print(len(all_loss))
# print(len(all_pos))

print(all_loss.shape)
print(all_pos.shape)
print(all_gt[0].shape)
# print(feedforward_loc.shape)
# ff_loc = feedforward_loc.cpu().numpy()
# assert False
# assert False
sum=0
sum2=0
sum3=0
for i in range(len(all_loss)):
    print("Testing trial {}".format(i))
    losses = all_loss[i]
    positions = all_pos[i][:, 0]
    gt_pos = all_gt[i].cpu().numpy()
    # loc=all_pos[i]
    #print(losses.shape)
    best = np.argmin(losses)
    #print(best)
    print(all_pos[i][best])
    print(gt_pos[:])
    sum=sum+np.sqrt(np.square(all_pos[i][best][0][0]-gt_pos[:][0][0])+np.square(all_pos[i][best][0][1]-gt_pos[:][0][1]))
    if(np.sqrt(np.square(all_pos[i][best][0][0]-gt_pos[:][0][0])+np.square(all_pos[i][best][0][1]-gt_pos[:][0][1])))<2:
        sum2+=1
    if(np.sqrt(np.square(all_pos[i][best][0][0]-gt_pos[:][0][0])+np.square(all_pos[i][best][0][1]-gt_pos[:][0][1])))<5:
        sum3+=1
    # plt.scatter(positions[:, 0], positions[:, 1], c=losses)
    # plt.scatter(gt_pos[:, 0], gt_pos[:, 1], marker="v", c="red")
    # plt.scatter(all_pos[i][best][0][0], all_pos[i][best][0][1], c="green", marker="v")
    # # plt.scatter(ff_loc[0,0], ff_loc[0,1],marker="v", c="green")
    # plt.axis("equal")
    # plt.show()

sum=sum/len(all_loss)
print(sum)
#3.1539771369562066
print(sum2)#319/1015
print(sum3)


# 3.02
# 378/1015
# 847/1015
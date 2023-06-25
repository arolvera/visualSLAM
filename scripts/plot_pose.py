import os
import matplotlib.pyplot as plt

def main():
    pose_predicted_x = []
    pose_predicted_y = []
    pose_gt_x = []
    pose_gt_y = []
    f_predicted = open(os.getcwd() + "/build/pose_predicted_2.txt")
    f_gt = open(os.getcwd() + "/KITTI_sequence_2/poses.txt")
    for line in f_gt:
        line = line.split(" ") 
        pose_gt_x.append(float(line[3]))
        pose_gt_y.append(float(line[11]))
    
    for line in f_predicted:
        line = line.split(" ") 
        pose_predicted_x.append(float(line[0]))
        pose_predicted_y.append(float(line[2]))

    plt.plot(pose_gt_x, pose_gt_y)
    plt.plot(pose_predicted_x, pose_predicted_y)
    plt.xlabel("X axis")
    plt.ylabel("Z axis")
    plt.title("VO Pose Ground Truth vs Predicted")
    plt.savefig("pose2.png")
    plt.show()
    
    f_predicted.close()
    f_gt.close()

if __name__ == "__main__":
    main();
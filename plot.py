from matplotlib import pyplot as plt

file1 = open('./driving_dataset/data.txt', 'r') 
Lines = file1.readlines() 

angles = []
frames = []

frame_no = 0

# Strips the newline character 
for line in Lines: 
    l = line.split(" ")
    angle = l[1]
    angles.append(float(angle))
    frames.append(int(frame_no))
    frame_no+=1
    # if(frame_no>1000):
    # 	break;

plt.plot(frames,angles)

plt.title('Epic Info')
plt.ylabel('Steering Angle')
plt.xlabel('Frame Number')

plt.show()
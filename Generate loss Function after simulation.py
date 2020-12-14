from matplotlib import pyplot as plt
import numpy as np
import os 
import pickle as pk


def make_graphs(board_size):
    files = os.listdir("C:\\Users\\kriti\\Desktop\\Warzone-main\\warzone4x4-simulation-run\\")   
    x_all_scores = []
    y_all_scores = []
    time_taken = []
    all_files = []
    for i in files:
        if("Warzone_board_"+str(board_size)+"_iterations" in i):
            all_files.append(i)
    for i in all_files:
        with open(path_2+i,"rb") as f: 
            (x, y, z) = pk.load(f)
            x_all_scores.extend(x)
            y_all_scores.extend(y)
            time_taken.extend(z)
    
    wins_x,wins_y= 0,0
    for i in range(len(x_all_scores)):
      if(x_all_scores[i]>y_all_scores[i]):
        wins_x+= 1
      else:
        wins_y+= 1
    
    print("Wins x: ",wins_x, " Wins y: ",wins_y )

    # print(x_all_scores,y_all_scores,time_taken)
    print_graph(board_size,x_all_scores,y_all_scores,time_taken)

def display_graph(board_size,x_axis,y_axis,x_ticks=np.arange(0,101,10),x_label="Games",y_label="Average Score at the end of game"):
    n = board_size

    plt.figure(figsize=(20,10))
    plt.xticks(x_ticks,rotation=90)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_axis, y_axis, '-',color='green')
    plt.title(" Board size "+str(n)+"*"+str(n))
    plt.show()


def print_graph(board_size,x_all_scores,y_all_scores,time_taken):
    n = board_size
    y_axis = np.array(x_all_scores)
    y_axis_2 = np.array(y_all_scores)


    min_a = np.min(y_axis_2)
    max_a = np.max(y_axis_2)
    print(np.max(y_axis),np.min(y_axis))
    print(min_a,max_a)
    x_axis = np.arange(1,len(x_all_scores)+1)

    plt.figure(figsize=(20,10))
    plt.xticks(np.arange(0,101,10),rotation=90)
    plt.xlabel("Games")
    plt.ylabel("Average Score at the end of game")
    plt.plot(x_axis, y_axis, '-')
    # plt.plot(x_axis, y_axis_2, '-')
    plt.title(" Board size "+str(n)+"*"+str(n))
    plt.show()

    # display_graph(n,np.arange(1,len(x_all_scores)+1),time_taken,y_label="Average time taken to complete the game")

board_size = 4
#path = "D:\\AI_Project\\kchugh\\"
path_2 = "C:\\Users\\kriti\\Desktop\\Warzone-main\\warzone4x4-simulation-run\\"
make_graphs(4)

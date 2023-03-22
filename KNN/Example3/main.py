# From https://www.youtube.com/watch?v=EzOMOyOd5JQ
import numpy as np
import pandas as pd
import time 

def distance_ecu(x_train, x_test_point):
    distances = []
    for row in range(len(x_train)):
        current_train_point = x_train[row]
        current_distance = 0

        for col in range(len(current_train_point)):
            current_distance += (current_train_point[col] - x_test_point[col])**2
        
        current_distance = np.sqrt(current_distance)

        distances.append(current_distance)
        print("Cur dis: " + str(current_distance) + " Distances: " + str(distances) )
        time.sleep(200)

    distances = pd.DataFrame(data=distances, columns=['dist'])
    return distances


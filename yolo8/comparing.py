import torch
import torch.nn.functional as F
from yolo8.forward import forward_function
from yolo8.backward import backward_function
from ultralytics import YOLO
import cv2 
import os 
import numpy as np 
import torch
import time 

def dl_model(shelf,model,forward_tensor,fruit_class,forward_PB_review):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(device)

    start_time_model = time.time()

   # forward_tensor = forward_function(shelf,model)
    backward_tensor = backward_function(shelf,model,fruit_class,forward_PB_review)
    if shelf == 4 :
        print(f"Forward Tensor : {forward_tensor}")
        print(f"Backward Tensor : {backward_tensor}")
  #  print(f"FORWARD TENSOR in shelf {shelf} : {forward_tensor}")
   # print(f"BACKWARD TENSOR in shelf {shelf} : {backward_tensor}")

    end_time_model = time.time()
  #  print(f"Time taken to load both the images  and model: {end_time_model - start_time_model} seconds")

    start_time_algo = time.time()

    if forward_tensor is None and backward_tensor is None:
     #   print("No plant Bed or fruit present")
       # print("Total number of fruits = 0")
        return None
        
    if forward_tensor is None and backward_tensor is not None:
        backward_fruits = 0
        for row in range (backward_tensor.shape[0]):
            if backward_tensor[row,5] == fruit_class:
                backward_fruits +=1
        return [backward_fruits,fruit_class]
    
    if backward_tensor is None and forward_tensor is not None:
        forward_fruits = 0
        for row in range (forward_tensor.shape[0]):
            if forward_tensor[row,5] == fruit_class:
                forward_fruits += 1
        return [forward_fruits,fruit_class]
        
    unique_classes_forward, counts = np.unique(forward_tensor[:, 5], return_counts=True)
    class_counts_filtered_forward = counts[unique_classes_forward !=3]
    unique_classes_filtered_forward = unique_classes_forward[unique_classes_forward != 3]
    
    unique_classes_backward, counts = np.unique(backward_tensor[:, 5], return_counts=True)
    class_counts_filtered_backward = counts[unique_classes_backward !=3]
    unique_classes_filtered_backward = unique_classes_backward[unique_classes_backward != 3]

    # Check if there are occurrences of the target class
    if unique_classes_filtered_forward.size > 0 or  unique_classes_filtered_backward.size>0:
        # # Set the most occurring class (excluding class 3) as the target class
        if unique_classes_filtered_forward.size>0:
            most_occuring_class = unique_classes_filtered_forward[class_counts_filtered_forward.argmax()].item()
            target_class = most_occuring_class
        else:
            most_occuring_class = unique_classes_filtered_backward[class_counts_filtered_backward.argmax()].item()
            target_class = most_occuring_class
        
        # Extract the 7th and 8th columns (coordinates)
        coordinates_forward = forward_tensor[:, 6:8]
        coordinates_backward = backward_tensor[:, 6:8]

        # Get indices where class 3 occurs
        indices_class3_forward = np.where(forward_tensor[:, 5] == 3)[0]
        indices_class3_backward = np.where(backward_tensor[:, 5] == 3)[0]

        def count_fruits(plant_bed_tensor):
            # if shelf == 11:
            #     print(f"PLANT BED = {plant_bed_tensor}")
            num_rows = plant_bed_tensor.shape[0]
            num_fruits = num_rows  # Initialize with the maximum possible number of fruits
          #  print(f"Num Fruits = {num_fruits}")
            
            # Iterate through all pairs of rows
            for i in range(num_rows):
                for j in range(i + 1, num_rows):
                    # Compute the absolute difference between rows
                    diff = np.abs(plant_bed_tensor[i, :] - plant_bed_tensor[j, :])
                    if shelf == 4:
                        print(f"Difference between row {i} and {j} = {diff}")
                    
                    # Set a threshold for considering the rows as the same fruit
                    threshold = 0.09  # You may adjust this threshold based on your data

                    # Check if all elements are less than the threshold
                    if np.all(diff < threshold):
                        
                        num_fruits -= 1  # Subtract 1 if rows are considered the same fruit
                        if shelf == 4:
                            print(f"New num fruits = {num_fruits}")
                        
            return num_fruits

     #   print(f"indices forward in shelf {shelf} = {len(indices_class3_forward)}")
     #   print(f"indices backward in shelf {shelf} = {len(indices_class3_backward)}")
        
        if len(indices_class3_backward) == len(indices_class3_forward):

            # Create a list to store tensors for each plant bed
            plant_bed_tensors = []

            # Iterate through the indices to separate plant beds using coordinates
            for i in range(len(indices_class3_forward)):
                if i < len(indices_class3_forward) - 1:
                    # Extract rows between two class 3 rows (exclusive)
                    plant_bed_tensor_forward = coordinates_forward[indices_class3_forward[i]+1:indices_class3_forward[i+1], :]
                    plant_bed_tensor_backward = coordinates_backward[indices_class3_backward[i]+1:indices_class3_backward[i+1], :]
                else:
                    # For the last plant bed
                    plant_bed_tensor_forward = coordinates_forward[indices_class3_forward[i]+1:, :]
                    plant_bed_tensor_backward = coordinates_backward[indices_class3_backward[i]+1:, :]

                # Concatenate forward and backward tensors along rows
                plant_bed_tensor = np.concatenate([plant_bed_tensor_forward, plant_bed_tensor_backward], axis=0)

                # Append the plant bed tensor to the list
                plant_bed_tensors.append(plant_bed_tensor)

            if len(indices_class3_backward) == 1:
                plant_bed_1 = plant_bed_tensors[0]
                num_fruits_bed_1 = count_fruits(plant_bed_1)
               # print(f"Total fruits = : {num_fruits_bed_1}")
                end_time_algo = time.time()
              #  print(f"Total time for Algo = {end_time_algo - start_time_algo} seconds")
                return [num_fruits_bed_1 , target_class]

            if len(indices_class3_backward) == 2:
                plant_bed_1 = plant_bed_tensors[0]
                plant_bed_2 = plant_bed_tensors[1]
                num_fruits_bed_1 = count_fruits(plant_bed_1)
                num_fruits_bed_2 = count_fruits(plant_bed_2)
               # print(f"Number of fruits in Plant Bed 1: {num_fruits_bed_1}")
              #  print(f"Number of fruits in Plant Bed 2: {num_fruits_bed_2}")
              #  print(f"Total fruits = : {num_fruits_bed_1+num_fruits_bed_2}")
                end_time_algo = time.time()
               # print(f"Total time for Algo = {end_time_algo - start_time_algo} seconds")
                return [num_fruits_bed_1 + num_fruits_bed_2, target_class]

            if len(indices_class3_backward) == 3:
                plant_bed_1 = plant_bed_tensors[0]
                plant_bed_2 = plant_bed_tensors[1]
                plant_bed_3 = plant_bed_tensors[2]
                num_fruits_bed_1 = count_fruits(plant_bed_1)
                num_fruits_bed_2 = count_fruits(plant_bed_2)
                num_fruits_bed_3 = count_fruits(plant_bed_3)

                # Print the results
                # print(f"Number of fruits in Plant Bed 1: {num_fruits_bed_1}")
                # print(f"Number of fruits in Plant Bed 2: {num_fruits_bed_2}")
                # print(f"Number of fruits in Plant Bed 3: {num_fruits_bed_3}")
                # print(f"Total fruits = : {num_fruits_bed_3+num_fruits_bed_1+num_fruits_bed_2}")
                end_time_algo = time.time()
                # print(f"Total time for Algo = {end_time_algo - start_time_algo} seconds")
                return [num_fruits_bed_1 + num_fruits_bed_2+num_fruits_bed_3, target_class]
                
        else:
         #   print("NUMBER OF PLANT BEDS MISMATCH IN FORWARD AND BACKWARD")
                        
            # Create a list to store tensors for each plant bed
            plant_bed_tensors = []
            
            if len(indices_class3_forward) < len(indices_class3_backward):
                chosen_value = len(indices_class3_forward)
                plant_beds_left = len(indices_class3_backward) - chosen_value
                i=0
                number_of_fruits_extra = 0
                actual_fruits = 0
                for row in range (backward_tensor.shape[0]):
                    class_class = backward_tensor[row,5] 
                    if class_class == 3:
                        i+=1
                    if class_class == fruit_class:
                        actual_fruits +=1
                    if i > chosen_value and class_class == fruit_class:
                        number_of_fruits_extra +=1
                if chosen_value == 0:
                    return [actual_fruits,fruit_class]
                        
                        
            else:
                chosen_value = len(indices_class3_backward)
                plant_beds_left = len(indices_class3_forward) - chosen_value
                i=0
                number_of_fruits_extra = 0
                actual_fruits = 0
                for row in range (forward_tensor.shape[0]):
                    class_class = forward_tensor[row,5] 
                    if class_class == 3:
                        i+=1
                    if class_class == fruit_class:
                        actual_fruits +=1 
                    if i > chosen_value and class_class == fruit_class:
                        number_of_fruits_extra +=1
                if chosen_value == 0:
                    return [actual_fruits,fruit_class]
                
            # Iterate through the indices to separate plant beds using coordinates
            for i in range(chosen_value):
                if i < chosen_value - 1:
                    # Extract rows between two class 3 rows (exclusive)
                    plant_bed_tensor_forward = coordinates_forward[indices_class3_forward[i]+1:indices_class3_forward[i+1], :]
                    plant_bed_tensor_backward = coordinates_backward[indices_class3_backward[i]+1:indices_class3_backward[i+1], :]
                else:
                    # For the last plant bed
                    plant_bed_tensor_forward = coordinates_forward[indices_class3_forward[i]+1:, :]
                    plant_bed_tensor_backward = coordinates_backward[indices_class3_backward[i]+1:, :]

                # Concatenate forward and backward tensors along rows
                plant_bed_tensor = np.concatenate([plant_bed_tensor_forward, plant_bed_tensor_backward], axis=0)

                # Append the plant bed tensor to the list
                plant_bed_tensors.append(plant_bed_tensor)
           # print(chosen_value, "of shelf: ",shelf)
            if chosen_value == 1:
                plant_bed_1 = plant_bed_tensors[0]
                num_fruits_bed_1 = count_fruits(plant_bed_1)
                # print(f"Number of Plant Beds : 1")
                # print(f"Number of extra fruits in {plant_beds_left} Plant beds : {number_of_fruits_extra}")
                # print(f"Total fruits = : {num_fruits_bed_1}")
                end_time_algo = time.time()
                #print(f"Total time for Algo = {end_time_algo - start_time_algo} seconds")
                return [num_fruits_bed_1+number_of_fruits_extra,target_class]
                
            if chosen_value == 2:
                plant_bed_1 = plant_bed_tensors[0]
                plant_bed_2 = plant_bed_tensors[1]
                num_fruits_bed_1 = count_fruits(plant_bed_1)
                num_fruits_bed_2 = count_fruits(plant_bed_2)
                # print(f"Number of fruits in Plant Bed 1: {num_fruits_bed_1}")
                # print(f"Number of fruits in Plant Bed 2: {num_fruits_bed_2}")
                # print(f"Number of extra fruits in {plant_beds_left} Plant beds : {number_of_fruits_extra}")
                # print(f"Total fruits = : {num_fruits_bed_1+num_fruits_bed_2}")
                end_time_algo = time.time()
               # print(f"Total time for Algo = {end_time_algo - start_time_algo} seconds")
                return [num_fruits_bed_1+num_fruits_bed_2+number_of_fruits_extra,target_class]
                
            if chosen_value == 3:
                plant_bed_1 = plant_bed_tensors[0]
                plant_bed_2 = plant_bed_tensors[1]
                plant_bed_3 = plant_bed_tensors[2]
                num_fruits_bed_1 = count_fruits(plant_bed_1)
                num_fruits_bed_2 = count_fruits(plant_bed_2)
                num_fruits_bed_3 = count_fruits(plant_bed_3)

                # Print the results
                # print(f"Number of fruits in Plant Bed 1: {num_fruits_bed_1}")
                # print(f"Number of fruits in Plant Bed 2: {num_fruits_bed_2}")
                # print(f"Number of fruits in Plant Bed 3: {num_fruits_bed_3}")
                # print(f"Number of extra fruits in {plant_beds_left} Plant beds : {number_of_fruits_extra}")
                # print(f"Total fruits = : {num_fruits_bed_3+num_fruits_bed_1+num_fruits_bed_2}")
                end_time_algo = time.time()
               # print(f"Total time for Algo = {end_time_algo - start_time_algo} seconds")
                return [num_fruits_bed_1+num_fruits_bed_2+num_fruits_bed_3+number_of_fruits_extra,target_class]
                
            end_time_algo = time.time()
           # print(f"Total time for Algo = {end_time_algo - start_time_algo} seconds")
            

    else:
      #  print("No occurrences of the target class found.")
        end_time_algo = time.time()
       # print(f"Total time for Algo = {end_time_algo - start_time_algo} seconds")
        return [0,"No Fruit"]

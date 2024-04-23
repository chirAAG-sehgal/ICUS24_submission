from ultralytics import YOLO
import cv2
import os
import numpy as np
import torch
import time
import statistics

def backward_function(shelf,model,fruit_class,forward_PB_review):
    
    def conv_image_2(coloured_image_path,depth_image_path):
        depth_image = cv2.imread(depth_image_path)
        colored_image = cv2.imread(coloured_image_path)

        _, depth_image_thresholded = cv2.threshold(depth_image, 10, 255, cv2.THRESH_TOZERO)
        _, depth_image_thresholded = cv2.threshold(depth_image_thresholded, 200, 255, cv2.THRESH_TOZERO_INV)

        depth_image_thresholded_grayscale = cv2.cvtColor(depth_image_thresholded, cv2.COLOR_BGR2GRAY)

        mask = cv2.inRange(depth_image_thresholded_grayscale, 10,200)

        result_image = cv2.bitwise_and(colored_image, colored_image,mask=mask)
        return result_image

    def rotate(image, angle):
        rows, cols = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), -angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
        return rotated_image

    def oriented_and_subtracted(image_path,depth_path,coordinate_path):
        img_preprocessed = conv_image_2(image_path,depth_path)
            
        with open(coordinate_path, "r") as file:
            angle = float(file.readline())
            image_preprocessed_rotated = rotate(img_preprocessed, angle)
        return image_preprocessed_rotated

    start_time_load = time.time()

    images_or_sub = []   # Creating an empty list to feed into the DL model
    # Reading images from saved folder
    folder = "images"
    for number in range(25):
        image_path = f"{folder}/{shelf}B/img_{shelf}B_{number}.jpg"
        depth_path = f"{folder}/{shelf}B/depth_{shelf}B_{number}.jpg"
        coord_path = f"{folder}/{shelf}B/coordinate_{shelf}B_{number}.txt"
        image_oriented_subtracted = oriented_and_subtracted(image_path, depth_path, coord_path)
        cv2.imwrite(f"Images_while_comparing/backward/{folder}_img_{shelf}_{number}.jpg",image_oriented_subtracted)
        images_or_sub.append(image_oriented_subtracted)

    results = model(images_or_sub, imgsz=(480, 640), conf=0.5,verbose=False)
    #print(f"Total NUmber of results = {len(results)}")
    end_time_load = time.time()
  #  print(f"Time to compute results in backward = {end_time_load - start_time_load}")
    
    start_time_algo = time.time()

    plant_bed_counts = [] # Creating a list to store the counts of each bed
    sorted_results_list = []

    i=0
    for r in results:
        coordinates = r.boxes.xyxy
        conf = r.boxes.conf.unsqueeze(1)
        cls = r.boxes.cls.unsqueeze(1)
        concatenated_results = torch.cat((coordinates, conf, cls), dim=1)
        sorted_results = concatenated_results[torch.argsort(concatenated_results[:, 2], descending=True)]
        sorted_results = sorted_results.cpu().detach().numpy()
        sorted_results_list.append(sorted_results)
        
        unique_classes, counts = np.unique(sorted_results[:, 5], return_counts=True)
        #print(f"Unique classes in backward image {i}= {unique_classes}, Counts = {counts}")
        
        i+=1

        # Calculate plant bed count
        if 3 in unique_classes and counts[unique_classes == 3].sum().item() <= 3:
            
            class_counts_filtered = counts[unique_classes == 3]
            plant_bed_counts.append(class_counts_filtered.sum().item())
            #print(f"Plant Bed Backward in shelf {shelf} = {class_counts_filtered.sum().item()}")
            
        elif len(unique_classes) == 0 :
            #print(f"No class found in backward shelf {shelf}")
            plant_bed_counts.append(0)
            
        else:
            #print(f"Skipped Image as plant beds in backward shelf {shelf} > 3")
            # Append a different value (None in this case)
            plant_bed_counts.append(None)
    
    filtered_plant_bed_counts = [count for count in plant_bed_counts if count is not None]
    general_plant_bed_count = statistics.mode(filtered_plant_bed_counts)
    
    #print(f"Chosen Plant Bed Value from Backwards Images in shelf {shelf}= {general_plant_bed_count} \n")

    valid_tensor_indices_plantbeds = [i for i, count in enumerate(plant_bed_counts) if count == general_plant_bed_count]

    if len(valid_tensor_indices_plantbeds) == 0 or general_plant_bed_count == 0:
      #  print(f"No valid tensors found in forward shelf {shelf}.")
        return None
    
    tensors_with_general_plant_beds = []
    
    for index in valid_tensor_indices_plantbeds:
        tensor = sorted_results_list[index]
        tensors_with_general_plant_beds.append(tensor)
    
    '''
    Now We will remove the fruits whichh lie outside of a plant bed 
    then remove the fruits which we don't want to identify.
    '''
    
    tensor_list_without_outside_fruit = []

    for tensor in tensors_with_general_plant_beds:
        
        class_column = tensor[:,5]
        
        center_coord_list = [] # Creating a list to store the coordinates of the center of the fruit
        plant_bed_coord_list = [] # Creating a list to store the plant Coordinates
        fruit_rows_index = [] # Creating a list to store the index at which the fruit is occuring in the tensor 
        rows_to_remove = [] # Creating a list to store the index of the rows to remove 
        
        for row in range (tensor.shape[0]):
          
            if class_column[row] != 3:
                center_coordinate = ((tensor[row,0] + tensor[row,2])/2),((tensor[row,1]+tensor[row,3])/2)
                center_coord_list.append(center_coordinate)
                fruit_rows_index.append(row)
                
            if class_column[row] == 3:
                plant_bed_xmin , plant_bed_ymin,plant_bed_xmax,plant_bed_ymax = tensor[row,:4] 
                plant_bed_coord = (plant_bed_xmin , plant_bed_ymin,plant_bed_xmax,plant_bed_ymax)
                #print(f"PB Coord : {plant_bed_coord}")
                plant_bed_coord_list.append(plant_bed_coord)
                
        #print("\nCenter Coordinates List:\n",center_coord_list,"\n")
        #print(f"Plant Bed Coordinate List : {plant_bed_coord_list}")        
            
        for row, center_coord in enumerate(center_coord_list):
            #print("Hello Again Again")
            
            center_in_any_plant_bed = any(
                plant_bed[0] <= center_coord[0] <= plant_bed[2] and plant_bed[1] <= center_coord[1] <= plant_bed[3]
                for plant_bed in plant_bed_coord_list
            )
            if not center_in_any_plant_bed:
                rows_to_remove.append(row)

        indices_to_delete = [fruit_rows_index[number] for number in rows_to_remove]
        #print(f"Indices to Delete : {indices_to_delete}")
        tensor_filtered = np.delete(tensor, indices_to_delete, axis=0)
        #print(f"Filtered Tensor Shape : {tensor_filtered.shape}\n")
        
        #print(f"Before : {tensor_filtered}")
        rows_to_remove=[]
        index_PB_forward , general_plant_bed_count_forward = forward_PB_review
        #print(f"Index PB Forward : {index_PB_forward}\ngeneral PB count Forward : {general_plant_bed_count_forward}")
        if general_plant_bed_count_forward == general_plant_bed_count:
            indices_class3_backward = np.where(tensor_filtered[:, 5] == 3)[0]
            #print(f"Indices class3 backward : {indices_class3_backward}")
            for indexes in index_PB_forward:
                #print(f"Indexes : {indexes}")
                tensor_filtered[indices_class3_backward[indexes],5] = 4
        
        #print(f"Before : {tensor_filtered}")
        #print(f"After : {tensor_filtered}")
        indices_class3_backward = np.where(tensor_filtered[:, 5] == 3)[0]
        #print(f"Indices clas3 backward : {indices_class3_backward}")
        
        rows_to_remove = []        
        for row in range (tensor_filtered.shape[0]):
            if row+1 < tensor_filtered.shape[0]:
                if tensor_filtered[row,5] == 3 and tensor_filtered[row+1,5] != fruit_class and tensor_filtered[row+1,5] != 3 and tensor_filtered[row+1,5] != 4:
                    rows_to_remove.append(row)
        
        tensor_filtered = np.delete(tensor_filtered,rows_to_remove,axis=0)
        
        
        rows_to_remove=[]
        
        for row in range (tensor_filtered.shape[0]):
            if tensor_filtered[row,5]!=3 and tensor_filtered[row,5]!=fruit_class:
                rows_to_remove.append(row)
        
        tensor_filtered = np.delete(tensor_filtered, rows_to_remove, axis=0)
            
        #tensor_filtered = torch.tensor(tensor_filtered.cpu().detach().numpy())
        #tensor = torch.tensor(tensor_filtered)
        
        #print(f"\nFiltered tensor : {tensor_filtered}\n")
        
        tensor_list_without_outside_fruit.append(tensor_filtered)
    
  #  print(f"Valid Tensor Indixes : {valid_tensor_indices_plantbeds}")
    
    fruit_count_list = []
    valid_fruit_index = []
    
    index=0
    
    for array  in tensor_list_without_outside_fruit:
        
        valid_sorted_results = array 
        valid_unique_classes, valid_counts = np.unique(valid_sorted_results[:, 5], return_counts=True)
        fruit_count = valid_counts[valid_unique_classes !=3].sum().item()
        valid_unique_classes_filtered =  valid_unique_classes[valid_unique_classes !=3]
        
        if valid_unique_classes_filtered.size >0:
            
            fruit_count_list.append(fruit_count)
            valid_fruit_index.append(index)
            #print(f"Fruit cound in image {index} : {fruit_count}")
            
        else:
            #print("No fruits Found")
            fruit_count_list.append(0)
            valid_fruit_index.append(index)
            #print(f"Fruit cound in image {index} : {fruit_count}")
        
        index+=1
            
    general_fruit_count = statistics.mode(fruit_count_list)
   # print(f"Chosen Fruit Count from Backward Images in shelf {shelf}= {general_fruit_count} \n")
    
    valid_tensor_indices_fruit = [i for i, count in enumerate(fruit_count_list) if count == general_fruit_count]
  #  print(f"valid tensor fruit count : {valid_tensor_indices_fruit}")

    middle_value = len(valid_tensor_indices_fruit) // 2
    middle_index = valid_tensor_indices_fruit[middle_value]
    chosen_tensor = tensor_list_without_outside_fruit[middle_index]
  #  print(f"Chosen image number forward = {valid_tensor_indices_fruit[middle_index]}")
    
    sorted_results = chosen_tensor

    chosen_image = images_or_sub[middle_index]
    cv2.imwrite(f"saved_images/{shelf}_backward.jpg",chosen_image)

    unique_classes, counts = np.unique(sorted_results[:, 5], return_counts=True)

    # Filter out the occurrences of class 3
    class_counts_filtered = counts[unique_classes != 3]
    unique_classes_filtered = unique_classes[unique_classes != 3]

    # Check if there are occurrences of the target class
    if unique_classes_filtered.size > 0:
        most_occuring_class = unique_classes_filtered[class_counts_filtered.argmax()].item()
        target_class = most_occuring_class
        
        reference_class = 3

        # Create a new tensor to store the results
        result_tensor = np.zeros((0, 8))  # 8 columns for the concatenated result

        # Iterate through rows and perform calculations
        for i in range(sorted_results.shape[0]):
            row = sorted_results[i, :]
            # Check if the row belongs to the target class (fruit)
            if row[-1] == target_class:
                target_class_center_x = (row[0] + row[2]) / 2  # Calculate the x-coordinate of the center of the target class bounding box
                target_class_center_y = (row[1] + row[3]) / 2  # Calculate the y-coordinate of the center of the target class bounding box
                # Check the above rows until a plant bed class is found
                for j in range(i - 1, -1, -1):
                    above_row = sorted_results[j, :]
                    if above_row[-1] == reference_class:
                        # Calculate the relative position of the fruit with respect to the right and upper edges of the plant bed
                        relative_position_right = (above_row[2] - target_class_center_x) / (above_row[2] - above_row[0])
                        relative_position_upper = (target_class_center_y - above_row[1]) / (above_row[3] - above_row[1])

                        # Append the results to the new tensor
                        # Append the results to the new tensor
                        new_row = np.array([[
                            row[0], row[1], row[2], row[3],  # x_min, y_min, x_max, y_max
                            row[4], row[5],  # conf, class
                            relative_position_right, relative_position_upper  # left edge, upper edge
                        ]])
                        result_tensor = np.concatenate([result_tensor,new_row.reshape(1,-1)],axis=0)
                        break
            elif row[-1] == reference_class:
                # For rows with class 3, add the row directly with right and upper edge set to 0
                new_row = np.array([[
                    row[0], row[1], row[2], row[3],  # x_min, y_min, x_max, y_max
                    row[4], row[5],  # conf, class
                    0.0, 0.0  # left edge, upper edge (set to 0 for class 3)
                ]])
                result_tensor = np.concatenate([result_tensor,new_row],axis=0)

        np.set_printoptions(precision=4, suppress=True)
        #print(f"Backward Result = {result_tensor}")
       # print(f"Backward Result shelf {shelf} : {result_tensor}")
        return result_tensor

    else:
      #  print("No occurrences of the target class found.")
        # Create a new tensor to store the results
        result_tensor = np.zeros((0, 8))  # 8 columns for the concatenated result
        reference_class = 3
        # Iterate through rows and add rows to the result tensor with class 3 (plant bed)
        for i in range(sorted_results.shape[0]):
            row = sorted_results[i, :]
            if row[-1] == reference_class:
                # For rows with class 3, add the row directly with left and upper edge set to 0
                new_row = np.array([[
                    row[0], row[1], row[2], row[3],  # x_min, y_min, x_max, y_max
                    row[4], row[5],  # conf, class
                    0.0, 0.0  # left edge, upper edge (set to 0 for class 3)
                ]])
                result_tensor = np.concatenate([result_tensor,new_row],axis=0)
        np.set_printoptions(precision=4, suppress=True)
        end_time_Algo = time.time()
        # print(f"Time taken by Algo = {end_time_Algo - start_time_algo} sec")
        #print(f"Backward Result shelf {shelf} : {result_tensor}")
        return result_tensor
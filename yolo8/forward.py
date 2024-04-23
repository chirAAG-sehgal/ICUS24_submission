from ultralytics import YOLO
import cv2
import os
import numpy as np
import torch
import time
import statistics

def forward_function(shelf,model,fruit_class):
    
    def conv_image_2(coloured_image_path, depth_image_path):
        depth_image = cv2.imread(depth_image_path)
        colored_image = cv2.imread(coloured_image_path)

        _, depth_image_thresholded = cv2.threshold(depth_image, 10, 255, cv2.THRESH_TOZERO)
        _, depth_image_thresholded = cv2.threshold(depth_image_thresholded, 200, 255, cv2.THRESH_TOZERO_INV)

        depth_image_thresholded_grayscale = cv2.cvtColor(depth_image_thresholded, cv2.COLOR_BGR2GRAY)

        mask = cv2.inRange(depth_image_thresholded_grayscale, 10, 200)

        result_image = cv2.bitwise_and(colored_image, colored_image, mask=mask)
        return result_image

    def rotate(image, angle):
        rows, cols = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), -angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
        return rotated_image

    def oriented_and_subtracted(image_path, depth_path, coordinate_path):
        img_preprocessed = conv_image_2(image_path, depth_path)

        with open(coordinate_path, "r") as file:
            angle = float(file.readline())
            image_preprocessed_rotated = rotate(img_preprocessed, angle)
        return image_preprocessed_rotated

    start_time_load = time.time()
    
    images_or_sub = []   # Creating an empty list to feed into the DL model
    # Reading images from saved folder
    folder = "images"
    for number in range(25):
        image_path = f"{folder}/{shelf}F/img_{shelf}F_{number}.jpg"
        depth_path = f"{folder}/{shelf}F/depth_{shelf}F_{number}.jpg"
        coord_path = f"{folder}/{shelf}F/coordinate_{shelf}F_{number}.txt"
        image_oriented_subtracted = oriented_and_subtracted(image_path, depth_path, coord_path)
        images_or_sub.append(image_oriented_subtracted)
    # Perform inference on the images
    results = model(images_or_sub, imgsz=(480, 640), conf=0.5,verbose=False)
    
    end_time_load = time.time()
    #print(f"Time to compute results in forward = {end_time_load - start_time_load}")
    
    start_time_algo = time.time()

    plant_bed_counts = [] # Creating a list to store the counts of each bed
    sorted_results_list = []
    
    i = 0
    for r in results:
        
        coordinates = r.boxes.xyxy
        conf = r.boxes.conf.unsqueeze(1)
        cls = r.boxes.cls.unsqueeze(1)
        concatenated_results = torch.cat((coordinates, conf, cls), dim=1)
        sorted_results = concatenated_results[torch.argsort(concatenated_results[:, 0])]
        sorted_results = sorted_results.cpu().detach().numpy()
        sorted_results_list.append(sorted_results)
        
        unique_classes, counts = np.unique(cls, return_counts=True)
        
            
            #print(f"Unique classes in forward image {i}= {unique_classes}, Counts = {counts}")
            
        
        i+=1
        
        if 3 in unique_classes and counts[unique_classes == 3].item() <= 3:
            
            class_counts_filtered = counts[unique_classes == 3]
            
                
                #print(f"Class count filtered  = {class_counts_filtered}")
            plant_bed_counts.append(class_counts_filtered.item())
            
                    
                #print(f"Plant Bed forward in shelf {shelf} = {class_counts_filtered.item()}")
            
        elif len(unique_classes) == 0 :
            
            
                
                #print(f"No class found in shelf {shelf}")
                plant_bed_counts.append(0)
            
        else:
            
            
                
                #print(f"Skipped Image as plant beds in forward shelf {shelf}> 3")
            # Append a different value (None in this case)
            plant_bed_counts.append(None)

    filtered_plant_bed_counts = [count for count in plant_bed_counts if count is not None]
    if filtered_plant_bed_counts!= None:    
        general_plant_bed_count = statistics.mode(filtered_plant_bed_counts)
    
    
            
            #print(f"Chosen Plant Bed Value from forwards Images in shelf {shelf}= {general_plant_bed_count} \n")

    valid_tensor_indices_plantbeds = [i for i, count in enumerate(plant_bed_counts) if count == general_plant_bed_count]

    if len(valid_tensor_indices_plantbeds) == 0 or general_plant_bed_count == 0:
        
            
            #print(f"No valid tensors found in forward shelf {shelf}.")
        return None,None
    
    filtered_tensor_list = []
    
    for index in valid_tensor_indices_plantbeds:
        tensor = sorted_results_list[index]
        filtered_tensor_list.append(tensor)
        
    ##print(f"{len(filtered_tensor_list)}")
    ##print(filtered_tensor_list)
    
    tensor_list_without_outside_fruit = []
        
    i=0
    forward_PB_review_list = []
    for tensor in filtered_tensor_list:
        
        
            
            #print(f"Original Tensor : {tensor}")
        
        i+=1
        class_column = tensor[:,5]
        
        center_coord_list = []
        plant_bed_coord_list = []
        fruit_rows_index = []
        rows_to_remove = []
        
        for row in range (tensor.shape[0]):

            if class_column[row] != 3:
                center_coordinate = ((tensor[row,0] + tensor[row,2])/2),((tensor[row,1]+tensor[row,3])/2)
                center_coord_list.append(center_coordinate)
                fruit_rows_index.append(row)
                
            if class_column[row] == 3:
                plant_bed_xmin , plant_bed_ymin,plant_bed_xmax,plant_bed_ymax = tensor[row,:4] 
                plant_bed_coord = (plant_bed_xmin , plant_bed_ymin,plant_bed_xmax,plant_bed_ymax)
                plant_bed_coord_list.append(plant_bed_coord)

        for row, center_coord in enumerate(center_coord_list):

            center_in_any_plant_bed = any(
                plant_bed[0] <= center_coord[0] <= plant_bed[2] and plant_bed[1] <= center_coord[1] <= plant_bed[3]
                for plant_bed in plant_bed_coord_list
            )
            if not center_in_any_plant_bed:
                rows_to_remove.append(row)

        indices_to_delete = [fruit_rows_index[number] for number in rows_to_remove]
        tensor_filtered = np.delete(tensor, indices_to_delete, axis=0)

        rows_to_remove = []
        
            
            #print(f"\nFiltered tensor after area deletiion: {tensor_filtered}\n")
        #print(tensor_filtered.shape[0])
        for row in range (tensor_filtered.shape[0]):
            if row+1 < tensor_filtered.shape[0]:
                if tensor_filtered[row,5] == 3. and tensor_filtered[row+1,5] != float(fruit_class) and tensor_filtered[row+1,5] != 3.:
                    rows_to_remove.append(row)
        
            
        #print(f"rows to remove : {rows_to_remove}")
        #print(type(tensor_filtered))
        tensor_filtered = np.delete(tensor_filtered,rows_to_remove,axis=0)
        
            
            #print(f"\nFiltered tensor after plant deletiion: {tensor_filtered}\n")
        no_of_plantbeds_deleted = len(rows_to_remove)
        indices_class3_forward = np.where(tensor_filtered[:, 5] == 3)[0]
        indices_to_remove = indices_class3_forward[np.isin(indices_class3_forward, rows_to_remove)]
        index_PB = np.where(np.isin(indices_class3_forward, indices_to_remove))[0]
        ##print(f"Index PB : {index_PB}")
        forward_PB_review = (index_PB,len(indices_class3_forward))
        forward_PB_review_list.append(forward_PB_review)
        ##print(f"Number of Plant Beds Deleted : {no_of_plantbeds_deleted}")
        ##print(f"Indices class 3 forward : {indices_class3_forward}")
        ##print(f"Indices to Remove : {indices_to_remove}")        
        
        
        
        rows_to_remove=[]
        
        for row in range (tensor_filtered.shape[0]):
            if tensor_filtered[row,5]!=3. and tensor_filtered[row,5]!=float(fruit_class):
                rows_to_remove.append(row)
        #print(f"rows to removed 2 {rows_to_remove}")
        tensor_filtered = np.delete(tensor_filtered, rows_to_remove, axis=0)

        
            
            #print(f"\nFiltered tensor final afterfruit removal : {tensor_filtered}\n")
        
        tensor_list_without_outside_fruit.append(tensor_filtered)
    
    
            
            #print(f"Valid Tensor Indixes : {valid_tensor_indices_plantbeds}")

    fruit_count_list = []
    valid_fruit_index = []
    
    index = 0
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
            ##print(f"Fruit cound in image {index} : {fruit_count}")
            
        index+=1
            
    general_fruit_count = statistics.mode(fruit_count_list)
    
            
            #print(f"Chosen Fruit Count from Forward Images in shelf {shelf}= {general_fruit_count} \n")
    
    valid_tensor_indices_fruit = [i for i, count in enumerate(fruit_count_list) if count == general_fruit_count]
    
            
            #print(f"valid tensor fruit count : {valid_tensor_indices_fruit}")
        
    middle_value = len(valid_tensor_indices_fruit) // 2
    middle_index = valid_tensor_indices_fruit[middle_value]
    #print(f"Tensor indexes : {valid_tensor_indices_fruit}")
   # print(f"Middle Value : {middle_value} | Middle Index : {middle_index}")
    chosen_tensor = tensor_list_without_outside_fruit[middle_index]
    forward_PB_review = forward_PB_review_list[middle_index]
    
            
            #print(f"Chosen image number forward = {valid_tensor_indices_plantbeds[middle_index]}")
    chosen_image = images_or_sub[middle_index]
    cv2.imwrite(f"saved_images/{shelf}_foward.jpg",chosen_image)
    
    sorted_results = chosen_tensor  

    unique_classes, counts = np.unique(sorted_results[:, 5], return_counts=True)

    # Filter out the occurrences of class 3
    class_counts_filtered = counts[unique_classes != 3]
    unique_classes_filtered = unique_classes[unique_classes != 3]

    # Check if there are occurrences of the target class
    if unique_classes_filtered.size > 0:
        most_occuring_class = unique_classes_filtered[class_counts_filtered.argmax()].item()
        target_class = most_occuring_class
        
        reference_class = 3  # Plant bed

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
                        # Calculate the relative position of the fruit with respect to the left and upper edges of the plant bed
                        relative_position_left = (target_class_center_x - above_row[0]) / (above_row[2] - above_row[0])
                        relative_position_upper = (target_class_center_y - above_row[1]) / (above_row[3] - above_row[1])

                        # Append the results to the new tensor
                        new_row = np.array([[
                            row[0], row[1], row[2], row[3],  # x_min, y_min, x_max, y_max
                            row[4], row[5],  # conf, class
                            relative_position_left, relative_position_upper  # left edge, upper edge
                        ]])
                        result_tensor = np.concatenate([result_tensor,new_row.reshape(1,-1)],axis=0)
                        break
            elif row[-1] == reference_class:
                # For rows with class 3, add the row directly with left and upper edge set to 0
                new_row = np.array([[
                    row[0], row[1], row[2], row[3],  # x_min, y_min, x_max, y_max
                    row[4], row[5],  # conf, class
                    0.0, 0.0  # left edge, upper edge (set to 0 for class 3)
                ]])
                result_tensor = np.concatenate([result_tensor,new_row],axis=0)

        np.set_printoptions(precision=4, suppress=True)
        end_time_Algo = time.time()
        ##print(f"Time taken by Algo = {end_time_Algo - start_time_algo} sec")
        #print(f"Forward Result shelf {shelf} : {result_tensor}")
        return result_tensor,forward_PB_review
    else:
        #print("No occurrences of the target class found.")
        result_tensor = np.zeros((0, 8))  
        reference_class = 3
        # Iterate through rows and add rows to the result tensor with class 3 (plant bed)
        for i in range(sorted_results.shape[0]):
            row = sorted_results[i, :]
            if row[-1] == reference_class:
                new_row = np.array([[
                    row[0], row[1], row[2], row[3],  # x_min, y_min, x_max, y_max
                    row[4], row[5],  # conf, class
                    0.0, 0.0  # left edge, upper edge (set to 0 for class 3)
                ]])
                result_tensor = np.concatenate([result_tensor,new_row],axis=0)
        
        np.set_printoptions(precision=4, suppress=True)
        end_time_Algo = time.time()
        # #print(f"Time taken by Algo = {end_time_Algo - start_time_algo} sec")
        #print(f"Forward Result shelf {shelf} : {result_tensor}")
        return result_tensor,forward_PB_review
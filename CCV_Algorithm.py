import cv2
import numpy as np
import cv2
import matplotlib.pyplot as plt


def calculate_ccv(image, num_bins=64, num_cells=8):
    # Apply Gaussian blur
    image = cv2.GaussianBlur(image, (3, 3), 0)  # You can adjust the kernel size (e.g., (5, 5)) and sigma value (e.g., 0) as needed
    
    # Convert image to Lab color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Split Lab image into channels
    L, a, b = cv2.split(lab_image)
    print("L channel shape:",L.shape)


    # Calculate grid size
    rows, cols = L.shape
    print("row from L channel shape:",rows)
    print("Column from L channel shape:",cols)
    cell_rows = rows // num_cells
    cell_cols = cols // num_cells
    print("Cell rows:",cell_rows)
    print("Cell columns:",cell_cols)
    # Initialize CCV histogram
    ccv_hist = np.zeros((num_cells, num_cells, num_cells), dtype=np.float32)
    print("CCV Hist Shape:",ccv_hist.shape)
    # Iterate through each cell
    counter = 0
    for i in range(num_cells):
        for j in range(num_cells):
            print("iteration number: [",i," : ",j,"]")
            print("L channel shape:",L.shape)
            print("rows and Columns from L channel shape:",cell_rows," ",cell_cols)
            print("CCV Hist Shape, Initialize:",ccv_hist.shape)
            print("------------------------------")
            # Extract cell
            cell_L = L[i*cell_rows:(i+1)*cell_rows, j*cell_cols:(j+1)*cell_cols]
            cell_a = a[i*cell_rows:(i+1)*cell_rows, j*cell_cols:(j+1)*cell_cols]
            cell_b = b[i*cell_rows:(i+1)*cell_rows, j*cell_cols:(j+1)*cell_cols]
            # print(f"L shape of iteration {j} : {cell_L.shape}")
            print("L shape: ",cell_L.shape)
            counter+=1 
            
            # Calculate histogram for each channel
            hist_L = cv2.calcHist([cell_L], [0], None, [num_bins], [0, 256])
            hist_a = cv2.calcHist([cell_a], [0], None, [num_bins], [0, 256])
            hist_b = cv2.calcHist([cell_b], [0], None, [num_bins], [0, 256])
            print("Hist L shape: ",hist_L.shape)
            # Normalize histograms
            hist_L /= (cell_rows * cell_cols)
            hist_a /= (cell_rows * cell_cols)
            hist_b /= (cell_rows * cell_cols)
            print("Normalized hist L Shape: ",hist_L.shape)
            # print("hist_a------",hist_a)
            # print("hist_b------",hist_b)
            
            
            # Concatenate histograms
            cell_hist = np.concatenate((hist_L, hist_a, hist_b), axis=1)
            print("Cell Hist shape or Concatinate ",cell_hist.shape)

            #  Averaging the values in cell_hist
            averaged_values = np.mean(cell_hist.reshape(-1, 8, 3), axis=(0, 2))
            print("Average Values of Cell Hist Shape: ",averaged_values.shape)
           
            # Assigning the averaged values to ccv_hist[i, j]
            ccv_hist[i, j] = averaged_values
            print("Assign Avg value to CCV Hist, shape: ",ccv_hist[i, j].shape)
            print('==============================')
            # print("Updated every iteration CCV Hist shape:",ccv_hist)
            # print('==============================')

    
    print(counter)  
    print("Final CCV Hist shape:",ccv_hist .shape)
    return ccv_hist.flatten()

# Example usage:
image_path = "example1.jpg"  # Path to your image
image = cv2.imread(image_path)

ccv_feature = calculate_ccv(image)
print("CCV feature vector shape:", ccv_feature.shape)
print("CCV feature vector:", ccv_feature)

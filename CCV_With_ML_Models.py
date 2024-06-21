import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
# ---------------------------------------------------------- Models
from sklearn.svm import SVC


# ---------------------------------------------------------- Calculate Color Coherent Vector
def calculate_ccv(image, num_bins=64, num_cells=8):

    # --------------------------------------------- Apply Gaussian blur
    image = cv2.GaussianBlur(image, (3, 3), 0)  # You can adjust the kernel size (e.g., (5, 5)) and sigma value (e.g., 0) as needed

    # --------------------------------------------- Convert image to Lab color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # --------------------------------------------- Split Lab image into channels
    L, a, b = cv2.split(lab_image)
    # print("L channel shape:",L.shape)

    # --------------------------------------------- Calculate grid size
    rows, cols = L.shape
    # print("row from L channel shape:",rows)
    # print("Column from L channel shape:",cols)
    cell_rows = rows // num_cells
    cell_cols = cols // num_cells
    # print("Cell row Shape:",cell_rows)
    # print("Cell column Shape:",cell_cols)

    # ---------------------------------------------Initialize CCV histogram
    ccv_hist = np.zeros((num_cells, num_cells, num_cells), dtype=np.float32)
    # print("CCV Hist Shape:",ccv_hist.shape)

    # ---------------------------------------------Iterate through each cell
    counter = 0
    for i in range(num_cells):
        for j in range(num_cells):

            # ---------------------------------------------Extract cell
            cell_L = L[i*cell_rows:(i+1)*cell_rows, j*cell_cols:(j+1)*cell_cols]
            cell_a = a[i*cell_rows:(i+1)*cell_rows, j*cell_cols:(j+1)*cell_cols]
            cell_b = b[i*cell_rows:(i+1)*cell_rows, j*cell_cols:(j+1)*cell_cols]
            # print(f"L shape of iteration {j} : {cell_L.shape}")

            counter+=1
            # print(counter)
            # print("------",cell_a)
            # print("------",cell_b)
            
            # --------------------------------------------- Calculate histogram for each channel
            hist_L = cv2.calcHist([cell_L], [0], None, [num_bins], [0, 256])
            hist_a = cv2.calcHist([cell_a], [0], None, [num_bins], [0, 256])
            hist_b = cv2.calcHist([cell_b], [0], None, [num_bins], [0, 256])
            # print(hist_L)

            # --------------------------------------------- Normalize histograms
            hist_L /= (cell_rows * cell_cols)
            hist_a /= (cell_rows * cell_cols)
            hist_b /= (cell_rows * cell_cols)
            # print("hist_L------",hist_L)
            # print("hist_a------",hist_a)
            # print("hist_b------",hist_b)
            
            # --------------------------------------------- Concatenate histograms
            cell_hist = np.concatenate((hist_L, hist_a, hist_b), axis=1)
            # print("Cell Hist shape; ",cell_hist.shape)

            # print("ccv_hist ************************************")
            # print("shape of ccv_hist[i, j]: ",ccv_hist[i, j].shape)
            # print("cell_hist ************************************")
            # print("shape of cell_hist: ",cell_hist.shape)
            # print("************************************")
            
            # --------------------------------------------- Averaging the values in cell_hist
            averaged_values = np.mean(cell_hist.reshape(-1, 8, 3), axis=(0, 2))
            # print("Average Values Shape:",averaged_values.shape)

            # --------------------------------------------- Assigning the averaged values to ccv_hist[i, j]
            ccv_hist[i, j] = averaged_values
            # print("CCV Hist shape: ",ccv_hist[i, j].shape)
            
    # print(counter)  
    # print("Final CCV Hist shape:",ccv_hist.flatten().shape)
    return ccv_hist.flatten()

# ---------------------------------------------------- Load dataset here and define images and labels
# ---------------------------------------------------- Directory where your dataset is stored
dataset_dir = "eat_drinks/"

# ---------------------------------------------------- List to store images and labels
images = []
labels = []

# ---------------------------------------------------- Iterate through each subdirectory (assuming each subdirectory represents a class)
for class_dir in os.listdir(dataset_dir):
    class_label = class_dir  # Assuming class directory name is the label
    class_dir_path = os.path.join(dataset_dir, class_dir)
    
    # ---------------------------------------------------- Iterate through each image in the class directory
    for img_name in os.listdir(class_dir_path):
        img_path = os.path.join(class_dir_path, img_name)
        image = cv2.imread(img_path)
        
        # ---------------------------------------------------- If image is loaded successfully
        if image is not None:
            images.append(image)
            labels.append(class_label)


# Define label mapping for classes A to L
# label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11}

# Convert labels to numeric format using label_mapping
label_mapping = {label: index for index, label in enumerate(sorted(set(labels)))}
# print(label_mapping)
labels = [label_mapping[label] for label in labels]

# print(labels)
# print(label_mapping)
# print(images)





# --------------------------------------------- Extract CCV features for all images
ccv_features = []
class_labels = []
for image, label in zip(images, labels):
    ccv_feature = calculate_ccv(image)
    ccv_features.append(ccv_feature)
    # class_labels.append(label_mapping[label])
ccv_features = np.array(ccv_features)
# class_labels = np.array(class_labels)
print("ccv_featuress shape:",ccv_features.shape)
# print(len(labels))

# ---------------------------------------------Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(ccv_features, labels, test_size=0.25, random_state=42)

# # ---------------------------------------------SVM Model
# # Train SVM classifier
# svm_classifier = SVC(kernel='linear')
# svm_classifier.fit(X_train, y_train)

# # Predict labels for test set
# y_pred = svm_classifier.predict(X_test)

# # Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("SVM Accuracy:", accuracy)




# ================================================================================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Instantiate models
logistic_regression = LogisticRegression()
decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier(n_estimators=120)
svm = SVC(degree=5)

# Train models
logistic_regression.fit(X_train, y_train)
decision_tree.fit(X_train, y_train)
random_forest.fit(X_train, y_train)
svm.fit(X_train, y_train)

# Predictions
y_pred_lr = logistic_regression.predict(X_test)
y_pred_dt = decision_tree.predict(X_test)
y_pred_rf = random_forest.predict(X_test)
y_pred_svm = svm.predict(X_test)

# Evaluate models
accuracy_lr = accuracy_score(y_test, y_pred_lr)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

print("Logistic Regression Accuracy:", accuracy_lr)
print("Decision Tree Accuracy:", accuracy_dt)
print("Random Forest Accuracy:", accuracy_rf)
print("Support Vector Machine Accuracy:", accuracy_svm)

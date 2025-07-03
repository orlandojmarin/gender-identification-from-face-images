# Gender Classification Using Facial Landmark Ratios

This project explores gender classification using geometric features extracted from facial landmarks in the AR Face Database. By computing facial ratios based on landmark coordinates, we trained and evaluated six machine learning classifiers to determine the gender of individuals in the dataset.

## Team Members

- **Orlando Marin**  
  Led the data cleaning and feature engineering processes. Implemented the K-Nearest Neighbors (KNN) and Random Forest classifiers. Designed and implemented the gender-balanced, person-wise train/test split. Served as the primary author of the final report.

- **Tatiana Eng**  
  Implemented the Artificial Neural Network (ANN) and Naive Bayes classifiers. Created visualizations including ROC curves and confusion matrices. Designed the presentation slides with input from team members.

- **Matt Kilmer**  
  Implemented and tuned the Decision Tree and Support Vector Machine (SVM) classifiers. Collaborated on the design of the presentation slides.

## Dataset

We used a subset of the publicly available AR Face Database, which contains frontal face images of 136 individuals (76 men and 60 women). Each image is annotated with 22 facial landmark points. The dataset includes variations in expression and lighting but maintains a consistent frontal view and landmark layout.

File naming convention:
- `m-xx-yy.pts` for male subjects
- `w-xx-yy.pts` for female subjects  
Where `xx` is the person ID and `yy` denotes expression or lighting condition.

## Feature Engineering

Seven geometric features were extracted using Euclidean distances between specific facial landmark points. These features were defined according to the project guidelines:

1. **Eye Length Ratio**: Length of the longer eye divided by the distance between points 8 and 13.
2. **Eye Distance Ratio**: Distance between eye centers divided by the distance between points 8 and 13.
3. **Nose Ratio**: Distance between points 15 and 16 divided by the distance between points 20 and 21.
4. **Lip Size Ratio**: Distance between points 2 and 3 divided by the distance between points 17 and 18.
5. **Lip Length Ratio**: Distance between points 2 and 3 divided by the distance between points 20 and 21.
6. **Eyebrow Length Ratio**: Longer of the distances between (4,5) or (6,7) divided by the distance between points 8 and 13.
7. **Aggressive Ratio**: Distance between points 10 and 19 divided by the distance between points 20 and 21.

Z-score normalization was applied to features used in models sensitive to feature scale.

## Train-Test Split

An 80/20 split was applied using a gender-balanced, person-wise strategy. Individuals were assigned exclusively to either the training or test set based on their unique person ID, ensuring no data leakage. Stratification preserved equal gender representation across both sets.

## Classifiers and Tuning

Six classifiers were tested:

- **K-Nearest Neighbors (KNN)**: Best performance at k = 18.
- **Random Forest**: Optimal with 30 estimators and max depth of 15.
- **Artificial Neural Network (ANN)**: Best configuration included one hidden layer of 100 neurons, learning rate = 0.01, and tolerance = 0.01.
- **Naive Bayes**: Used Gaussian Naive Bayes without tuning.
- **Decision Tree**: Optimal with max_depth = 8, min_samples_split = 2, min_samples_leaf = 5.
- **Support Vector Machine (SVM)**: Best results with RBF kernel and C = 1.

## Evaluation Metrics

Performance was evaluated using the following metrics:

- **Accuracy**: Overall prediction correctness
- **Precision**: Proportion of positive predictions that are correct
- **Recall**: Proportion of actual positives that are correctly identified
- **AUC**: Area under the ROC curve, representing separability between classes
- **Confusion Matrix**: Breakdown of true/false positives and negatives

## Results Summary

The best-performing model was Random Forest, followed closely by Artificial Neural Network. Full metric results are summarized below:

- **Random Forest**: Accuracy = 77.27%, Precision = 76.00%, Recall = 89.06%, AUC = 0.7362
- **ANN**: Accuracy = 76.36%, Precision = 75.68%, Recall = 87.50%, AUC = 0.7741
- **Naive Bayes**: Accuracy = 72.73%, Precision = 75.00%, Recall = 79.69%, AUC = 0.7738
- **KNN**: Accuracy = 71.82%, Precision = 73.24%, Recall = 81.25%, AUC = 0.7215
- **SVM**: Accuracy = 70.91%, Precision = 69.51%, Recall = 89.06%, AUC = 0.7446
- **Decision Tree**: Accuracy = 63.64%, Precision = 67.65%, Recall = 71.88%, AUC = 0.6929

## Conclusion

This study shows that gender classification can be effectively performed using engineered geometric features derived from facial landmarks. The Random Forest and Artificial Neural Network classifiers produced the strongest results, demonstrating their ability to model complex patterns in facial structure. KNN, Naive Bayes, and SVM also performed well, especially in terms of recall. The person-wise data split and careful feature selection contributed to the consistent performance across models.

## Limitations

The dataset contains a moderate class imbalance, with more male images than female. Although our person-wise and gender-balanced splitting ensured that there was no data leakage, it may still influence model outcomes. Additionally, the landmark data includes only 22 points per face, which limits facial detail. The lack of demographic metadata such as age or ethnicity further reduces the model's ability to generalize.

## Future Work

Future improvements may include expanding the dataset to achieve better gender balance and incorporating more facial landmarks to increase detail. Additional features such as facial symmetry and angles could enhance prediction accuracy. Including metadata such as age group or skin tone may also improve model performance and fairness in real-world applications.

## How to Run

1. Navigate to the project notebook on GitHub.
2. Click the **"Open in Colab"** button at the top of the notebook (or use the Google Colab icon if available).
3. In Google Colab, click **"Runtime" → "Run all"** to execute all code cells from top to bottom.
4. The notebook will run all data processing, modeling, and evaluation steps automatically.

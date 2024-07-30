# Final-Project-guvi
# 1. Bank Risk Controller Systems

## Overview
This project involves developing a machine learning model to predict loan defaults using various classification algorithms. The model is built using a cleaned dataset and includes features for Exploratory Data Analysis (EDA), model training, and prediction through a Streamlit web application.

## Features
   Data Display: View the dataset used for model building and performance metrics.
   
   EDA: Perform and visualize Exploratory Data Analysis on the dataset.
   
   Prediction: Predict whether a customer will default on a loan based on user input.

# Project Structure
      
      ├── data
      │   ├── cleaned_data.csv
      ├── models
      │   ├── DecisionTreeClassifier_model.pkl
      │   ├── LogisticRegression_model.pkl
      │   ├── RandomForestClassifier_model.pkl
      │   ├── GradientBoostingClassifier_model.pkl
      ├── eda
      │   ├── eda_plots.py
      ├── notebooks
      │   ├── EDA.ipynb
      │   ├── Model_Training.ipynb
      ├── app
      │   ├── app.py
      ├── README.md
      ├── requirements.txt

# Installation
## Clone the repository:

      git clone https://github.com/your_username/loan-default-prediction.git
      cd loan-default-prediction
## Create and activate a virtual environment:

      python -m venv venv
      source venv/bin/activate  # On Windows use `venv\Scripts\activate`
      
## Install the required packages:
      
      pip install -r requirements.txt
      
## Run the Streamlit application:

      streamlit run app.py
## Usage
### Data Display
   View the entire dataset.
   
   View model performance metrics including ROC AUC, Precision, Recall, F1 Score, Accuracy, Log Loss, and Confusion Matrix.
### EDA
   Refresh button to generate and visualize various plots based on numerical, categorical, and binary features.
   
   Numerical Features: Distribution plots and correlation matrix.
   
   Categorical Features: Count plots and bar charts.
   
   Binary Features: Count plots.
   
### Prediction
   Input values for various features to predict loan default.
   
   Display the prediction result and probability of default.

## Model Metrics
     ## Model Metrics

      | Model                         | ROC AUC | Precision | Recall | F1 Score | Accuracy | Log Loss | Confusion Matrix                  |
      |-------------------------------|---------|-----------|--------|----------|----------|----------|------------------------------------|
      | Decision Tree                 | 0.9755  | 0.9449    | 0.9562 | 0.9505   | 0.9914   | 0.3094   | [[257003, 1359], [1068, 23311]]   |
      | Logistic Regression           | 0.6862  | 0.6786    | 0.0008 | 0.0016   | 0.9138   | 0.2756   | [[258353, 9], [24360, 19]]        |
      | Random Forest Classifier      | 0.9986  | 1.0000    | 0.9481 | 0.9734   | 0.9955   | 0.0237   | [[258362, 0], [1265, 23114]]      |
      | Gradient Boosting Classifier  | 0.7120  | 0.7023    | 0.0038 | 0.0075   | 0.9140   | 0.2699   | [[258323, 39], [24287, 92]]       |
      
## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.


-------------------------------------------------------------RECOMMENDATION SYSTEM----------------------------------------------

# 2.Recommendation-system

## Overview
This project is an E-commerce recommendation system built using Python and Streamlit. The application allows users to select products, add them to a cart, and receive product recommendations based on the items in their cart. The system uses the Apriori algorithm for generating association rules from the dataset.

## Features
  Product selection and quantity input
  
  Add products to the cart
  
  View cart with options to increase or decrease quantities or remove items
  
  Receive recommendations based on cart contents
  
  Recommendations are generated using the Apriori algorithm
  
  User-friendly interface built with Streamlit

## Clone the repository:

    git clone https://github.com/yourusername/repo-name.git
    cd repo-name
    
## Create and activate a virtual environment:

    python -m venv env
    source env/bin/activate   # On Windows use `env\Scripts\activate`

## Install the required dependencies:

    pip install -r requirements.txt
    
Run the Streamlit app:

    streamlit run app.py

## Usage
### Start the application:
  Run the Streamlit app as described above.

### Select a product and quantity:
  Use the sidebar to select a product from the dropdown and specify the quantity.

### Add to cart:
  Click the "Add to Cart" button to add the selected product and quantity to the cart.

### View and manage cart:
  The cart is displayed at the bottom of the main page, where you can adjust quantities or remove items.

### View recommendations:
  Recommendations based on the items in your cart are displayed on the right side of the page.

## Dataset
The dataset used in this project was obtained from Kaggle and includes the following columns:

BillNo

Itemname

Quantity

Date

Price

CustomerID

Country

## Apriori Algorithm
The Apriori algorithm is used to generate association rules from the transaction data. It identifies frequent itemsets and derives rules that highlight the relationships between items in the dataset.

## Contributing
Contributions are welcome! Please create a pull request or open an issue to discuss the changes you would like to make.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
Special thanks to the contributors of the open-source libraries and tools used in this project.
Thanks to Kaggle for providing the dataset.


## Contact
If you have any questions or suggestions, feel free to reach out:

Email: thirusaravana98@gmail.com
GitHub: https://github.com/thirumal4198

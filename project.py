import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import pickle
print(0)
# Load dataset
df = pd.read_csv('cleaned_data.csv')

# Add profile photo and name
#st.sidebar.image('profile_photo.jpg', width=150)
st.sidebar.write("THIRUMAL M")

# The rest of your Streamlit code

# Sidebar menu
st.sidebar.title('Navigation')
menu = st.sidebar.radio('Select Menu', ['Data', 'EDA - Visual', 'Prediction','Recommendation Systems'])

# Load the trained Decision Tree model
model_path = 'RFC.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file, encoding='latin1')

#load the encoder 
with open('encoder.pkl','rb') as f:
    encoder = pickle.load(f)


#load the scaler 
with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)

# Load the feature columns used during training
with open('X_train_columns.pkl', 'rb') as file:
    X_train_columns = pickle.load(file)

# Sidebar 1: Data
if menu == 'Data':
    st.title('Dataset')
    #st.write('### Full Dataset')
    

    st.write('### Model Performance Metrics')
    metrics = {
    'Random Forest Classifier': {
        'accuracy': 0.9999787043328938,
        'precision': 0.9999787043404333,
        'F1 score': 0.9999787043329157,
        'Recall': 0.9999787043328938
    },
    'Extra Tree Classifier': {
        'accuracy': 0.9955117934788376,
        'precision': 0.9955329959531547,
        'F1 score': 0.9954575894361939,
        'Recall': 0.9955117934788376
    },
    'Gradient Booster': {
        'accuracy': 0.6537769801582461,
        'precision': 0.6539503941522566,
        'F1 score': 0.6536859101312886,
        'Recall': 0.6537769801582461
    },
    'Logistic Regression': {
        'accuracy': 0.6387828945457925,
        'precision': 0.638943426443736,
        'F1 score': 0.6386858957553351,
        'Recall': 0.6387828945457925
    },
    'Decision Tree': {
        'accuracy': 0.9975626141012164,
        'precision': 0.9975742076288434,
        'F1 score': 0.9975625967629249,
        'Recall': 0.9975626141012164
    }
}

    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics).transpose().reset_index().rename(columns={'index': 'Model'})

    # Format the DataFrame to 5 decimal places
    metrics_df = metrics_df.applymap(lambda x: f'{x:.5f}' if isinstance(x, float) else x)
    st.dataframe(metrics_df)
    st.dataframe(df.head(20))



# Sidebar 2: EDA - Visual
elif menu == 'EDA - Visual':
    st.title('Exploratory Data Analysis')

    if st.button('Refresh EDA'):
        # Numerical columns
        numerical_cols = [
            'EXT_SOURCE_2', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY_x',
            'AMT_GOODS_PRICE_x', 'DAYS_EMPLOYED', 'DAYS_LAST_PHONE_CHANGE',
            'DAYS_ID_PUBLISH', 'DAYS_REGISTRATION'
        ]

        # Binary columns
        binary_cols = [
            'FLAG_PHONE', 'FLAG_WORK_PHONE', 'FLAG_MOBIL', 'FLAG_EMAIL',
            'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY'
        ]

        # Categorical columns
        categorical_cols = [
            'NAME_CONTRACT_TYPE_x', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
            'NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE',
            'WEEKDAY_APPR_PROCESS_START_x'
        ]

        # Ordinal columns
        ordinal_cols = ['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']

        # Correlation matrix for numerical features
        st.write('### Correlation Matrix')
        corr_matrix = df[numerical_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True)
        st.plotly_chart(fig)

        # Helper function to create plots in columns
        def plot_in_columns(cols, title):
            st.write(f'### {title}')
            col1, col2 = st.columns(2)
            for idx, col in enumerate(cols):
                fig_hist = px.histogram(df, x=col)
                fig_box = px.box(df, y=col)
                if idx % 2 == 0:
                    col1.plotly_chart(fig_hist)
                    col1.plotly_chart(fig_box)
                else:
                    col2.plotly_chart(fig_hist)
                    col2.plotly_chart(fig_box)

        # Numerical feature distributions
        plot_in_columns(numerical_cols, 'Distributions of Numerical Features')

        # Categorical feature distributions
        st.write('### Distributions of Categorical Features')
        col1, col2 = st.columns(2)
        for idx, col in enumerate(categorical_cols):
            fig = px.histogram(df, x=col)
            if idx % 2 == 0:
                col1.plotly_chart(fig)
            else:
                col2.plotly_chart(fig)

        # Binary feature distributions
        st.write('### Distributions of Binary Features')
        col1, col2 = st.columns(2)
        for idx, col in enumerate(binary_cols):
            fig = px.histogram(df, x=col)
            if idx % 2 == 0:
                col1.plotly_chart(fig)
            else:
                col2.plotly_chart(fig)

        # Ordinal feature distributions
        st.write('### Distributions of Ordinal Features')
        col1, col2 = st.columns(2)
        for idx, col in enumerate(ordinal_cols):
            fig = px.histogram(df, x=col)
            if idx % 2 == 0:
                col1.plotly_chart(fig)
            else:
                col2.plotly_chart(fig)

# Sidebar 3: Prediction
elif menu == 'Prediction':
    st.title('Loan Default Prediction')

    st.write('Enter the following features to predict whether a customer will default:')
    
    # Binary columns
    binary_cols = [
        'FLAG_PHONE', 'FLAG_WORK_PHONE', 'FLAG_MOBIL', 'FLAG_EMAIL',
        'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY'
    ]

    # Define columns for input
    col1, col2, col3 = st.columns(3)
    user_input = {}

    # Create a form to capture user input
    with st.form(key='prediction_form'):
        for idx, col in enumerate(df.columns):
            default_value = 0
            if col != 'TARGET':  # Assuming 'TARGET' is the target variable, not an input
                if col in binary_cols:
                    # Handle binary columns with 0 and 1 selection boxes
                    if idx % 3 == 0:
                        with col1:
                            user_input[col] = st.selectbox(col, options=[0, 1], format_func=lambda x: f"{x}")
                    elif idx % 3 == 1:
                        with col2:
                            user_input[col] = st.selectbox(col, options=[0, 1], format_func=lambda x: f"{x}")
                    else:
                        with col3:
                            user_input[col] = st.selectbox(col, options=[0, 1], format_func=lambda x: f"{x}")
                else:
                    # Handle other columns
                    if idx % 3 == 0:
                        with col1:
                            if df[col].dtype == 'object':
                                user_input[col] = st.selectbox(col, df[col].unique())
                            else:
                                default_value = float(df[col].iloc[default_value])
                                if col == 'EXT_source2':
                                    user_input[col] = st.number_input(col, value=default_value, format="%.5f")
                                elif col in ['DAYS_ID_PUBLISH', 'DAYS_REGISTRATION', 'DAYS_LAST_PHONE_CHANGE', 'DAYS_BIRTH']:
                                    user_input[col] = st.number_input(col, value=default_value, format="%.0f")
                                else:
                                    user_input[col] = st.number_input(col, value=default_value)
                    elif idx % 3 == 1:
                        with col2:
                            if df[col].dtype == 'object':
                                user_input[col] = st.selectbox(col, df[col].unique())
                            else:
                                default_value = float(df[col].iloc[default_value])
                                if col == 'EXT_source2':
                                    user_input[col] = st.number_input(col, value=default_value, format="%.5f")
                                elif col in ['DAYS_ID_PUBLISH', 'DAYS_REGISTRATION', 'DAYS_LAST_PHONE_CHANGE', 'DAYS_BIRTH']:
                                    user_input[col] = st.number_input(col, value=default_value, format="%.0f")
                                else:
                                    user_input[col] = st.number_input(col, value=default_value)
                    else:
                        with col3:
                            if df[col].dtype == 'object':
                                user_input[col] = st.selectbox(col, df[col].unique())
                            else:
                                default_value = float(df[col].iloc[default_value])
                                if col == 'EXT_source2':
                                    user_input[col] = st.number_input(col, value=default_value, format="%.5f")
                                elif col in ['DAYS_ID_PUBLISH', 'DAYS_REGISTRATION', 'DAYS_LAST_PHONE_CHANGE', 'DAYS_BIRTH']:
                                    user_input[col] = st.number_input(col, value=default_value, format="%.0f")
                                else:
                                    user_input[col] = st.number_input(col, value=default_value)

        # Submit button for the form
        submit_button = st.form_submit_button(label='Predict')



    if submit_button:
        # Convert user input to dataframe
        input_df = pd.DataFrame([user_input])

        # List of categorical columns
        categorical_cols = ['NAME_CONTRACT_TYPE_x', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY','NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE',
       'WEEKDAY_APPR_PROCESS_START_x']
        
        # Fit and transform the categorical columns
        encoded_cols = encoder.transform(input_df[categorical_cols])
        # Convert to DataFrame
        encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))
        
        # Drop the original categorical columns
        input_df.drop(categorical_cols, axis=1, inplace=True)
        
        # Concatenate the encoded DataFrame with the original DataFrame
        input_df = pd.concat([input_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

        columns_to_drop =  ['CNT_FAM_MEMBERS', 'AMT_CREDIT_x']
        input_df = input_df.drop(columns=columns_to_drop)

        # One-hot encode categorical columns
        numerical_cols = input_df.select_dtypes(include=['int64','float64']).columns
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        # Align input dataframe with training dataframe columns
        missing_cols = set(X_train_columns) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0
        input_df = input_df[X_train_columns]

        # Predict
        #st.dataframe(input_df.head())
        prediction = model.predict(input_df)
        st.write(prediction)
        prediction_proba = model.predict_proba(input_df)[:, 1]

        if prediction[0] == 1:
            st.markdown(f'<div style="background-color:tomato;color:white;padding:30px;">The customer is predicted to default with a probability of {prediction_proba[0]:.2f}.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background-color:lightgreen;color:black;padding:30px;">The customer is predicted not to default with a probability of {prediction_proba[0]:.2f}.</div>', unsafe_allow_html=True)

# Sidebar 4: Recommendation systems
if menu == 'Recommendation Systems':

    # Load the pickled data
    with open('RC_cleaned_data.pkl', 'rb') as f:
        data = pickle.load(f)

    with open('product_rec.pkl', 'rb') as f:
        product_rec = pickle.load(f)

    # Function to get unique recommendations
    def get_recommendations(cart_products, rules_df):
        recommended_products = []
        for item in cart_products:
            recommendations = rules_df[rules_df['Product'].apply(lambda x: item in x)]
            unique_recommendations = pd.Series([rec for sublist in recommendations['Recom'] for rec in sublist]).drop_duplicates().reset_index(drop=True)
            recommended_products.extend(unique_recommendations)
        return list(pd.Series(recommended_products).drop_duplicates().reset_index(drop=True))[:8]  # Return max 8 unique recommendations

    # Streamlit app layout
    st.title("E-commerce Recommendation System")

    # Initialize session state for cart and recommended selections
    if 'cart' not in st.session_state:
        st.session_state.cart = []
    if 'recommended_selections' not in st.session_state:
        st.session_state.recommended_selections = []

    # Create a two-column layout
    col1, col2 = st.columns(2)

    # Product selection in the left column
    with col1:
        st.header("Select a Product")
        product_list = [item for sublist in product_rec['Product'] for item in sublist]
        product_list = list(set(product_list))  # Unique products
        selected_product = st.selectbox("Select a Product", product_list)
        quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
        if st.button("Add to Cart"):
            st.session_state.cart.append((selected_product, quantity))
            st.success(f"Added {quantity} of {selected_product} to cart")

    # Display recommendations in the right column
    with col2:
        st.header("Recommended Products")
        if st.session_state.cart:
            cart_products = [item[0] for item in st.session_state.cart]
            recommended_products = get_recommendations(cart_products, product_rec)

            if recommended_products:
                st.write("Based on your cart items, we recommend the following products:")
                for i, product in enumerate(recommended_products):
                    if st.checkbox(product, key=f'recommend_{i}'):
                        if product not in st.session_state.recommended_selections:
                            st.session_state.recommended_selections.append(product)
                    else:
                        if product in st.session_state.recommended_selections:
                            st.session_state.recommended_selections.remove(product)

                if st.button("Add Selected Recommendations to Cart"):
                    for product in st.session_state.recommended_selections:
                        st.session_state.cart.append((product, 1))  # Default quantity to 1
                    st.session_state.recommended_selections.clear()  # Clear selections after adding to cart
                    st.success("Selected recommended products added to cart.")
            else:
                st.write("No recommendations available.")
        else:
            st.write("Add items to your cart to see recommendations.")

    # Display cart at the bottom
    st.header("Cart")
    if st.session_state.cart:
        cart_items = pd.DataFrame(st.session_state.cart, columns=['Product', 'Quantity'])

        for i, (product, quantity) in enumerate(st.session_state.cart):
            col1, col2, col3, col4 = st.columns(4)
            col1.write(product)
            new_quantity = col2.number_input('Quantity', min_value=1, value=quantity, step=1, key=f'qty_{i}')
            if col3.button('Update', key=f'update_{i}'):
                st.session_state.cart[i] = (product, new_quantity)
                st.experimental_rerun()
            if col4.button('Remove', key=f'remove_{i}'):
                st.session_state.cart.pop(i)
                #st.experimental_rerun()

        st.table(pd.DataFrame(st.session_state.cart, columns=['Product', 'Quantity']))
    else:
        st.write('Your cart is empty.')

    # Run the app (this line should be removed since it's not needed for Streamlit to work properly)
    # if __name__ == '__main__':
    #     st.run()

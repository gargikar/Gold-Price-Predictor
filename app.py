# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from PIL import Image

# Load the gold price data
gold_data = pd.read_csv('gold_price_dataset.csv')  

# Prepare the data for model training
X = gold_data[['SPX', 'USO', 'SLV', 'EUR/USD']]  # Features
y = gold_data['GLD']   # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape,X_test.shape)



reg = RandomForestRegressor()
reg.fit(X_train,y_train)
pred = reg.predict(X_test)     # Predict the gold prices
score = r2_score(y_test,pred)  # Calculate R2 score


# Create a Streamlit web app
def app():
    # Set the app title
    st.title('Gold Price Predictor')
    img = Image.open('img.jpg')
    st.image(img,width=100,use_column_width=True)

    #for sidebar
    st.sidebar.title("Gold Price Prediction Project")
    st.sidebar.subheader("Made by [Gargi Kar]")
    st.sidebar.write("Connect with me :")
    st.sidebar.write("[LinkedIn](https://www.linkedin.com/in/gargi-kar-4a236a201/)")
    st.sidebar.write("[Github](https://github.com/gargikar/)")
    
    
    #for user input
    st.title('User Input')
    spx = st.number_input('SPX', value=X['SPX'].mean())
    uso = st.number_input('USO', value=X['USO'].mean())
    slv = st.number_input('SLV', value=X['SLV'].mean())
    eurusd = st.number_input('EUR/USD', value=X['EUR/USD'].mean())

    # Predict the gold price for the given input
    price_pred = reg.predict([[spx, uso, slv, eurusd]])
    
    # Display the predicted gold price
    st.subheader('Predicted Gold Price : ${:.2f}'.format(price_pred[0]))
    
    # Display the R2 score
    st.subheader('Model Performance : {:.2f}'.format(score))
   

# Run the app
if __name__ == '__main__':
    app()





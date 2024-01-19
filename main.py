import gradio as gr
import pandas as pd
from io import BytesIO
import keras
import joblib
import numpy as np

'''important variable'''
MAXPOLL = 500
MINPOLL = 0
columns = (['pollution', 'dew', 'temp', 'press', 
            "wnd_dir", 'wnd_spd', 'snow', 'rain'])

# import model
model = keras.models.load_model('model_LSTM')

# preprocessing utilities
mapping = joblib.load('mapping.joblib')
scaler = joblib.load('scaler.joblib')


'''main code'''
def production(input_csv):
    
    # Read the CSV file from binary content
    input_data = BytesIO(input_csv)
    
    x = pd.read_csv(input_data)

    # check if shape is correct
    if x.shape[0] != 12:
        return (-1)

    # encoding wind direction
    x['wnd_dir'] = x['wnd_dir'].map(mapping)

    x = x[columns]
    x[columns] = scaler.transform(x[columns])

    # creating look-back window
    x = np.array(x)

    n_future = 1
    n_past = 11

    #  Test Sets
    X = []
    y = []
    for i in range(n_past, len(x) - n_future+1):
        X.append(x[i - n_past:i, 1:x.shape[1]])
        y.append(x[i + n_future - 1:i + n_future, 0])
    X_test, y_test = np.array(X), np.array(y)

    # print('X_test shape : {}, y_test shape : {}'.format(X_test.shape, y_test.shape))

    test_predictions = model.predict(X_test).flatten()
    test_results = pd.DataFrame(data={'Train Predictions': test_predictions, 'Actual':y_test.flatten()})
    # print(test_results.head())

    pred = test_predictions * (MAXPOLL - MINPOLL) + MINPOLL 
    act = y_test.flatten() * (MAXPOLL - MINPOLL) + MINPOLL
    print('PREDICTED: ', pred, 'ACTUAL: ', act)

    return str(f'Actual: {int(act)}, Predicted: {int(pred)}')


iface = gr.Interface(
    fn=production,
    inputs=gr.File(type="binary", label="Upload last 11 days of pollution data in csv format"),
    outputs= ['text'],
)

iface.launch() 






















'''RESIDUE'''

# import gradio as gr

# # imputs: date, dew, temperature, press, wind speed, snow, rain, pollution yesterday
# # output: predicted pollution

# import joblib
# from sklearn.preprocessing import MinMaxScaler
# import numpy as np
# from tensorflow import keras

# def predict_pollution(date, dew, temperature, press, wind_speed, snow, rain, pollution_yesterday):
#     # Load the trained regressor model using joblib
#     regressor = joblib.load('lstm')  # Update the path accordingly
    
#     # Prepare input features for prediction
#     input_features = np.array([[dew, temperature, press, wind_speed, snow, rain, pollution_yesterday]])
    
#     # Feature scaling
#     sc = MinMaxScaler(feature_range=(0, 1))
#     input_features_scaled = sc.fit_transform(input_features)
    
#     # Reshape input for LSTM model
#     input_features_reshaped = np.reshape(input_features_scaled, (input_features_scaled.shape[0], input_features_scaled.shape[1], 1))
    
#     # Make prediction
#     predicted_pollution_scaled = regressor.predict(input_features_reshaped)
    
#     # Inverse transform to get the original scale
#     predicted_pollution = sc.inverse_transform(predicted_pollution_scaled)
    
#     return predicted_pollution[0][0]



# # def predict_pollution(date, dew, temperature, press, wind_speed, snow, rain, pollution_yesterday):
# #     return int( date+ dew+ temperature+ press+ wind_speed+ snow+ rain+ pollution_yesterday)

# production = gr.Interface(
#     fn=predict_pollution,
#     inputs=["text", "text", "text", "text", "text", "text", "text", "text"],
#     outputs=["text"],
#     examples= [
#         ['02-01-2010', '-8.5', '-5.125', '1024.75', '24.86', '0.708333333', '0', '10.04166667']
#     ],
#     # description= "Predict pollution",
#     theme= 'darkhuggingface',
#     title='Pollution Prediction Model',
#     allow_flagging= 'never'
# )

# production.launch()
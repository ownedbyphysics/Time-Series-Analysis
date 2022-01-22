import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
sns.set(style='whitegrid', palette=("Set1"), font_scale=1.6, 
        rc={"figure.figsize": [15, 8]})
import pmdarima as pm
import pickle

df = pd.read_csv('airline_passengers.csv')

arima_settings = {
            'data' : pd.read_csv('airline_passengers.csv'),
            'stationarity_technique' : 'log_difference',   #['log', 'log', 'difference', 'log_difference']
            'target_column' : 'Passengers',
            }
            
class ARIMA_models():
    """Generic module that converts transform the timeseries to stationary data
       using different techniques, builds via grid search the best possible 
       ARIMA model for the input data, validates/saves/loads the model and use
       it to make prediction for given future timesteps
  
    Args:
        settings: an initialization dictionary of the form:
        >>> arima_settings = {
        ...     'data': df,
        ...     'stationarity_technique': 'stationarity_technique name' #['raw_data', 'log', 'difference', 'log_difference'],
        ...     'target_column': 'column name',
        ... }
    
    Returns:
        Αn array with predicted values for the corresponding timesteps ahead set by the user
        
    Example:
         >>> ar = ARIMA_models(arima_settings)
         >>> ar.stationarity() 
         >>> ar.build_model() 
         >>> ar.validate() 
         >>> ar.visualize() 
         >>> ar.save_model() 
         >>> ar.load_model() 
         >>> ar.predict(int)   
    """
    def __init__(self):
        self.settings = arima_settings
        self.data = self.settings['data']
        self.target = self.settings['target_column']
        self.stationarity_technique = self.settings['stationarity_technique']
        try:
            if not isinstance(self.data, pd.DataFrame):
                raise AttributeError("the given data is not a dataframe object")
            if self.data.empty:
                raise AttributeError("the given data is empty")
            if self.settings['target_column'] not in self.data.columns:
                raise AttributeError("the given dataframe does not contain a \
                                column named:", self.settings['target_column'])
        except Exception as other_exception:
            print(other_exception)
        
    def stationarity(self):
        """Converts the timeseries to a stationary form"""
        if self.stationarity_technique == 'log':
            self.data['Log' + self.target] = np.log(self.data[self.target])
        elif self.stationarity_technique == 'difference':
            self.data['Diff' + self.target] = self.data[self.target].diff()
        elif self.stationarity_technique == 'log_difference':
            self.data['DiffLog' + self.target] = np.log(self.data[self.target]).diff()
         
        self.stationary_target = self.data.iloc[:,-1:].columns[0]
            
    def build_model(self, train_timesteps = 12):
        self.train_timesteps = train_timesteps 
        self.train = self.data.iloc[:-train_timesteps]
        self.test = self.data.iloc[-train_timesteps:]
        self.model = pm.auto_arima(self.train[self.stationary_target].dropna(),
                      trace=True,
                      suppress_warnings=True,
                      seasonal=True, m=6)
        
    def rmse(self, t, y):
        return np.sqrt(np.mean((t - y)**2))
    
    def inverse_transform(self, result_array):
        """This method allows the stationary predicted data
        to be converted back to the original scale values"""
        if self.stationarity_technique == 'log':
            result_array = np.exp(result_array)
        elif self.stationarity_technique == 'difference':
            result_array[0] = self.test[self.target].iloc[0]
            result_array = result_array.cumsum()
        elif self.stationarity_technique == 'log_difference':
            result_array[0] = np.log(self.test[self.target].iloc[0])
            result_array = np.exp(result_array.cumsum())
            
        return result_array
   
    def validate(self):    
        self.test_pred, self.confint = self.model.predict(n_periods=self.train_timesteps, 
                                                           return_conf_int=True)
        
        self.test_pred = self.inverse_transform(self.test_pred)
    
        return self.rmse(self.test[self.target], self.test_pred)
    
    def visualize(self):
        """Autoregressive methods need also a visualization for validation purposes
        as the selected metric (ie rmse) may not be a valid indicator alone"""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.test.index, self.test[self.target], label='data')
        ax.plot(self.test.index, self.test_pred, label='forecast')
        ax.fill_between(self.test.index, \
                        self.confint[:,0], self.confint[:,1], \
                        color='red', alpha=0.3)
        ax.legend();
    
    def save_model(self):
        with open('myModel','wb') as f:
            pickle.dump(self.model,f)
        
    def load_model(self):
        with open('myModel','rb') as f:
            model = pickle.load(f)
        self.model = model
          
    def predict(self, timesteps_ahead = 7):    
        test_pred, confint = self.model.predict(n_periods=timesteps_ahead, 
                                           return_conf_int=True)
        
        test_pred = self.inverse_transform(test_pred)
        return test_pred, confint
            
            
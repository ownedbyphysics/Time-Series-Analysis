import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
sns.set(style='whitegrid', palette=("Set1"), font_scale=1.6, rc={"figure.figsize": [15, 8]})


df =  pd.read_csv('orders_with_products.csv', index_col = 0)
df['Sales'] = df['Price'] * df['Qty']



ETS_settings = {
            'data' : df.copy(),
            'target_column' : 'Sales',
            'days': 20,
            'alpha': 0.2,
            'model_type':['additive'],  # additive or multiplicative
            'period': 30
          }

class ETS_models():
    """Moving average and ETS components
  
    Args:
        settings: an initialization dictionary of the form:
        >>> ETS_settings = {
        ...     'data': df,
        ...     'days': integer, number of days to be taken account of, for moving average calculations,
        ...     'alpha': float, decay factor
        ...     'target_column': 'column name',
        ...     'model_type': depends on trend  # additive or multiplicative
        ...     'period': integer, number of days for forecasting into the future
        ... }
        
    Returns:
        dict: the predicted values of the input period of days ahead
  
      To use:
     >>> ets = ETS_models(ETS_settings)
     >>> df, sma = ets.SMA()
     >>> df2, wmma = ets.EWMA()
     >>> ets.decomposition()
     >>> ets.decomposition()
     >>> ets.forecast()  
     """
    
    def __init__(self, ETS_settings):
        """Inits ETS models with a specific settings dictionary:
            data: pd.DataFrame
            alpha: int, between 0 and 1
         """
        self.settings = ETS_settings
        self.data = self.settings['data']
        self.target = self.settings['target_column']
        self.days = self.settings['days']
        self.alpha = self.settings['alpha']
        self.model_type = self.settings['model_type'][0]
        self.period = self.settings['period']
        
    def SMA(self):
        """
        Computes and returns the simple moving average of the series 
        Returns a dataframe and a plot
        """ 
        data = self.settings['data'].copy()
        target = self.settings['target_column']
        data['SMA-'+str(self.days)] = data[target].rolling(self.days).mean()
        return data, data.plot(figsize=(10, 5));
    
    def EWMA(self):
        """
        Computes the expontentially-weighted moving average of the series 
        Returns a dataframe and a plot
        """
        target = self.settings['target_column']
        data = self.settings['data'].copy()  
        ewma_values=[]
        for value in data[target].values:
            if len(ewma_values)>0:
                xhat = self.alpha*value + (1-self.alpha)*ewma_values[-1]
            else:
                xhat = value
            ewma_values.append(xhat)
        data['EWMA'] = ewma_values
        return data, data.plot(figsize=(10, 5));
    
    def decomposition(self):
        """
        Decomposes the series into trend, seasonal, and residual components assuming the model type
        Returns a plot object
        """
        seasonal_decompose(self.data[self.target], model=self.model_type, period=self.period).plot()
        
    def forecast(self, days_ahead = 12):
        """
        Function to perform forecasting 
        for the the days ahead set by the used 

        Parameters: 
        days_ahead: int
        
        Returns:
        a_dict (dict of str: int/float)

        Example:
        >>> Forecasting(settings).forecast(7)
        """
        data = self.data
        train =  data.iloc[:data.shape[0]-days_ahead]
        test = data.iloc[train.shape[0]:]

        model = ExponentialSmoothing(train[self.settings['target_column']], 
                            trend = self.model_type,
                             seasonal = self.model_type,
                             #seasonal_periods = seasonal_periods,
                             damped_trend = False,
                            ).fit()
        
        predictions = model.forecast(days_ahead)
        predictions_dict = round(predictions,0).to_dict()
        
        train[self.settings['target_column']].plot(legend = True, label = 'Train')
        test[self.settings['target_column']].plot(legend = True, label = 'Test')
        round(predictions,0).plot(legend = True, label = 'preds')  # xlim = ['2020-04-22', '2020-05-21'] ,ylim = (0,500))

        return predictions_dict
            
        
    
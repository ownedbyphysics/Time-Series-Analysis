import pandas as pd
import seaborn as sns
sns.set(style='whitegrid', palette=("Set1"), font_scale=1.6, rc={"figure.figsize": [15, 8]})

df =  pd.read_csv('orders_with_products.csv', index_col = 0)
df['Sales'] = df['Price'] * df['Qty']

timeseries_settings = {
            'data' : df,
            'date_column': 'Date',
            'period' : 'day', #['day', 'week', 'month', 'year']
            'target_column' : 'Sales',
            'fillna': 'lastValue', #['mean', 'meanLastDays', 'lastValue'],
    
            #'meanLastDays': 3
}



class ConvertToTimeseries():
    """Generic module that converts a dataframe to a timeseries dataset.
       The user can choose if the timeseries will be grouped by day, week, month, year
       
       Args:
        settings: an initialization dictionary of the form:
        >>> timeseries_settings = {
        ...     'data': df,
        ...     'date_column': Date column to be grouped by
        ...     'target_column': column name,
        ...     'fillna': option of technique to fill in the produced missing values after grouping by date, #['mean', 'meanLastDays', 'lastValue'],
        ... }
    
    Returns:
        A dataframe where Date is the index and the only column is the target column. 
        This is the suitable input format for most of timeseries predictive modules 
  
      Example:
     >>> f = ConvertToTimeseries(settings)
     >>> f.convert() 
     """
    
    def __init__(self, timeseries_settings):
        """Inits conversion with a specific settings dictionary:
            data: pd.DataFrame
            date_column: str
            'period': str
            target_column: str
            fillna: str
         """
        self.settings = timeseries_settings
        
    def convert(self):
        """
        Function to convert a given dataframe to a timeseries one. 

        Parameters: 
        -
        
        Returns:
        pd.DataFrame

        Example:
        >>> Forecasting(settings).convert_df_to_timeseries()
        """
        settings = self.settings
        data = self.settings['data']
        period = self.settings['period']
        try:
            if not isinstance(data, pd.DataFrame):
                raise AttributeError("the given data is not a dataframe object")
            if data.empty:
                raise AttributeError("the given data is empty")
            if settings['target_column'] not in data.columns:
                raise AttributeError("the given dataframe does not contain a column named:", settings['target_column'])
            if settings['date_column'] not in data.columns:
                raise AttributeError("the given dataframe does not contain a column named:", settings['date_column'])
        except Exception as other_exception:
            print(other_exception)

        data = data[[settings['date_column'], settings['target_column']]]
        data[settings['date_column']] = pd.to_datetime(data[settings['date_column']], utc=True)
        if period == 'week':
            data[settings['date_column']] = data[settings['date_column']].dt.to_period('w')
            data = data.sort_values(by=['Date'])
            data = data.groupby(settings['date_column'], as_index=True).sum()
        elif period == 'month':
            data[settings['date_column']] = data[settings['date_column']].dt.to_period('m')
            data = data.sort_values(by=['Date'])
            data = data.groupby(settings['date_column'], as_index=True).sum()
        elif period == 'year':
            data[settings['date_column']] = data[settings['date_column']].dt.to_period('y')
            data = data.sort_values(by=['Date'])
            data = data.groupby(settings['date_column'], as_index=True).sum()
        else:
            data[settings['date_column']] = data[settings['date_column']].dt.date
            data = data.groupby(settings['date_column'], as_index=True).sum()
            data.reset_index(inplace = True)
            fillDates = pd.date_range(start=data[settings['date_column']].min(), end=data[settings['date_column']].max())
            data = data.set_index('Date').reindex(fillDates).rename_axis('Date')
            
        data.reset_index(inplace = True)
        if data[settings['target_column']].isnull().any():
                   
            if settings['fillna'] == 'mean':
                data = data.fillna(round(data[settings['target_column']].mean()))
            elif settings['fillna'] == 'lastValue':
                try: 
                    data.fillna(method='ffill', inplace=True)
                except:
                    data.fillna(method='bfill', inplace=True)
                    
        data.index = data[settings['date_column']]
        data.drop(settings['date_column'], axis = 1, inplace = True)
        
        return data    

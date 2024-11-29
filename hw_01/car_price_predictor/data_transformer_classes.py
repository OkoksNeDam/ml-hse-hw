import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NameTransformer(BaseEstimator, TransformerMixin):
    """
       Трансформирует столбец name.
   """

    def __init__(self):
        self.column_name = 'name'

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['name'] = X['name'].apply(lambda x: x.split()[0])
        return X


class TorqueTransformer(BaseEstimator, TransformerMixin):
    """
       Трансформирует столбец torque.
   """

    def fit(self, X, y=None):
        return self

    def parse_torque_column(self, x):
        """
         Парсим столбец torque, чтобы получить два значения: torque и rpm.
         Если для каких-то значений попался интервал, то берем среднее.
         Значение torque будет иметь единицы измерения Nm: Nm = 9.8 * kgm.
        """
        if not x:
            return x
        x = ' '.join(x.split())
        x = x.lower()
        x = x.replace(',', '')
        found_numbers = re.compile(r'[-+]?(?:\d*\.*\d+)').findall(x)
        # Пример: 190Nm@ 2000rpm
        if re.compile(r'[-+]?(?:\d*\.*\d+)nm@ [-+]?(?:\d*\.*\d+)rpm').fullmatch(x):
            return list(map(float, found_numbers))
        # Пример: 250Nm@ 1500-2500rpm
        if re.compile(r'[-+]?(?:\d*\.*\d+)nm@ [-+]?(?:\d*\.*\d+)[-|~][-+]?(?:\d*\.*\d+)rpm').fullmatch(x):
            return [float(found_numbers[0]), (float(found_numbers[1]) + abs(float(found_numbers[2]))) / 2]
        # Пример: 22.4 kgm at 1750-2750rpm
        if re.compile(r'[-+]?(?:\d*\.*\d+)\s?kgm at [-+]?(?:\d*\.*\d+)[-|~][-+]?(?:\d*\.*\d+)\s?rpm').fullmatch(x):
            return [float(found_numbers[0]) * 9.8, (float(found_numbers[1]) + abs(float(found_numbers[2]))) / 2]
        # Пример: 7.8@ 4,500(kgm@ rpm)
        if re.compile(r'[-+]?(?:\d*\.*\d+)@ [-+]?(?:\d*\.*\d+)\(kgm@ rpm\)').fullmatch(x):
            return [float(found_numbers[0]) * 9.8, float(found_numbers[1])]
        # Пример: 13.1kgm@ 4600rpm
        if re.compile(r'[-+]?(?:\d*\.*\d+)kgm@ [-+]?(?:\d*\.*\d+)rpm').fullmatch(x):
            return [float(found_numbers[0]) * 9.8, float(found_numbers[1])]
        # Пример: 200Nm
        if re.compile(r'[-+]?(?:\d*\.*\d+)nm').fullmatch(x):
            return [float(found_numbers[0]), None]
        # Пример: 14.9 KGM at 3000 RPM
        if re.compile(r'[-+]?(?:\d*\.*\d+)\s?kgm at [-+]?(?:\d*\.*\d+)\s?rpm').fullmatch(x):
            return [float(found_numbers[0]) * 9.8, float(found_numbers[1])]
        # Пример: 48@ 3,000+/-500(NM@ rpm)
        if re.compile(r'[-+]?(?:\d*\.*\d+)@ [-+]?(?:\d*\.*\d+)\+/-[-+]?(?:\d*\.*\d+)\(nm@ rpm\)').fullmatch(x):
            return [float(found_numbers[0]), float(found_numbers[1])]
        # Пример: 380Nm(38.7kgm)@ 2500rpm
        if re.compile(r'[-+]?(?:\d*\.*\d+)nm\([-+]?(?:\d*\.*\d+)kgm\)@ [-+]?(?:\d*\.*\d+)rpm').fullmatch(x):
            return [float(found_numbers[0]), float(found_numbers[2])]
        # Пример: 210 / 1900
        if re.compile(r'[-+]?(?:\d*\.*\d+) / [-+]?(?:\d*\.*\d+)').fullmatch(x):
            return [float(found_numbers[0]), float(found_numbers[1])]
        # Пример: 250@ 1250-5000rpm
        if re.compile(r'[-+]?(?:\d*\.*\d+)@ [-+]?(?:\d*\.*\d+)[-|~][-+]?(?:\d*\.*\d+)rpm').fullmatch(x):
            return [float(found_numbers[0]), (float(found_numbers[1]) + abs(float(found_numbers[2]))) / 2]
        # Пример: 250 nm at 2750 rpm
        if re.compile(r'[-+]?(?:\d*\.*\d+)\s?nm at [-+]?(?:\d*\.*\d+)\s?rpm').fullmatch(x):
            return [float(found_numbers[0]), float(found_numbers[1])]
        # Пример: 20.4@ 1400-3400(kgm@ rpm)
        if re.compile(r'[-+]?(?:\d*\.*\d+)@ [-+]?(?:\d*\.*\d+)[-|~][-+]?(?:\d*\.*\d+)\(kgm@ rpm\)').fullmatch(x):
            return [float(found_numbers[0]) * 9.8, (float(found_numbers[1]) + abs(float(found_numbers[2]))) / 2]
        # Пример: 180 nm at 1440-1500rpm
        if re.compile(r'[-+]?(?:\d*\.*\d+)\s?nm at [-+]?(?:\d*\.*\d+)[-|~][-+]?(?:\d*\.*\d+)\s?rpm').fullmatch(x):
            return [float(found_numbers[0]), (float(found_numbers[1]) + abs(float(found_numbers[2]))) / 2]
        # Пример: 51nm@ 4000+/-500rpm
        if re.compile(r'[-+]?(?:\d*\.*\d+)\s?nm@ [-+]?(?:\d*\.*\d+)\+/-[-+]?(?:\d*\.*\d+)\s?rpm').fullmatch(x):
            return [float(found_numbers[0]), float(found_numbers[1])]
        # Пример: 135.4nm@ 2500
        if re.compile(r'[-+]?(?:\d*\.*\d+)\s?nm@ [-+]?(?:\d*\.*\d+)').fullmatch(x):
            return [float(found_numbers[0]), float(found_numbers[1])]
        # Пример: 510@ 1600-2400
        if re.compile(r'[-+]?(?:\d*\.*\d+)@ [-+]?(?:\d*\.*\d+)-[-+]?(?:\d*\.*\d+)').fullmatch(x):
            return [float(found_numbers[0]), (float(found_numbers[1]) + abs(float(found_numbers[2]))) / 2]
        # Пример: 400 nm /2000 rpm
        if re.compile(r'[-+]?(?:\d*\.*\d+)\s?nm /[-+]?(?:\d*\.*\d+)\s?rpm').fullmatch(x):
            return [float(found_numbers[0]), float(found_numbers[1])]
        # Пример: 190nm@ 2000-3000
        if re.compile(r'[-+]?(?:\d*\.*\d+)\s?nm@ [-+]?(?:\d*\.*\d+)-[-+]?(?:\d*\.*\d+)').fullmatch(x):
            return [float(found_numbers[0]), (float(found_numbers[1]) + abs(float(found_numbers[2]))) / 2]
        # Пример: 110(11.2)@ 4800
        if re.compile(r'[-+]?(?:\d*\.*\d+)\([-+]?(?:\d*\.*\d+)\)@ [-+]?(?:\d*\.*\d+)').fullmatch(x):
            return [float(found_numbers[0]), float(found_numbers[2])]

    def transform(self, X):
        X = X.rename(columns={"torque": "torque_old"})

        X['torque_old'] = X['torque_old'].apply(self.parse_torque_column)
        X['torque'] = X['torque_old'].apply(lambda x: x if not x else x[0])
        X['max_torque_rpm'] = X['torque_old'].apply(lambda x: x if not x else x[1])
        X = X.drop(columns='torque_old')

        return X


class MaxPowerTransformer(BaseEstimator, TransformerMixin):
    """
       Трансформирует столбец max_power.
   """

    def __init__(self):
        self.column_name = 'max_power'

    def fit(self, X, y=None):
        return self

    def remove_unit_power(self, x):
        if not x:
            return x
        if x in ['0', ' bhp']:
            return None
        return float(x.split()[0])

    def transform(self, X):
        X = pd.DataFrame(X)
        X[self.column_name] = X[self.column_name].apply(self.remove_unit_power)
        return X


class EngineTransformer(BaseEstimator, TransformerMixin):
    """
       Трансформирует столбец engine.
   """

    def __init__(self):
        self.column_name = 'engine'

    def fit(self, X, y=None):
        return self

    def remove_unit_power(self, x):
        if not x:
            return x
        return float(x.split()[0])

    def transform(self, X):
        X = pd.DataFrame(X)
        X[self.column_name] = X[self.column_name].apply(self.remove_unit_power)
        return X


class MileageTransformer(BaseEstimator, TransformerMixin):
    """
       Трансформирует столбец mileage.
   """

    def __init__(self):
        self.column_name = 'mileage'

    def fit(self, X, y=None):
        return self

    def from_kmkg_to_kmpl(self, x):
        """
          Переводит из km/kg в kmpl следующим образом: kmpl = 1.4 * km/kg
      """
        if not x:
            return x
        value, measure = x.split()
        # Переводим из km/kg в kmpl.
        if measure == "km/kg":
            return float(value) * 1.4
        else:
            return float(value)

    def transform(self, X):
        X = pd.DataFrame(X)
        X[self.column_name] = X[self.column_name].apply(self.from_kmkg_to_kmpl)
        return X


class BaseDataTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.transformers = [MileageTransformer(), EngineTransformer(),
                             MaxPowerTransformer(), TorqueTransformer(),
                             NameTransformer()]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.replace({np.nan: None})

        for transformer in self.transformers:
            X = transformer.transform(X)
        return X


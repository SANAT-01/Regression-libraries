import numpy as np

def rmse(y_pred, y_actual):
    """Calculate the Root Mean Squared Error (RMSE) between two arrays"""
    return np.mean((y_actual - y_pred) ** 2)

def r2(y_pred, y_actual):
    """Calculate the R-squared (R2) between two arrays"""
    a = np.sum((y_actual - y_pred)**2)
    b = np.sum((y_actual - np.mean(y_actual))**2)
    r2 = 1 - (a / b)
    return r2

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def k_cross_fold(x_train, y_train, model, k):
  Rmse = float("inf")
  pre = None
  intercept = None
  prev = None
  coeff = None
  lg = len(x_train)
  split_amount = int(lg/k)
  for i in range(k):
      if i == 0:
        train_x = x_train.iloc[split_amount*(i+1) : ]
        train_y = y_train[split_amount*(i+1) : ]
        test_x = x_train.iloc[ : split_amount*(i+1)]
        test_y = y_train[ : split_amount*(i+1)]

      elif i == k-1:
        train_x = x_train.iloc[ : split_amount*(i)]
        train_y = y_train[ : split_amount*(i)]
        test_x = x_train.iloc[split_amount*(i) : ]
        test_y = y_train[split_amount*(i) : ]

      else :
        test_x = x_train.iloc[int(split_amount* (i)) : int(split_amount* (i+1))]
        test_y =  y_train[int(split_amount* (i)) : int(split_amount* (i+1))]
        train_x = x_train.iloc[ : int(split_amount* (i))]
        train_x = np.append(train_x, x_train.iloc[int(split_amount* (i+1)) : ], axis = 0)
        train_y = y_train[ : int(split_amount* (i))]
        train_y = np.append(train_y, y_train[int(split_amount* (i+1)) : ], axis = 0)
#       print(train_x.shape)
      model.fit(np.array(train_x),train_y)
      y_hat = model.predict(test_x)
      coeff = model.coef_
      intercept = model.intercept_
      print('MAPE on train data: ', mape(test_y, y_hat)**(0.5))

      if Rmse > mape(test_y, y_hat)**(0.5):
        Rmse = mape(test_y, y_hat)**(0.5)
        pre = intercept
        prev = coeff

  return Rmse, pre, prev
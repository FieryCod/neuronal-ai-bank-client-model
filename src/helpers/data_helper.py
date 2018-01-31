""" Pandas module for working with the datasets """
import pandas as pd


def get_dummied_data(data):
  """
    Data comes unprepared from the .csv file MLPClassifier
    expects only floats from categorical values

    :param data: data
    :returns: dummied_data
    :rtype: pd
  """

  job_dummied = pd.get_dummies(data['job'])
  marital_dummied = pd.get_dummies(data['marital'])
  education_dummied = pd.get_dummies(data['education'])
  default_dummied = pd.get_dummies(data['default'])
  housing_dummied = pd.get_dummies(data['housing'])
  loan_dummied = pd.get_dummies(data['loan'])
  contact_dummied = pd.get_dummies(data['contact'])
  month_dummied = pd.get_dummies(data['month'])
  poutcome_dummied = pd.get_dummies(data['poutcome'])

  return pd.concat(
    [
      job_dummied, marital_dummied, education_dummied, default_dummied,
      housing_dummied, loan_dummied, contact_dummied, month_dummied,
      poutcome_dummied
    ],
    axis=1)

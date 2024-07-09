import random
import matplotlib.colors as mcolors
import pandas as panda
import matplotlib.pyplot as plot
import numpy as pie
import math
import os.path as path
from collections import defaultdict
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_tweedie_deviance as mtd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import seaborn as sns

#Filling block
#Corrects invalid data.
def fill_with_iterated_data(c, v):
  """Returns a column with Nan data filled with iterated one.
     Input: a column with data (series), a string with additional value.
  Ex.of return: [Apple, fruit №1, orange, fruit №3, ...]
  """
  return c.where(c.notnull(), [(v+str(i)) for i in c.index], axis=0)

def fill_with_data(c, v):
  """Returns a column filled with value.
  Input: a column with data (series), a value to insert.
  """
  return c.fillna(v)

def fill_with_floor_median(c):
  """Returns a column where Nan is filled with a floored median of the column.
  Input: a column with data (series).
  """
  return c.fillna(math.floor(c.median()))

def fill_with_ceil_median(c):
  """Returns a column where Nan is filled with a ceiling median of the column.
  Input: a column with data (series).
  """
  return c.fillna(math.ceil(c.median()))

def fill_with_ceil_median_two_columns(two_c_s, be_bigger=False):
  """Returns two interconnected columns where Nan of the first column
     should be filled with either bigger or smaller value than 
     the second one contains (or be filled with a ceiled median).
  Input: two columns with data (series), condition: should value be bigger or not.
  """
  m = math.ceil(two_c_s[two_c_s.keys()[0]].median())  # a ceiled median of the second column.
  return two_c_s.apply(
      lambda row: row[two_c_s.keys()[0]]
      if (math.isnan(row[two_c_s.keys()[0]]) == False) else
         (m if (math.isnan(row[two_c_s.keys()[1]]) == True or
                check_b_smaller(m, row[two_c_s.keys()[1]]) and be_bigger == False or
                check_b_smaller(row[two_c_s.keys()[1]], m) and be_bigger == True)
      else row[two_c_s.keys()[1]]), axis = 1)

def check_b_smaller(a, b):
  """Returns true if first value is bigger than the second one; otherwise - false.
  Input: a result of the check.
  """
  return a > b

def ConcatTwoColumns(c1, c2): 
  """Returns two columns like a tuple array.
  Input: a tuple array.
  """
  return panda.concat([c1, c2], axis=1)

def fill_with_mode(с):
  """Returns a column where Nan is filled with a mode of the column.
  Input: a column with data (series).
  """
  return с.fillna(с.mode()[0])

def check_in_name(cname, columns):
  """Returns true if value in array; otherwise - false.
  Input: a certain value, a list of values.
  """
  return cname in columns

def fill_data(c_names, no_c_data):
  """Returns a table with no-Nan data (fully filled).
  Input: a list of column names, a data table.
  """
  if(check_in_name(c_names[0], no_c_data.columns)):
    no_c_data[c_names[0]] = fill_with_iterated_data(no_c_data[c_names[0]], "Квест №")
  if(check_in_name(c_names[1], no_c_data.columns) and
     check_in_name(c_names[2], no_c_data.columns)):
       no_c_data[c_names[1]] = fill_with_ceil_median_two_columns(
        ConcatTwoColumns(no_c_data[c_names[1]], no_c_data[c_names[2]]), True)
       no_c_data[c_names[2]] = fill_with_ceil_median_two_columns(
        ConcatTwoColumns(no_c_data[c_names[2]], no_c_data[c_names[1]]))
  elif (check_in_name(c_names[1], no_c_data.columns)):
    fill_with_ceil_median(no_c_data[c_names[1]])
  elif (check_in_name(c_names[2], no_c_data.columns)):
    fill_with_ceil_median(no_c_data[c_names[2]])
  if(check_in_name(c_names[3], no_c_data.columns)):
    no_c_data[c_names[3]] = fill_with_floor_median(no_c_data[c_names[3]])
  if(check_in_name(c_names[4], no_c_data.columns)):
    no_c_data[c_names[4]] = fill_with_data(no_c_data[c_names[4]], 0)
  if(check_in_name(c_names[5], no_c_data.columns)):
    no_c_data[c_names[5]] = fill_with_data(no_c_data[c_names[5]], 0)
  if(check_in_name(c_names[6], no_c_data.columns)):
    no_c_data[c_names[6]] = fill_with_ceil_median(no_c_data[c_names[6]])
  if(check_in_name(c_names[7], no_c_data.columns)):
    no_c_data[c_names[7]] = fill_with_floor_median(no_c_data[c_names[7]])
  if(check_in_name(c_names[8], no_c_data.columns)):
    no_c_data[c_names[8]] = fill_with_floor_median(no_c_data[c_names[8]])
  if(check_in_name(c_names[9], no_c_data.columns)):
    no_c_data[c_names[9]] = fill_with_mode(no_c_data[c_names[9]])
  return no_c_data

def exclusion_generator(base_list, to_excludes):
  """Returns items which is not in the list.
  Input: a list with items, names to exclude.
  """
  for item in base_list:
    if item not in to_excludes:
      yield item

def exclude_incorrect_info(data, list_string_to_null_c, not_int_c): 
  """Returns list with types applied to each column.
  Input: a data table, list of column names where string is to be Nan, 
  list of column names that are not integer.
  """
  for c in list_string_to_null_c:
    data[c] = set_string_nan(data[c])
  for с in exclusion_generator(data.columns, not_int_c):
    data[с] = set_string_nan(data[с])
    data[с] = switch_to_int(data[с])
  return data

def set_string_nan(с):
  """Returns a column with string converted to Nan if a value is not a number.
  Input: a column with data (series).
  """
  return panda.to_numeric(с, errors='coerce')

def switch_to_int(с):
  """Returns a column with number converted to integer if it is an integer.
  Input: a column with data (series).
  """
  return с.mask(с != с//1)

def delete_rows_with_na_column(data, с):
  """Returns a table without rows where a certain column contains Nan.
  Input: a data table.
  """
  return data[data[с].notna()]

def delete_column(data, columns):
  """Returns a table without named columns.
  Input: a data table, columns to exclude.
  """
  return data.drop(columns, axis=1)

def delete_uninformative_rows(data, info_n):
  """Returns a table without rows where there are less columns
     filled with data than the number states.
  Input: a data table, number of required columns.
  """
  return data.dropna(thresh=info_n)

def delete_uninformative_columns(data, percent):
  """Returns a table without columns where there are less columns
     filled with data than the percent states.
  Input: a data table, required informative percent.
  """
  return data.dropna(thresh=math.ceil(len(data.index) * percent / 100), axis=1)

def delete_empty_data(data):
  """Returns a table where all uniformative data is deleted and 
     a table without string data.
  Input: a data table.
  """
  data = delete_rows_with_na_column(data, "Стоимость")
  data = delete_uninformative_rows(data, 5)  # Characteristics №6 can be added based on other info.
  no_c_s_data = delete_column(data, "Название")
  no_c_s_data = delete_uninformative_columns(no_c_s_data, 75)
  no_c_s_data.insert(0, 'Название', data["Название"])
  return data, no_c_s_data

def if_not_enough_data(data, min_n_columns, min_n_rows, must_columns):
  """Returns true if some of conditions are not met: number of rows, 
    number of columns, required columns.
  Input: a data table, a minimum number of columns, a minimum number of rows,
  names of required columns.
  """
  return len(data.columns) < min_n_columns or\
         len(data.index) < min_n_rows or\
         not check_in_name(must_columns, data.columns)

def set_c_type(c, type_name):
    return c.astype(type_name)

def finally_convert(data, float_c_names, string_c_names):
  """Returns a data table that is fully converted to the needed type.
  Input: a data table, lists of column names where data is float, string.
  """
  if(check_list_contained(float_c_names, data.columns)):
    data[float_c_names] = set_c_type(data[float_c_names], "float32")
  if(check_list_contained(string_c_names, data.columns)):
    data[string_c_names] = set_c_type(data[string_c_names], "string")
  for column in exclusion_generator(data.columns, ("Рейтинг", "Название")):
    data[column] = set_c_type(data[column], "int")
  return data
#

#Opening File Block
#Processes data.
def write_file_name():
  """Returns a file name written by user."""
  print("Пожалуйста, введите название файла.")
  return input()

def check_file_exists(filename):
  """Returns true if file exists; otherwise - false.
  Input: a filename.
  """
  return path.exists(filename)

def do_main_read_work(filename):
  """Returns true if data can be used for training; otherwise - false;
     processed data.
  Input: a filename.
  """
  c = ["Название", "Минимальное число человек", "Максимальное число человек",\
       "Уровень сложности", "Уровень страха", "Число актёров", "Возраст",\
       "Рейтинг", "Количество команд", "Длительность", "Стоимость"]
  data = panda.read_csv(filename, sep='|', header=None, names=c)
  data = exclude_incorrect_info(data, ["Рейтинг"], ["Название", "Рейтинг"])
  data, no_c_data = delete_empty_data(data)
  if(if_not_enough_data(no_c_data, 5, 5, "Стоимость")):
    print("Недостаточно данных. Пожалуйста, проверьте, все ли данные верны.")
    return False, no_c_data
  else:
    fill_data(c, no_c_data)
  return True, finally_convert(no_c_data, ["Рейтинг"], ["Название"])
#

#Training block
#Teaches model to predict data.
def train_module(data, target_name, excluded_c_names):
  """Returns a trained model and trained/tested aspects
     (features, target train; target and predicted results).
  Input: a data table, a target column name, a list of excluded columns.
  """
  data = delete_column(data, excluded_c_names)
  features = delete_column(data, target_name)
  target = data[target_name]
  features_train, features_test, target_train, target_test = \
      tts(features, target, test_size = 0.25)
  model = LinearRegression(positive=True)
  model = model.fit(features_train, target_train)
  result = model.predict(features_test)
  return model, features_train, target_train, target_test, result
#

#Assessing block
#Checks how good trained model is.
def mape(actual, pred):
  """Returns mean absolute percentage error (MAPE),
     or mean absolute percentage deviation (MAPD).
  Input: actual and predicted values.
  """
  return pie.mean(pie.abs((actual - pred) / actual))

def get_coeff_correlation(model):
  """Outputs linear regression equation.
  Input: a trained model.
  """
  equation = f"y = {model.intercept_}"
  for i, coef in enumerate(model.coef_):
    equation += f" + ({coef} * X{i+1})"
  print(f"Уравнение линейной регрессии: {equation}.")

def get_p_value(features_train, target_train):
  """Outputs features that are most valued for the result.
  Input: features list, target data list.
  """
  print("Значимость признаков:")
  features_train = sm.add_constant(features_train)
  smmodel = sm.OLS(target_train, features_train).fit()
  if(not smmodel.pvalues.isnull().any()):
    for i, p_value in enumerate(smmodel.pvalues[1:]):
      print(f"Признак {i+1}: {p_value}")
    print(f"Наиболее значимый признак - {smmodel.pvalues.index[smmodel.pvalues == min(smmodel.pvalues)][0]}.")
  else:
    print("Недостаточно данных для анализа значимости признаков.")

def check_mse(target_train, target_test, result, false_model_result):
  """Returns two results of Mean Squared Error
     applied to real and fake model results.
  Input: target data used for training, used for testing, real, fake result.
  """
  return mse(target_test, result), mse(target_test, false_model_result)

def check_mae(target_train, target_test, result, false_model_result):
  """Returns two results of Mean Absolute Error
     applied to real and fake model results.
  Input: target data used for training, used for testing, real, fake result.
  """
  return mae(target_test, result), mae(target_test, false_model_result)

def check_mpd(target_train, target_test, result, false_model_result):
  """Returns two results of Mean Poisson Deviance
     applied to real and fake model results.
  Input: target data used for training, used for testing, real, fake result.
  """
  return mtd(target_test, result, power=1), mtd(target_test, false_model_result, power=1)

def state_for_the_record(name, short_name, check_data):
  """Outputs the result of metrics.
  Input: a metric name, a shorten metric name, 
         checked data (real and fake model results in a tuple).
  """
  print(f"Проверка расчётом {name} показала, что модель ", end='')
  if(check_data[0] > check_data[1]):
    print("не ", end='')
  print(f"справилась с задачей.\nРезультаты {short_name} модели: {check_data[0]}; для среднего значения: {check_data[1]}.")

def check_r2(target_test, result):
  """Outputs the result of coefficient of determination.
  Input: target data used for testing, real result.
  """
  r2score = r2_score(target_test, result)
  match r2score:
    case _ if r2score == 1:
      print("MSE = 0. Модель работает без ошибок.")
    case _ if r2score == 0:
      print("Модель показывает такие же результаты, как и среднее значение.")
    case _ if r2score < 0:
      print("Качество модели очень низкое. Результаты хуже, чем при расчёте среднего.")
    case _ if r2score > 0:
      print("Качество модели высокое. Результаты лучше, чем при расчёте среднего.")
    case _ if r2score > 1:
      print("Ошибка, недопустимое значение.")
    case _:
      print("Ошибка.")
  print(f"Коэффициент детерминации = {r2score}.")

def get_false_model_result(target_train, target_test):
  """Returns a fake model results (made with mean values).
  Input: target data used for training, used for testing.
  """
  return [target_train.mean()] * len(target_test)

def check_correctness(model, features_train, target_train, target_test, result):
  """Outputs whole assessment check results.
  Input: a trained model, features used for training, 
         target data used for training, used for testing, real result.
  """
  print("Проверка результатов модели:")
  get_coeff_correlation(model)
  print("*")
  get_p_value(features_train, target_train)
  print("*")
  falsemodelresult = get_false_model_result(target_train, target_test)
  state_for_the_record("cреднеквадратичной ошибки", "MSE",\
                    check_mse(target_train, target_test, result, falsemodelresult))
  print("*")
  state_for_the_record("средней абсолютной ошибки", "MAE",\
                    check_mae(target_train, target_test, result, falsemodelresult))
  print("*")
  state_for_the_record("среднего отклонения Пуассона", "MPD",\
                    check_mpd(target_train, target_test, result, falsemodelresult))
  print("*")
  check_r2(target_test, result)
  print("*")
#

#Adding data block
#Sums data up.
def add_data_to_output(data, target_test, column_name, result):
  """Returns a data table with additional block added based on indexes.
  Input: a data table.
  """
  for index, row in data.iterrows():
    if(index in target_test.index):
      data.at[index, column_name] = \
        round(result[target_test.index.get_loc(index)], 2)
    else:
      data.at[index, column_name] = pie.nan
  return data
#

#Drawing block
#Draws diagrams.
def check_list_contained(a, b):
  """Returns True if first array contains values from the second; otherwise - False.
  Input: two arrays.
  """
  for i in range(len(b)):
    if pie.array_equal(a, b[i:i+len(a)]): 
      return True
  return False

def get_random_colours(number, exceptions=[]):
  """Returns a list of unique random CSS4 colours considering exceptions.
  Input: a number of required colours, list of exceptions.
  """
  colours = exceptions
  while(len(colours) == 0 or
        check_list_contained(colours, exceptions) or
        len(set(colours)) != len(colours)):
    colours = random.choices(list(mcolors.CSS4_COLORS.keys()), k=number)
  return colours

def draw_compare_result(data, main_c_name, name):
  """Draws a histogram with several (or one) columns.
  Input: a data table, name of the main column, name of the graph.
  """
  colours = get_random_colours(len(data.columns), list(["white"]))
  data.plot(title=name, x=main_c_name,
            kind="bar", color=colours, ec=colours[-1])

def show_pie_count(c, name, shorte=""):
  """Draws a pie chart based on the counting of data.
  Input: a column to cound data in, name of the graph, 
         a shorten name of values to count.
  """
  fig, ax = plot.subplots()
  countc = c.value_counts()
  countc = countc.iloc[pie.lexsort([-countc.index, -countc.values])]
  ax.pie(countc, shadow = False, startangle = 90)
  ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
  fig = plot.gcf()
  fig.set_size_inches(5,5)
  fig.suptitle(name, fontsize=16)
  circle = plot.Circle(xy=(0,0), radius=0.5, facecolor='white')
  plot.gca().add_artist(circle)
  plot.legend(
      labels = [f'{pie.round(x, 2)} {shorte} - {round(countc[x]/len(c)*100,1)}%'
                for x in countc.index],
      bbox_to_anchor=(1,1),
      fontsize=10)
  plot.show()

def show_heat_map(data, id_name, c_name, v_name, name):
  """Draws a heat map based on three parameters.
  Input: a data table, an index column, a basic column, a column with main values, 
         a name of the graph.
  """
  glue = data.pivot(index=id_name, columns=c_name, values=v_name)
  fig, ax = plot.subplots(figsize=(len(data)/4, len(data)/4))
  plot.suptitle(name)
  sns.heatmap(glue, square=True, linecolor=get_random_colours(1), linewidth=.5)

def show_analytics(data):
  """Outputs analytics.
  Input: a data table.
  """
  print("Информация в графиках")
  if(check_in_name("Стоимость", data.columns)):
    show_pie_count(data["Стоимость"], "Количество квестов по стоимости", "руб.")
  if(check_in_name("Число актёров", data.columns)):
    show_pie_count(data["Число актёров"], "Количество квестов по числу актёров", "чел.")
  if(check_in_name("Уровень страха", data.columns)):
    show_pie_count(data["Уровень страха"], "Количество квестов по уровню страха")
  if(check_in_name("Уровень сложности", data.columns)):
    show_pie_count(data["Уровень сложности"], "Количество квестов по уровню сложности")
  if(check_in_name("Рейтинг", data.columns)):
    show_pie_count(data["Рейтинг"].round(1), "Количество квестов по рейтингу")
  small_data = delete_rows_with_na_column(data, "Рассчитанный результат")
  draw_compare_result(small_data[["Название", "Стоимость", "Рассчитанный результат"]],\
                    "Название", "Сравнение реальной стоимости и полученного результата")
  if(check_in_name("Рейтинг", data.columns)):
    show_heat_map(data, "Название", "Стоимость", "Рейтинг",\
              "Зависимость стоимости и рейтинга друг от друга")
#
#Printing block
#Adds style to data.
def output_data(data):
  """Outputs data table.
  Input: a data table.
  """
  print("Полученные данные:")
  panda.set_option('display.max_rows', None)
  panda.set_option('display.max_columns', None)
  panda.set_option('display.width', 800)
  panda.set_option('display.colheader_justify', 'center')
  display(data)
#

#Main block
#Runs programme.
def main():
  """Runs programme."""
  filename = write_file_name()
  if check_file_exists(filename):
    boolbool, data = do_main_read_work(filename) #qs.txt
    if(boolbool):
      model, features_train, target_train, target_test, result =\
      train_module(data, "Стоимость", ["Название"])
      data = add_data_to_output(data, target_test, "Рассчитанный результат", result)
      output_data(data)
      check_correctness(model, features_train, target_train, target_test, result)
      show_analytics(data)
  else: print("Такого файла не существует. \
               Пожалуйста, перезапустите программу и попробуйте ввести другое название.")

main()  # Hello, World!

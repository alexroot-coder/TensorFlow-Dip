import os
import csv
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras import Sequential
from keras.layers import BatchNormalization
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import time
from tensorflow.python.client import device_lib
from tensorflow.python.framework import ops
from keras import backend as K
import openpyxl
from datetime import date
from operator import itemgetter
import keras_tuner as kt
# from kt import RandomSearch, Hyperband, BayesianOptimization


def build_model(hp):
    model = keras.Sequential()
    activation_choice = hp.Choice('activation', values=['elu', 'selu', 'tanh', 'relu', 'softplus'])
    # model.add(BatchNormalization())
    model.add(Dense(units=hp.Int('units_input', min_value=16, max_value=512, step=32),
                    activation=activation_choice, input_dim=6))
    for i in range(hp.Int("num_layers", 2, 5)):
        model.add(Dense(units=hp.Int('units_hidden_' + str(i), min_value=512, max_value=1024, step=32),
                        activation=activation_choice))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='MSE', metrics=['accuracy'])
    # optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop', 'SGD']),
    # model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop', 'SGD']), loss='mse', metrics=['accuracy'])
    return model


def get_data(path2, path):
    data_input = []
    data_test = []
    data_output = []
    with open(path, newline='') as csvfile:
        data = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in data:
            # print(row)
            data_input.append([float(str(row[1].replace(",", "."))), float(row[2].replace(",", ".")),
                               float(row[3].replace(",", ".")), float(row[4].replace(",", "."))])
            data_output.append([float(row[0].replace(",", "."))])

    with open(path2, newline='') as csvfile:
        data = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in data:
            data_test.append([float(str(row[0].replace(",", "."))), float(row[1].replace(",", ".")),
                              float(row[2].replace(",", ".")), float(row[3].replace(",", "."))])

    return data_input, data_output, data_test


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_big_data(path):
    train_data = []
    test_data = []
    all_data = []
    ALL_BIG_DATA_RETURN = []
    wb_obj = openpyxl.load_workbook(path)
    for name in wb_obj.sheetnames:
        temp_data = []
        sheet_obj = wb_obj[name]
        for i in range(0, 72, 9):
            for j in range(2, 8):
                temp_data.append(sheet_obj.cell(row=2, column=j + i).value)


        colums_data = list(chunks(temp_data, 6))

        for i in range(len(colums_data)):
             colums_data[i] = [colums_data[i][0], colums_data[i][1], colums_data[i][4], colums_data[i][5]]

        temp_data.clear()
        temp = []
        for j in range(0, 315, 38):
            for i in range(0, 72, 9):
                # for g in range(2, 8):
                #     temp.append(sheet_obj.cell(row=2, column=g + i).value)
                for h in range(13, 50):
                    for k in range(1, 10):
                        temp_data.append(sheet_obj.cell(row=h + j, column=k + i).value)

        colums_and_rows_data = list(chunks(temp_data, 9))

        j = 0
        for i in range(len(colums_and_rows_data)):
            if i > 0 and i % 37 == 0:
                j += 1
            if j > 7:
                j = 0
            colums_and_rows_data[i] = colums_and_rows_data[i] + colums_data[j]


        last_not_none = float(colums_and_rows_data[0][0].replace("ГА= ", "").replace(",", "."))
        for i in range(len(colums_and_rows_data)):
            if colums_and_rows_data[i][0] != None:
                colums_and_rows_data[i][0] = float(colums_and_rows_data[i][0].replace("ГА= ", "").replace(",", "."))
                last_not_none = float(colums_and_rows_data[i][0])
            else:
                colums_and_rows_data[i][0] = last_not_none

        for i in range(len(colums_and_rows_data)):
            colums_and_rows_data[i] = [colums_and_rows_data[i][0], colums_and_rows_data[i][2], colums_and_rows_data[i][8],
                     colums_and_rows_data[i][9], colums_and_rows_data[i][10], colums_and_rows_data[i][11],
                     colums_and_rows_data[i][12]]


        colums_and_rows_data = list(chunks(colums_and_rows_data, 37))

        for i in range(len(colums_and_rows_data)):
            colums_and_rows_data[i] = colums_and_rows_data[i][5:]

        colums_and_rows_data = list(chunks(colums_and_rows_data, 8))

        for i in range(len(colums_and_rows_data)):
            # print(i, colums_and_rows_data[i])
            all_data.append(colums_and_rows_data[i])
            if i % 2 == 0:
                train_data.append(colums_and_rows_data[i])
            else:
                test_data.append(colums_and_rows_data[i])
        #
        # for i in range(len(train_data)):
        #     print(i, train_data[i])
    # print(len(train_data), len(test_data))

        # for i in range(len(colums_and_rows_data)):
        #     ALL_BIG_DATA_RETURN.append(colums_and_rows_data[i])
        #
        #
        # # for i in range(len(ALL_BIG_DATA_RETURN)):
        # #     ALL_BIG_DATA_RETURN[i][0] = float(ALL_BIG_DATA_RETURN[i][0].replace(",", "."))
        #
        # ALL_BIG_DATA_RETURN = sorted(ALL_BIG_DATA_RETURN, key=itemgetter(0))
        #
        # for i in range(500):
        #     print(i, ALL_BIG_DATA_RETURN[i])

    return train_data, test_data, all_data


def prepare_data(a):
    data_input = []
    data_output = []
    print(len(a))
    for i in range(len(a)):
        for j in range(len(a[i])):
            for k in range(len(a[i][j])):
                # print(i, a[i][j][k])
                data_output.append([float(a[i][j][k][0])])
                data_input.append([float(a[i][j][k][1]), float(a[i][j][k][2]), float(a[i][j][k][3]),
                                   float(a[i][j][k][4]), float(a[i][j][k][5]),
                                   float(a[i][j][k][6])])

    return data_input, data_output


def get_model():
    model = keras.Sequential([  # BatchNormalization,
                              Input(shape=(15,)),
                              Dense(75, activation='softplus'),
                              Dense(300, activation='softplus'),
                              Dense(500, activation='softplus'),
                              Dense(1, activation='linear'),
                              ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001))  # tf.keras.optimizers.Adam(0.1) sgd
    plot_model(model, to_file='model_1.png')
    return model


def train_model(model, data_input, data_output, epochs, verbose):
    history = model.fit(data_input, data_output, epochs=epochs, verbose=verbose)
    plt.plot(history.history['loss'])
    plt.grid(True)
    plt.savefig(f"MSE_{time.time()}.png")
    # plt.show()
    # return history


# def norm_data(data_input, data_test):
#     a = []
#     b = []
#     c = []
#     d = []
#     for i in range(1200):
#         a.append(data_input[i][0])
#     print(min(a), max(a))
#     for i in range(1200):
#         data_input[i][0] = ((data_input[i][0] - min(a)) / (max(a) - min(a)))
#
#     for i in range(1200):
#         b.append(data_input[i][3])
#     print(min(b), max(b))
#     for i in range(1200):
#         data_input[i][3] = ((data_input[i][3] - min(b)) / (max(b) - min(b)))
#
#     for i in range(1000):
#         c.append(data_test[i][0])
#     print(min(c), max(c))
#     for i in range(1000):
#         data_test[i][0] = ((data_test[i][0] - min(c)) / (max(c) - min(c)))
#
#     for i in range(1000):
#         d.append(data_test[i][3])
#     print(min(d), max(d))
#     for i in range(1000):
#         data_test[i][3] = ((data_test[i][3] - min(d)) / (max(d) - min(d)))
#
#     return data_input, data_test

def norm_data(data):
    norm_1 = []
    # norm_3 = []
    norm_5 = []
    norm_sca = []
    norm_sza = []

    for elem in data:
        norm_sca.append(float(elem[0]))
        norm_1.append(float(elem[1]))
        # norm_3.append(float(elem[3]))
        norm_sza.append(float(elem[4]))
        norm_5.append(float(elem[5]))

    min_norm_1, max_norm_1 = min(norm_1), max(norm_1)
    # min_norm_3, max_norm_3 = min(norm_3), max(norm_3)
    min_norm_5, max_norm_5 = min(norm_5), max(norm_5)
    min_sca, max_sca = min(norm_sca), max(norm_sca)
    min_sza, max_sza = min(norm_sza), max(norm_sza)

    for i in range(len(norm_sca)):
        # norm_azimuth[i] = ((norm_azimuth[i] - min_azimuth) / max_azimuth - min_azimuth)
        norm_sca[i] = ((norm_sca[i] - min_sca) / (max_sca - min_sca))
        norm_sza[i] = ((norm_sza[i] - min_sza) / (max_sza - min_sza))
        norm_1[i] = ((norm_1[i] - min_norm_1) / (max_norm_1 - min_norm_1))
        # norm_3[i] = ((norm_3[i] - min_norm_3) / (max_norm_3 - min_norm_3))
        norm_5[i] = ((norm_5[i] - min_norm_5) / (max_norm_5 - min_norm_5))

    for i in range(len(data)):
        data[i][0] = norm_sca[i]
        #data[i][1] = norm_1[i]
        data[i][2] = data[i][2] / 10
        data[i][3] = data[i][3] * 10
        data[i][4] = norm_sza[i]
        # data[i][5] = norm_5[i]

    return data


def norm_answer(data):
    norm_ans = []

    for elem in data:
        norm_ans.append(float(elem[0]))

    min_norm_ans = min(norm_ans)
    max_norm_ans = max(norm_ans)

    for i in range(len(data)):
        norm_ans[i] = ((norm_ans[i] - min_norm_ans) / (max_norm_ans - min_norm_ans))

    for i in range(len(data)):
        data[i] = norm_ans[i]

    return data


def main():
    # path2 = "C:\\Users\\27les\\Desktop\\input_parametrs.csv"
    # path = "C:\\Users\\27les\\Desktop\\train102.csv"

    # path3 = "E:\\from desktop 29.12.21\\dip\\AOD_0075.xlsx"
    # train_data, test_data, all_data = get_big_data(path3)
    #
    # # train_input_data, train_output_data = prepare_data(train_data)
    # train_input_data, train_output_data = prepare_data(all_data)
    # test_input_data, test_output_data = prepare_data(test_data)
    # #
    # # for j in train_output_data:
    # #     print(j)
    #
    # prepared_data_train_input = np.array(norm_data(train_input_data))
    # prepared_data_train_output = np.array(norm_answer(train_output_data))
    # #prepared_data_train_input = np.array(train_input_data)
    # #prepared_data_train_output = np.array(train_output_data)/13.85
    #
    # print(len(prepared_data_train_input), len(prepared_data_train_output))
    # np.set_printoptions(suppress=True)
    # print(prepared_data_train_input)
    # print(prepared_data_train_output)
    #
    # # for i, j in zip(prepared_data_train_input, prepared_data_train_output):
    # #     print(i, j)
    #
    # test_prepared_data_input = np.array(norm_data(test_input_data))
    # test_prepared_data_output = np.array(norm_answer(test_output_data))
    # #test_prepared_data_input = np.array(test_input_data)
    # #test_prepared_data_output = np.array(test_output_data)/13.85
    # start = time.time()
    # tuner = kt.RandomSearch(
    #     build_model,  # функция создания модели
    #     objective='val_accuracy',  # метрика, которую нужно оптимизировать -
    #     # доля правильных ответов на проверочном наборе данных
    #     max_trials=10,  # максимальное количество запусков обучения
    #     directory='test_directory1'  # каталог, куда сохраняются обученные сети
    # )
    #
    # tuner.search_space_summary()
    # tuner.search(prepared_data_train_input,  # Данные для обучения
    #              prepared_data_train_output,  # Правильные ответы
    #              batch_size=32,
    #              # validation_data=(test_prepared_data_input, test_prepared_data_output),
    #              validation_split=0.2,
    #              epochs=500,  # Количество эпох обучения
    #              )
    #
    # tuner.results_summary()
    #
    # models = tuner.get_best_models(num_models=3)
    # i = 0
    # for model in models:
    #     model.build()
    #     model.summary()
    #     model.evaluate(test_prepared_data_input, test_prepared_data_output)
    #     print()
    #     model.save(f"C:\\Users\\27les\\PycharmProjects\\DIPLOM_TENDORFLOW\\model{i}.pb")
    #     i+=1
    #
    # print('time took: {0:.4f}'.format(time.time() - start))

    # print(prepared_data_train_input)
    # print(prepared_data_train_output)

    # start = time.time()
    # with tf.device('/cpu:0'):
    #     model = keras.Sequential([  # BatchNormalization,
    #
    #         # BatchNormalization(),
    #         # Input(shape=(6,)),
    #         Dense(21, input_dim=6, activation='softmax'),
    #         Dense(21, activation='softplus'),
    #         Dense(17, activation='softplus'),
    #         Dense(5, activation='softplus'),
    #         Dense(1, activation='softplus'),
    #         Dense(1, activation='linear'),
    #     ])
    #     model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.01))
    #
    # with tf.device('/cpu:0'):
    #     history = model.fit(prepared_data_train_input, prepared_data_train_output, epochs=500, verbose=1)
    # print('gpu time took: {0:.4f}'.format(time.time() - start))
    #
    # model.save("C:\\Users\\27les\\PycharmProjects\\DIPLOM_TENDORFLOW\\123.pb")
    # plt.plot(history.history['loss'])
    # plt.grid(True)
    # plt.savefig(f"MSE_{date.today()}.png")
    # plt.show()
    # #
    # # # print(model.predict[prepared_data_input[9543]])
    # # # print(model.get_weights())
    # model.save("C:\\Users\\27les\\PycharmProjects\\DIPLOM_TENDORFLOW\\123.pb")

    # model = keras.models.load_model("C:\\Users\\27les\\PycharmProjects\\DIPLOM_TENDORFLOW\\123.pb")
    #
    # data_test = []
    # i = 0
    #
    # count = 500
    #
    # for inpt, out in zip(test_prepared_data_input, test_prepared_data_output):
    #     if i == count:
    #         break
    #     #print(model.predict([[inpt[0], inpt[1], inpt[2], inpt[3], inpt[4],
    #                                      #inpt[5]]])[0][0], out)
    #     data_test.append(model.predict([[inpt[0], inpt[1], inpt[2], inpt[3], inpt[4],
    #                                      inpt[5]]])[0][0])
    #     i+=1
    #
    # a = test_prepared_data_output[0:count]
    # plt.plot(data_test, color='r', label='data_predict')
    # plt.plot(a, color='g', label='data_wait')
    # plt.xlabel("count")
    # plt.ylabel("data")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

   # train_input = np.array([[50, 18], [60, 20], [75, 45], [65, 21]])
    train_input = np.array([[0, 0.3], [0.4, 0.074], [0.7, 0.3], [0.6, 0.4], [0.0, 0.0]])
    train_output = np.array([[0.3], [0.474], [1.0], [1.0], [0.0]])

    with tf.device('/cpu:0'):
        model = keras.Sequential([  # BatchNormalization,
            # Input(shape=(6,)),
            Dense(10, input_dim=2, activation='softplus'),
            Dense(40, activation='softplus'),
            Dense(15, activation='softplus'),
            Dense(10, activation='softplus'),
            Dense(1, activation='linear'),
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.01))

    with tf.device('/cpu:0'):
        history = model.fit(train_input, train_output, epochs=500, verbose=1)

    model.save("C:\\Users\\27les\\PycharmProjects\\DIPLOM_TENDORFLOW\\123.pb")
    plt.plot(history.history['loss'])
    plt.grid(True)
    plt.savefig(f"MSE_{date.today()}.png")
    plt.show()

    print(model.predict([[0, 0]]))  # 0
    print(model.predict([[0.5, 0.06]]))  # 0.56
    print(model.predict([[0.3, 0.3]]))  # 0.6
    print(model.predict([[0.5, 0.5]]))  # 1


if __name__ == "__main__":
    main()

import csv
import math
import numpy as np


def norm_srednee_otklon(data_input):
    '''
    нормализация данных для модели Деминова
    по 2-му способу https://www.youtube.com/watch?v=rRDRlc7xolU&t=1s
    '''
    a = []
    b = []
    c = []
    d = []
    e = []
    f = []
    for elem in data_input:
        a.append(float(elem[0]))
        b.append(float(elem[1]))
        c.append(float(elem[2]))
        d.append(float(elem[3]))
        e.append(float(elem[4]))
        f.append(float(elem[5]))

    avg_a = sum(a) / len(a)
    avg_b = sum(b) / len(b)
    avg_c = sum(c) / len(c)
    avg_d = sum(d) / len(d)
    avg_e = sum(e) / len(e)
    avg_f = sum(f) / len(f)

    temp_a = 0
    for i in range(len(a)):
        temp_a += (a[i] - avg_a) ** 2

    s_a = math.sqrt((1 / (len(a) - 1)) * temp_a)

    temp_b = 0
    for i in range(len(b)):
        temp_b += (b[i] - avg_b) ** 2
    s_b = math.sqrt((1 / (len(b) - 1)) * temp_b)

    temp_c = 0
    for i in range(len(c)):
        temp_c += (c[i] - avg_c) ** 2
    s_c = math.sqrt((1 / (len(c) - 1)) * temp_c)

    temp_d = 0
    for i in range(len(d)):
        temp_d += (d[i] - avg_d) ** 2
    s_d = math.sqrt((1 / (len(d) - 1)) * temp_d)

    temp_e = 0
    for i in range(len(e)):
        temp_e += (e[i] - avg_e) ** 2
    s_e = math.sqrt((1 / (len(e) - 1)) * temp_e)

    temp_f = 0
    for i in range(len(f)):
        temp_f += (f[i] - avg_f) ** 2
    s_f = math.sqrt((1 / (len(f) - 1)) * temp_f)

    for i in range(len(data_input)):
        data_input[i][0] = (a[i] - avg_a) / s_a
        data_input[i][1] = (b[i] - avg_b) / s_b
        data_input[i][2] = (c[i] - avg_c) / s_c
        data_input[i][3] = (d[i] - avg_d) / s_d
        data_input[i][4] = (e[i] - avg_e) / s_e
        data_input[i][5] = (f[i] - avg_f) / s_f

    return data_input, [[avg_a, s_a], [avg_b, s_b], [avg_c, s_c], [avg_d, s_d], [avg_e, s_e], [avg_f, s_f]]


def prepare_data(a):
    '''
    подготовка данных
    перевод из str в  float
    формирование обучающей выборки data_input - входные данные, data_output - выходные данные
    '''
    data_input = []
    data_output = []
    print(len(a))
    for i in range(len(a)):
        data_input.append([float(a[i][1]), float(a[i][2]), float(a[i][3]), float(a[i][4]), float(a[i][5]), float(a[i][6])])
        data_output.append([float(a[i][0])])
            # for k in range(len(a[i][j])):
            #     data_output.append([float(a[i][j][k][0])])
            #     data_input.append([float(a[i][j][k][1]), float(a[i][j][k][2]), float(a[i][j][k][3]),
            #                        float(a[i][j][k][4]), float(a[i][j][k][5]),
            #                        float(a[i][j][k][6])])
                # data_input.append([float(a[i][j][k][1]), float(a[i][j][k][2]),
                #                    float(a[i][j][k][5]),
                #                    float(a[i][j][k][6])])

    return data_input, data_output


def norm_data(data):
    '''
    нормализация данных для моей модели
    по 1-му способу minmax https://www.youtube.com/watch?v=rRDRlc7xolU&t=1s
    '''

    norm_sca = []
    norm_rwi = []
    norm_wave = []
    norm_aod = []
    norm_sza = []
    norm_albedo = []

    for elem in data:
        norm_sca.append(float(elem[0]))
        norm_rwi.append(float(elem[1]))
        norm_wave.append(float(elem[2]))
        norm_aod.append(float(elem[3]))
        norm_sza.append(float(elem[4]))
        norm_albedo.append(float(elem[5]))

    min_norm_sca, max_norm_sca = min(norm_sca), max(norm_sca)
    min_norm_rwi, max_norm_rwi = min(norm_rwi), max(norm_rwi)
    min_norm_wave, max_norm_wave = min(norm_wave), max(norm_wave)
    min_norm_aod, max_norm_aod = min(norm_aod), max(norm_aod)
    min_norm_sza, max_norm_sza = min(norm_sza), max(norm_sza)
    min_norm_albedo, max_norm_albedo = min(norm_albedo), max(norm_albedo)
    print(min_norm_albedo, max_norm_albedo)

    for i in range(len(norm_sca)):
        norm_sca[i] = ((norm_sca[i] - min_norm_sca) / (max_norm_sca - min_norm_sca))
        norm_rwi[i] = ((norm_rwi[i] - min_norm_rwi) / (max_norm_rwi - min_norm_rwi))
        norm_wave[i] = ((norm_wave[i] - min_norm_wave) / (max_norm_wave - min_norm_wave))
        norm_aod[i] = ((norm_aod[i] - min_norm_aod) / (max_norm_aod - min_norm_aod))
        norm_sza[i] = ((norm_sza[i] - min_norm_sza) / (max_norm_sza - min_norm_sza))
        norm_albedo[i] = ((norm_albedo[i] - min_norm_albedo) / (max_norm_albedo - min_norm_albedo))

    for i in range(len(data)):
        data[i][0] = norm_sca[i]
        data[i][1] = norm_rwi[i]
        data[i][2] = norm_wave[i]
        data[i][3] = norm_aod[i]
        data[i][4] = norm_sza[i]
        data[i][5] = norm_albedo[i]

    return data, [[min_norm_sca, max_norm_sca], [min_norm_rwi, max_norm_rwi], [min_norm_wave, max_norm_wave], [min_norm_aod, max_norm_aod], [min_norm_sza, max_norm_sza], [min_norm_albedo, max_norm_albedo]]


path = "TRAIN_DATA.csv"
data_for_prepare = []

with open(path, newline='') as csvfile:
    data = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in data:
        # data_for_prepare.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6])])
        data_for_prepare.append([row[0], row[1], row[2], row[3], row[4], row[5], row[6]])

train_data_input, train_data_output = prepare_data(data_for_prepare)
prepared_data_input, srednee = norm_srednee_otklon(train_data_input)
prepared_data_input = np.array(prepared_data_input)
prepared_data_output = np.array(train_data_output) / 13.85
#
#
with open("srednee.csv", "w", encoding='UTF8', newline='') as f:
    www = csv.writer(f)
    www.writerow([str(srednee[0][0]), str(srednee[0][1]),
                  str(srednee[1][0]), str(srednee[1][1]),
                  str(srednee[2][0]), str(srednee[2][1]),
                  str(srednee[3][0]), str(srednee[3][1]),
                  str(srednee[4][0]), str(srednee[4][1]),
                  str(srednee[5][0]), str(srednee[5][1])])
    for i, j in zip(prepared_data_input, prepared_data_output):
        www.writerow([str(i[0]), str(i[1]), str(i[2]), str(i[3]), str(i[4]), str(i[5]), str(j[0])])


del train_data_input, train_data_output, prepared_data_input, prepared_data_output

train_data_input, train_data_output = prepare_data(data_for_prepare)
prepared_data_input, minmax = norm_data(train_data_input)
prepared_data_input = np.array(prepared_data_input)
prepared_data_output = np.array(train_data_output) / 13.85

with open("minmax.csv", "w", encoding='UTF8', newline='') as f:
    www = csv.writer(f)
    www.writerow([str(minmax[0][0]), str(minmax[0][1]),
                  str(minmax[1][0]), str(minmax[1][1]),
                  str(minmax[2][0]), str(minmax[2][1]),
                  str(minmax[3][0]), str(minmax[3][1]),
                  str(minmax[4][0]), str(minmax[4][1]),
                  str(minmax[5][0]), str(minmax[5][1])])
    for i, j in zip(prepared_data_input, prepared_data_output):
        www.writerow([str(i[0]), str(i[1]), str(i[2]), str(i[3]), str(i[4]), str(i[5]), str(j[0])])


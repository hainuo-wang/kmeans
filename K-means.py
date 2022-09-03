import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xlrd
from xlrd import xldate_as_tuple
import numpy as np
import pandas as pd
from xlutils import copy
from sklearn import datasets
from sklearn import preprocessing


def get_timeMap():
    data = xlrd.open_workbook("result.xls")  # 打开excel
    table = data.sheet_by_name("朗贝20140415-0612")  # 读sheet
    timeMap = {}
    for i in range(1, table.nrows):
        row = table.row_values(i)  # 行的数据放在数组里
        date = xldate_as_tuple(row[0], 0)
        if date[1] == 4 and date[2] == 15:
            key = str(date[3]) + ":" + str(date[4]) + ":" + str(date[5])
            timeMap[key] = i
    enc = preprocessing.OneHotEncoder()
    studyData = []
    for item in timeMap.values():
        studyData.append([item])
    enc.fit(studyData)  # fit来学习编码
    endcodeData = enc.transform(studyData).toarray().tolist()  # 进行编码
    count = 0
    for key in timeMap.keys():
        timeMap[key] = endcodeData[count]
        count = count + 1
    return timeMap


# def get_result():
timeMap = get_timeMap()
data = xlrd.open_workbook("result.xls")
table = data.sheet_by_name("朗贝20140415-0612")
use11 = table.col_values(2)
use22 = table.col_values(3)
use1 = np.array(use11)
use2 = np.array(use22)
vectors1 = []
vectors2 = []
for i in range(1, table.nrows):
    row = table.row_values(i)
    date = xldate_as_tuple(row[0], 0)
    key = str(date[3]) + ":" + str(date[4]) + ":" + str(date[5])

    import copy

    item1 = copy.deepcopy(timeMap[key])
    item1.append(use1[i - 1])
    vectors1.append(item1)
    item2 = copy.deepcopy(timeMap[key])
    item2.append(use2[i - 1])
    vectors2.append(item2)


# result1,result2=get_result()
def choose_by_silhouette(X):
    tests = [2, 3, 4, 5]
    sc_scores = []
    for k in tests:
        kmeans_model = KMeans(n_clusters=k).fit(X)
        sc_score = metrics.silhouette_score(X, kmeans_model.labels_, metric='euclidean')
        sc_scores.append(sc_score)
    return sc_scores


times = []
for count in range(0, len(times)):
    times[count] = times[count].replace(2014, 4, 15)


def cluster_by_K(x1, x2, X, k, name):
    '''
    :param x1:   时间列表
    :param x2:   用水量列表
    :param X:    样本向量构成的矩阵
    :param k:    k值
    :param name: 图片名称
    :return:
    '''
    data = xlrd.open_workbook("result.xls")  # 打开excel
    from xlutils.copy import copy
    wb = xlutils.copy.copy(data)
    sheet = wb.get_sheet(0)
    if name == 'DMA1':
        cloumn = 3
    else:
        cloumn = 4
    sheet.write(0, cloumn, name)
    plt.figure(figsize=(8, 6))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
    markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']
    kmeans_model = KMeans(n_clusters=k).fit(X)
    for i, l in enumerate(kmeans_model.labels_):
        plt.subplot(k, 1, 1 + l)
        sheet.write(i + 1, cloumn, str(l))
        print(i)
        plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l])
    plt.show()
    wb.save("result.xls")
    for t in range(1, k + 1):
        plt.subplot(k, 1, t)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.gcf().autofmt_xdate()  # 自动旋转日期标记
        plt.title('第' + str(t) + '种', fontproperties=font)
    plt.show()
    plt.savefig(name + '.png')
    return kmeans_model.labels_

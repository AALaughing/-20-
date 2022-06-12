from sklearn.tree import DecisionTreeRegressor
from data_process import *
from sklearn.linear_model import LogisticRegression
from numpy import ravel
import xlwt


# init data
def init_data(train_data, test_data, feature_id):
    columns = corr_feature(feature_id=feature_id)
    train_data = select_data(train_data, columns)
    # print(train_data)
    test_data = select_data(test_data, columns)
    train_data = fill_loss_data(train_data)
    test_data = fill_loss_data(test_data)
    # print(train_data.columns)
    x_train = train_data.drop(['match'], axis=1)
    y_train = train_data[['match']]
    x_test = test_data.drop(['match'], axis=1)
    y_test = test_data[['match']]
    return x_train, y_train, x_test, y_test


# obtain accuracy
def compute_accuracy(sample, data, param):
    if len(sample) != len(data):
        return 'wrong'
    for i in range(0, len(sample) - 1):
        if sample[i] > param:
            sample[i] = 1
        else:
            sample[i] = 0
    acc_number = 0
    for i, a in zip(range(0, len(sample) - 1), data.values):
        if sample[i] == a:
            acc_number += 1
    return acc_number / len(sample)


# train model
def train_model(feature_id, train_data, test_data, C, max_iter,d):
    x_train, y_train, x_test, y_test = init_data(train_data, test_data, feature_id)
    # for a in y_train.values:
    #     print(a[0])
    model = LogisticRegression(penalty='l1', dual=False, tol=0.01, C=C, fit_intercept=True,
                               intercept_scaling=1, solver='liblinear', max_iter=max_iter)
    model.fit(x_train, ravel(y_train))
    # y_result = model.predict(x_train)
    test_result = model.predict(x_test)
    train_acc = model.score(x_train, y_train)
    test_acc = model.score(x_test, y_test)
    # train_acc = compute_accuracy(y_result, y_train, param)
    # test_acc = compute_accuracy(test_result, y_test, param)
    # print("训练准确率:%.4f" % (train_acc * 100))
    # print("第%d测试准确率:%.4f" % (d,(test_acc * 100)))
    # print(test_result)

    return model, train_acc, test_acc,test_result


# 网格法调超参数
def grid_search():
    d=0
    train_data, test_data = load_data()
    test_data = get_ground_truth(test_data)
    Cs = [0.8, 1.0, 1.2, 1.4, 1.6]
    max_iters = [100, 500, 1000, 2000, 5000]
    feature_ids = [i for i in range(1, 14)]
    train_acc = 0.0
    test_acc = 0
    best_acc = 0.0
    best_log = {
        'model': None,
        'train_accuracy': 0.0,
        'test_accuracy': 0.0,
        'C': 0.0,
        'max_iter': 0,
        'feature_id': 1
    }
    for feature_id in feature_ids:
        for C in Cs:
            for max_iter in max_iters:
                d+=1
                # print("feature_id:", feature_id, ",C:", C, ",max_iter:", max_iter)
                model, train_acc, test_acc,test_result = train_model(feature_id, train_data, test_data, C, max_iter, d)
                if best_acc < test_acc:
                    best_acc = test_acc
                    best_log['model'] = model
                    best_log['train_accuracy'] = train_acc
                    best_log['test_accuracy'] = test_acc
                    best_log['C'] = C
                    best_log['max_iter'] = max_iter
                    best_log['feature_id'] = feature_id
                    test_result=test_result
    print(best_log,test_result)
    f = xlwt.Workbook('encoding = utf-8')  # 设置工作簿编码
    sheet1 = f.add_sheet('sheet1', cell_overwrite_ok=True)  # 创建sheet工作表
    for i in range(len(test_result)):
        sheet1.write(i, 0, test_result[i])  # 写入数据参数对应 行, 列, 值
    f.save('text.xls')  # 保存.xls到当前工作目录

    # for feature_id in feature_ids:
    #     print("feature_id:", feature_id)
    #     model, train_acc, test_acc = train_model(feature_id, train_data, test_data, 0.8, 500)
    #     if best_acc < test_acc:
    #         best_acc = test_acc


if __name__ == '__main__':
    grid_search()
    # train_data, test_data = load_data()
    # test_data = get_ground_truth(test_data)
    # train_model(5, train_data, test_data, 1.0, 1000)
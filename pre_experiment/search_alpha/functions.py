# coding: UTF-8

import csv
import random
import numpy as np 
import math
import statistics

'''
    csvファイルの読み込み
'''
def read_csv(filename):
    with open(filename + '.csv', 'r') as csv_file:
        data = list(csv.reader(csv_file))
    
    return data

'''
    csvファイルへの書き込み
'''
def write_csv(filename, data):
    with open(filename + '.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for row in data:
            writer.writerow(row)

'''
    ランダムな親個体の選択
'''
def random_parent(data, num_parents):
    parents_index = np.full(num_parents, -1)
    parents = np.zeros((num_parents, len(data[0])))
    for parent in range(num_parents):
        parent_index = random.randint(0, len(data) - 1)
        while parent_index in parents_index:
            parent_index = random.randint(0, len(data) - 1)
        parents_index[parent] = parent_index
        parents[parent] = np.copy(data[parent_index])
    
    return parents, parents_index

'''
    REX
    乱数には正規分布を使用
'''
def REX(parent_vector):
    weight = np.mean(parent_vector)
    child_temp = weight
    for element in parent_vector:
        child_temp += (element - weight) * np.random.normal(0, math.sqrt(1/len(parent_vector)))
        
    return child_temp

'''
    重み付けによる要素の取得
'''
def choose_roulette(data, num_parents, solutions, bias):
    weight_list = []
    parents_index = []
    parents = []
    evaluations = get_evaluations_list(data, solutions, bias)
    rank_list = get_ranking_list(evaluations)
    for elements in range(len(data)):
        weight = len(data) - rank_list[elements] + 1
        for w in range(weight):
            weight_list.append(elements)

    for parent in range(num_parents):
        parent_index = random.choice(weight_list)
        while parent_index in parents_index:
            parent_index = random.choice(weight_list)
        parents_index.append(parent_index)
        
        parent_vector = []
        for element in range(len(data[parent_index])):
            parent_vector.append(data[parent_index][element])
        parents.append(parent_vector)

    return np.array(parents)

'''
    拡張XLM
'''
def XLM(parent_vector):
    weight = np.sum(parent_vector) / len(parent_vector)
    child = parent_vector[0]
    for element in parent_vector:
        child += (element - weight) * np.random.normal(0, 1/len(parent_vector))

    return child


'''
    エリート保存分をリストに追加
    エリート保存する数を％で定義
'''
def elite_preservation_percent(data, elite_preservation_rate, children):
    for row in data:
        if float(row[len(row) - 1]) <= len(data) * elite_preservation_rate:
            vector = []
            for element in range(len(row) - 1):
                vector.append(float(row[element]))
            children.append(vector)

    return children 

'''
    エリート保存分をリストに追加
    エリート保存する数を個数で定義
'''
def elite_preservation_num(data, num_elite_preservation, children):
    for row in range(len(data)):
        if row < num_elite_preservation:
            vector = []
            for dimention in range(len(data[row]) - 1):
                vector.append(float(data[row][dimention]))
            children.append(vector)

    return children 

'''
    データの中身を全てfloat型に変換する
'''
def transform_to_float(data):
    new_matrix = []
    for row in data:
        vector = []
        for column in row:
            vector.append(float(column))
        new_matrix.append(vector)
    
    return new_matrix

'''
    ユークリッド距離
'''
def euclid_distance(vector1, vector2):
    distance = np.sum((vector1 - vector2) ** 2)
    
    return math.sqrt(distance)

'''
    局所解とバイアスに分ける
'''
def divide_solutions_bias(solutions_data):
    solutions = []
    bias = []
    for row in solutions_data:
        vector = []
        for column in range(len(row) - 1):
            vector.append(row[column])
        solutions.append(vector)
        bias.append(row[len(row) - 1])

    return solutions, bias

'''
    最適な局所解のインデックスの取得
'''
def get_best_solution_index(bias):
    maximum = bias[0]
    index = 0
    for current_index in range(len(bias)):
        if maximum < bias[current_index]:
            maximum = bias[current_index]
            index = current_index

    return index

'''
    局所解とそれぞれの解のバイアスを元に評価値を返す
'''
def get_evaluation_value(vector, solutions, bias):
    evaluations = np.zeros(len(solutions))
    for solution_index in range(len(solutions)):
        evaluations[solution_index] = (vector[0] - solutions[solution_index][0]) * (vector[0] - solutions[solution_index][0])
        
    index = np.argmin(evaluations)
    minimum = np.min(evaluations)
    minimum *= 20 - bias[index]

    return minimum

'''
    評価値のリストを返す
'''
def get_evaluations_list(data, solutions, bias):
    evaluations = []
    for element in data:
        evaluations.append(get_evaluation_value(element, solutions, bias))

    return np.array(evaluations)

'''
    評価値のリストからインデックス固定でランキングのリストを返す。
'''
def get_ranking_list(evaluations):
    rank_list = []
        
    for index1 in range(len(evaluations)):
        count = 0
        for index2 in range(len(evaluations)):
            if evaluations[index1] > evaluations[index2]:
                count += 1
        rank_list.append(count + 1)
    
    return np.array(rank_list)

'''
    最も近い局所解のインデックスを返す
'''
def get_nearest_solution_index(vector, solutions):
    index = 0
    distance = euclid_distance(vector, solutions[0])
    for current_index in range(len(solutions)):
        if distance > euclid_distance(vector, solutions[current_index]):
            distance = euclid_distance(vector, solutions[current_index])
            index = current_index
    
    return index

'''
    評価の結果を返す
'''
def get_result(data, evaluations, num_experiment, best_solution_index, solutions):
    evaluation_vector = []
    evaluation_vector.append(num_experiment)
    minimum = np.min(evaluations)
    index = np.argmin(evaluations)
    nearest_solution_index = get_nearest_solution_index(data[index], solutions)
    evaluation_vector.append(minimum)
    evaluation_vector.append(statistics.mean(evaluations))
    evaluation_vector.append(statistics.variance(evaluations))
    if nearest_solution_index == best_solution_index:
        evaluation_vector.append('Yes')
    else:
        evaluation_vector.append('No')
    
    return evaluation_vector

'''
    評価値が最も良かったものの平均、分散および最適解に入った確率のベクトル返す
'''
def get_final_result(results):
    best_evaluations = []
    is_best_solutions = []
    result_vector = []
    result_vector.append('')
    for result in results:
        best_evaluations.append(result[1])
        is_best_solutions.append(result[4])
    result_vector.append(statistics.mean(best_evaluations))
    result_vector.append(statistics.variance(best_evaluations))
    
    num_yes = 0
    for is_best_solution in is_best_solutions:
        if is_best_solution == 'Yes':
            num_yes += 1
    
    result_vector.append(num_yes / len(is_best_solutions))

    return result_vector

'''
    結果をcsvファイルに書き込む
'''
def write_result(filename, results, whole_result):
    header = ['試行回数', '最小評価値', '評価値平均', '評価値分散', '最適解に入ったか否か']
    whole_header = ['', '最小評価値の平均', '最小評価値の分散', '最適解の確率']
    with open(filename + '.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for row in results:
            writer.writerow(row)
        
        writer.writerow(whole_header)
        writer.writerow(whole_result)

'''
    指定した次元の平均と標準偏差を取得する
'''
def get_mean_sd(data, dimention):
    vector = np.zeros(len(data[0]))
    for individual in range(len(data)):
        vector[individual] = data[individual][dimention]
    mean = np.mean(vector)
    standard_deviation = np.std(vector)
    
    return mean, standard_deviation

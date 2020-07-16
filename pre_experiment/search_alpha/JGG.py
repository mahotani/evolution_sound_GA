# coding: UTF-8

import csv
import numpy as np 
import functions

'''
    交叉
'''
def crossover(parents, num_parents, num_dimentions):
    child = np.zeros(num_dimentions)
    for child_index in range(len(parents[0])):
        parents_vector = np.zeros(num_parents)
        for parents_index in range(len(parents)):
            parents_vector[parents_index] = parents[parents_index][child_index]
            # parents_vector.append(parents[parents_index][child_index])
        child[child_index] = functions.REX(parents_vector)
        # child.append(functions.XLM(parents_vector))

    return child

'''
    JGGによる次世代の生成
'''
def next_generation_JGG(data, solutions, bias, num_parents, num_children, num_dimentions):
    # 親個体と親個体のindexを取得
    parents, parents_index = functions.random_parent(data, num_parents)

    # 子個体を入れる0行列を用意
    children = np.zeros((num_children, num_dimentions))

    # crossover()を使用して子個体の生成
    for cross in range(num_children):
        child_vector = crossover(parents, num_parents, num_dimentions)
        children[cross] = np.copy(child_vector)
    
    # 評価値の取得
    evaluations = functions.get_evaluations_list(children, solutions, bias)
    # 評価値を元に子個体のランキングを取得
    rank_list = functions.get_ranking_list(evaluations)

    # TODO: ここの良子個体と親個体の交換コードをforを使ってきれいにする
    # children[0:3]の中で一番良い子個体のindexを取得
    min_index1 = np.argmin(evaluations[0:4])
    # children[0:3]の一番良い子個体と親個体の交換
    data[parents_index[0]] = np.copy(children[min_index1])

    # children[4:7]の中で一番良い子個体のindexを取得
    min_index2 = np.argmin(evaluations[4:8]) + 4
    # children[0:3]の一番良い子個体と親個体の交換
    data[parents_index[1]] = np.copy(children[min_index2])

    # children[8:11]の中で一番良い子個体のindexを取得
    min_index3 = np.argmin(evaluations[8:12]) + 8
    # children[8:11]の一番良い子個体と親個体の交換
    data[parents_index[2]] = np.copy(children[min_index3])

    # children[12:15]の中で一番良い子個体のindexを取得
    min_index4 = np.argmin(evaluations[12:16]) + 12
    # children[12:15]の一番良い子個体と親個体の交換
    data[parents_index[3]] = np.copy(children[min_index4])

    return data


'''
    Main
'''
# 一度の交叉で使う親の数
num_parents = 4
# 一度の交叉で生まれる子の数
num_children = 16
# 読み込むファイル
read_filename = 'csv_files/mock_initial_individuals'
# 書き込むファイル
write_mean = 'csv_files/alpha=1/means'
write_std = 'csv_files/alpha=1/standard_deviations'
# 実行回数
num_execute = 1000
# 潜在空間の次元数
num_dimentions = 100

# 局所解ファイル
solutions_file = 'csv_files/mock_solution_zeros'
# 評価結果のファイル
result_file = 'csv_files/evaluation_result'

# 局所解ファイルの読み込み
solutions_data = functions.read_csv(solutions_file)
del solutions_data[0]
solutions_data = functions.transform_to_float(solutions_data)

# 局所解とバイアスに分ける
solutions, bias = functions.divide_solutions_bias(solutions_data)
solutions = np.array(solutions)
bias = np.array(bias)

# 評価値の結果のリスト
evaluations_result = []

# 平均を入れるリスト
means = []

# 標準偏差を入れるリスト
standard_deviations = []



for num_experiment in range(1, 101):
    print(num_experiment)
    # 対象のデータの読み込み
    data = functions.read_csv(read_filename)
    del data[0]
    data = functions.transform_to_float(data)
    data = np.array(data)
    this_mean, this_std = functions.get_mean_sd(data, 99)
    mean = [this_mean]
    std = [this_std]
    for num in range(num_execute):
        data = next_generation_JGG(data, solutions, bias, num_parents, num_children, num_dimentions)
        this_mean, this_std = functions.get_mean_sd(data, 99)
        mean.append(this_mean)
        std.append(this_std)
        # print('-------')
        # print(functions.get_result(data, functions.get_evaluations_list(data, solutions, bias), num_experiment, functions.get_best_solution_index(bias), solutions))

    evaluations = functions.get_evaluations_list(data, solutions, bias)
    evaluations_vector = functions.get_result(data, evaluations, num_experiment, functions.get_best_solution_index(bias), solutions)
    evaluations_result.append(evaluations_vector)
    means.append(mean)
    standard_deviations.append(std)

functions.write_csv(write_mean, means)
functions.write_csv(write_std, standard_deviations)

# final_result = functions.get_final_result(evaluations_result)
# functions.write_result(result_file, evaluations_result, final_result)

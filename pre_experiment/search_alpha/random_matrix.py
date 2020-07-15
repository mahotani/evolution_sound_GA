# coding: UTF-8

import csv
import numpy as np 
import math
import statistics
import copy

'''
    csvファイルへの書き込み
'''
def write_csv(data, filename):
    header = []
    
    for dimention in range(1, len(data[0])):
        header.append('次元%i' % (dimention))
    header.append('平均')
    
    with open(filename + '.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for row in data:
            writer.writerow(row)

'''
    +1か-1をそれぞれ1/2の確率で出力する
'''
def noise():
    return 1 if np.random.rand() >= 0.5 else -1

'''
    ランダムベクトルの生成
    要素は全て[-1, 1]の範囲
'''
def make_random_vector(num_dimentions):
    random_vector = []
    for dimention in range(num_dimentions):
        random_vector.append(np.random.rand() * noise())

    return random_vector

'''
    ランダム行列の生成
    要素は全て[-1, 1]の範囲
'''
def make_random_matrix(num_elements, num_dimentions):
    random_matrix = []
    for element in range(num_elements):
        random_matrix.append(make_random_vector(num_dimentions))
    return random_matrix

'''
    ユークリッド距離
'''
def euclid_distance(vector1, vector2):
    distance = 0
    for index in range(len(vector1)):
        distance += (vector1[index] - vector2[index]) * (vector1[index] - vector2[index])
    
    return math.sqrt(distance)

'''
    各要素と他の要素との距離の行列を返す
'''
def get_distance_matrix(data):
    distance_matrix = []

    for vector1 in data:
        distance_vector = []
        for vector2 in data:
            distance_vector.append(euclid_distance(vector1, vector2))
        distance_matrix.append(distance_vector)

    return distance_matrix

'''
    各ベクトルのユークリッド距離を取って各ベクトルと他のベクトルとの距離の平均を出し
    最小のもののインデックスを返す。
    各ベクトルの平均と全体の平均と分散も出す。
'''
def each_distance(distance_matrix):
    all_vecotors_mean = []
    for vector in distance_matrix:
        all_vecotors_mean.append(sum(vector) / len(vector))
    
    minimum = all_vecotors_mean[0]
    index = 0
    for current_index in range(1, len(distance_matrix)):
        if minimum > all_vecotors_mean[current_index]:
            minimum = all_vecotors_mean[current_index]
            index = current_index

    all_means = statistics.mean(all_vecotors_mean)
    # all_variance = statistics.variance(all_vecotors_mean)
    all_vecotors_mean.append(all_means)
    # all_vecotors_mean.append(all_variance)

    return index, all_vecotors_mean

'''
    指定したindexの要素を入れ替える
'''
def replace_element(matrix, index, new_vector):
    matrix[index] = new_vector
    return matrix

'''
    指定したindexの要素と他の要素とのユークリッド距離を更新する
'''
def update_distance(data, distance_matrix, index):
    new_distances_matrix = copy.deepcopy(distance_matrix)
    # 横成分の更新
    for vector in range(len(data)):
        distance = euclid_distance(data[index], data[vector])
        new_distances_matrix[index][vector] = distance

    # 縦成分の更新
    for vector in range(len(data)):
        new_distances_matrix[vector][index] = euclid_distance(data[vector], data[index])
    
    return new_distances_matrix

'''
    ユークリッド距離の履歴のcsvファイル用のヘッダーを返す
'''
def header(num_elements):
    header = []
    header.append('試行回数')
    for element in range(num_elements):
        header.append('要素%i' % (element))
    
    header.append('全体の平均')
    header.append('全体の分散')
    return header

'''
    それぞれの要素にランダムになるようにリストの末尾に順位をつける
'''
def add_ranking(random_matrix):
    rank = []
    for index in range(len(random_matrix)):
        rank.append(index + 1)
    
    for swap in range(len(random_matrix)):
        element_index1 = np.random.randint(0, len(random_matrix) - 1)
        element_index2 = np.random.randint(0, len(random_matrix) - 1)
        rank[element_index1], rank[element_index2] = rank[element_index2], rank[element_index1]

    for index in range(len(random_matrix)):
        random_matrix[index].append(rank[index])

    return random_matrix

'''
    Main
'''
# 個体数
num_elements = 100
# 次元数
num_dimentions = 100
# ユークリッド距離の小さい要素を連続で入れ替えなかった回数
num_noreplace = 0



# ユークリッド距離の履歴
euclid_history = []

# [-1, 1]の範囲でランダム行列を作成
random_matrix = make_random_matrix(num_elements, num_dimentions)
        
# ユークリッド距離の行列
euclid_matrix = get_distance_matrix(random_matrix)

index, distances = each_distance(euclid_matrix)
for num in range(1, 2):
    while num_noreplace < 1000:
        print(num_noreplace)
        temp = euclid_matrix
        # ユークリッド距離の平均のリストと最も小さい距離のインデックスの取得
        index, distances = each_distance(euclid_matrix)
        # ユークリッド距離に関するリストの更新
        euclid_history.append(distances)
        # 差し替え候補のベクトルを作成
        new_vector = make_random_vector(num_dimentions)
        # 仮に差し替えた後の個体群行列の作成
        temp_matrix = replace_element(random_matrix, index, new_vector)
        # 仮に差し替えた行列のユークリッド距離の行列を取得
        temp_euclid_matrix = update_distance(temp_matrix, euclid_matrix, index)
        # 仮に差し替えた行列のユークリッド距離の平均のリストと最も小さい距離のインデックスの取得
        new_index, new_distances = each_distance(temp_euclid_matrix)

        # 差し替えた方が距離が大きい場合は個体群行列を差し替えて更新し、num_noreplaceをリセットする
        if distances[100] < new_distances[100]:
            random_matrix = temp_matrix
            num_noreplace = 0
            # ユークリッド距離行列を更新する
            euclid_matrix = temp_euclid_matrix
        # 差し替えない方が良い場合、個体群行列を更新せずnum_noreplaceを+1する
        else:
            num_noreplace += 1
    '''
    # 重み付け(解空間用)
    for row in random_matrix:
        row.append(np.random.rand() * 20)
    '''

    # それぞれの個体にランダムでランキングを割り振る
    # add_ranking(random_matrix)

    # 書き込むファイル
    filename = 'csv_files/mock_initial_individuals'

    # 結果をcsvに書き込む
    write_csv(random_matrix, filename)

    # ユークリッド距離の履歴をcsvファイルに書き込む
    write_csv(euclid_history, 'csv_files/euclid_history')

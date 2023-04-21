import numpy as np
import pandas as pd
import mmh3
from scipy.stats import poisson, nbinom
import math
from discretegauss import sample_dgauss

def hash(x, seed=9):
    x = str(x)
    h = mmh3.hash(x, seed=seed)
    return h

def rightmost_binary_1_position(num):
    i = 0
    while (num >> i) & 1 == 0:
        i += 1
        if i == 24:
            break
    return i + 1
f=open('./MP-SPDZ/Player-Data/Input-P0-0', 'w')
def MPS_plus(numbers, Fu, bits_for_bucket_index, seed_bias=0):
    bucket_count = 2 ** bits_for_bucket_index
    seed_bias = seed_bias * bucket_count
    buckets_A = [[0 for column_index in range(25)] for row_index in range(bucket_count) ]
    buckets_B = [[0 for column_index in range(25)] for row_index in range(bucket_count) ]
    buckets_C = [[0 for column_index in range(25)] for row_index in range(bucket_count) ]
    buckets_D = [[0 for column_index in range(25)] for row_index in range(bucket_count) ]
    for v_num, v in enumerate(numbers):
        hash_row = hash(v, seed=bucket_count + seed_bias)
        i = hash_row & (bucket_count - 1)
        w = hash(v, seed=i + seed_bias)
        buckets_A[i][rightmost_binary_1_position(w) - 1] += v
        buckets_B[i][rightmost_binary_1_position(w) - 1] += int(v) * int(v)
        buckets_C[i][rightmost_binary_1_position(w) - 1] += 1
        buckets_D[i][rightmost_binary_1_position(w) - 1] += Fu[v_num]
    return [buckets_A, buckets_B, buckets_C, buckets_D]

def merge_sketch(sketch_1, sketch_2, row_num, column_num):
    for row_index in range(row_num):
        for column_index in range(column_num):
            sketch_1[row_index][column_index] = sketch_1[row_index][column_index] + sketch_2[row_index][column_index]
    return sketch_1

def searching_sigma(epsilon, delta,sensitivity, num_provider):

    epsilon_d_1 = -math.sqrt(-2*math.log(delta)) + math.sqrt(-2*math.log(delta) + 2*epsilon)
    sigma_min = pow(10, -5)
    sigma_max = 100000000000.0
    a = 1
    while a>0:
        tau_d_min = 0.0
        tau_d_max = 0.0
        for k in range(1,num_provider-1):
            tau_d_min += math.exp(-2*k*math.pi*math.pi*sigma_min*sigma_min/(k+1))
        tau_d_min = tau_d_min*10
        for k in range(1,num_provider-1):
            tau_d_min += math.exp(-2*k*math.pi*math.pi*sigma_max*sigma_max/(k+1))
        tau_d_max = tau_d_max*10
        epsilon_d_1_min = min(math.sqrt(sensitivity*sensitivity/((num_provider-1)*sigma_min*sigma_min) + tau_d_min/2), sensitivity/(math.sqrt(num_provider-1)*sigma_min)+tau_d_min)
        epsilon_d_1_max = min(math.sqrt(sensitivity*sensitivity/((num_provider-1)*sigma_max*sigma_max) + tau_d_max/2), sensitivity/(math.sqrt(num_provider-1)*sigma_max)+tau_d_max)
        if a==1:
            if epsilon_d_1_min < epsilon_d_1:
                print("sigma_min is too huge")
                break
            if epsilon_d_1_max > epsilon_d_1:
                print("sigma_max is too small")
                break
        a = a+1
        if a>10000:
            print('there are too many iterations')
            break
        sigma_middle = (sigma_min + sigma_max)/2
        tau_d_middle = 0.0
        for k in range(1, num_provider - 1):
            tau_d_middle += math.exp(-2 * k * math.pi * math.pi * sigma_middle * sigma_middle / (k + 1))
        tau_d_middle = tau_d_middle * 10
        epsilon_d_1_middle = min(math.sqrt(sensitivity*sensitivity/((num_provider-1)*sigma_middle*sigma_middle) + tau_d_middle/2), sensitivity/(math.sqrt(num_provider-1)*sigma_middle)+tau_d_middle)
        if epsilon_d_1_middle > epsilon_d_1:
            sigma_min = sigma_middle
        else:
            sigma_max = sigma_middle
        if sigma_max - sigma_min < pow(10,-5):
            print("We find the sigma")
            print(sigma_max)
            break
    return sigma_max*sigma_max



def experiment_MPS_plus(num_elements, provider_data_num, distribution,epsilon,delta,
                                    m_bit=10,  num_provider=2):

    com_column = []
    max_error_all = []
    shuffle_distance_all = []
    num_prop_pre_all = []
    max_fre = 15
    sensitivity = 1
    sketch_type = 'MPS_plus'
    sigma2 = searching_sigma(epsilon, delta,sensitivity, num_provider)

    if distribution == 'homogeneous':
        raw_data_frequency = [[create_ID_data_set(num_elements, provider_data_num), create_homogeneous_frequency_set(provider_data_num)]  for index in range(num_provider)]
    else:
        raw_data_frequency = [[create_ID_data_set(num_elements, provider_data_num), create_heterogeneous_frequency_set(provider_data_num)] for index in range(num_provider)]
    data_frequency_distributed = [pd.concat(raw_data_frequency[id], axis=1) for id in
                                  range(len(raw_data_frequency))]
    result_pre = pd.concat(data_frequency_distributed, axis=0)
    result_pre_array = np.array(result_pre)
    result_pre.columns = ['data', 'frequency']
    result_pre = np.array(result_pre.groupby(by=['data'])['frequency'].sum().reset_index())[:, 1]
    frequency_analysis = pd.value_counts(result_pre).sort_index()
    num_prop_pre = np.zeros(max_fre)
    num_sum_pre = np.sum(np.array(frequency_analysis.values))
    index = 1
    for index_fre in frequency_analysis.index:
        if index_fre < max_fre:
            num_prop_pre[index_fre - 1] = frequency_analysis.values[index - 1] / num_sum_pre
        else:
            num_prop_pre[max_fre - 1] = frequency_analysis.values[index - 1] / num_sum_pre
        index = index + 1
    num_prop_pre_all.append(np.array(num_prop_pre))
    for provider_id in range(num_provider):
        data_pre = result_pre_array[provider_id * provider_data_num: (provider_id + 1) * provider_data_num, :]
        sketch_list = MPS_plus(data_pre[:,0], data_pre[:,1], m_bit)
        distributed_sketch_A = sketch_list[0]
        distributed_sketch_B = sketch_list[1]
        distributed_sketch_C = sketch_list[2]
        distributed_sketch_D = sketch_list[3]

        for sketch_index in range(4):
            for row_index in range(pow(2, m_bit)):
                for column_index in range(25):
                    f.write(str(sketch_list[sketch_index][row_index][column_index]))
                    f.write(' ')

        if provider_id == 0:
            sketch_A = distributed_sketch_A
            sketch_B = distributed_sketch_B
            sketch_C = distributed_sketch_C
            sketch_D = distributed_sketch_D
        else:
            sketch_A = merge_sketch(sketch_A, distributed_sketch_A, pow(2, m_bit), 25)
            sketch_B = merge_sketch(sketch_B, distributed_sketch_B, pow(2, m_bit), 25)
            sketch_C = merge_sketch(sketch_C, distributed_sketch_C, pow(2, m_bit), 25)
            sketch_D = merge_sketch(sketch_D, distributed_sketch_D, pow(2, m_bit), 25)

        distributed_sketch_A_df = pd.DataFrame(distributed_sketch_A)
        distributed_sketch_A_df.to_pickle(
            f'./MP-SPDZ/distributed_sketch_data/mbit={m_bit}_sketchtype={sketch_type}_proportion={proportion}_numprovider={num_provider}_{distribution}_numelement={num_elements}_distributed_sketchA_provider={provider_id}.pkl')
        distributed_sketch_B_df = pd.DataFrame(distributed_sketch_B)
        distributed_sketch_B_df.to_pickle(
            f'./MP-SPDZ/distributed_sketch_data/mbit={m_bit}_sketchtype={sketch_type}_proportion={proportion}_numprovider={num_provider}_{distribution}_numelement={num_elements}_distributed_sketchB_provider={provider_id}.pkl')
        distributed_sketch_C_df = pd.DataFrame(distributed_sketch_C)
        distributed_sketch_C_df.to_pickle(
            f'./MP-SPDZ/distributed_sketch_data/mbit={m_bit}_sketchtype={sketch_type}_proportion={proportion}_numprovider={num_provider}_{distribution}_numelement={num_elements}_distributed_sketchC_provider={provider_id}.pkl')
        distributed_sketch_D_df = pd.DataFrame(distributed_sketch_D)
        distributed_sketch_D_df.to_pickle(
            f'./MP-SPDZ/distributed_sketch_data/mbit={m_bit}_sketchtype={sketch_type}_proportion={proportion}_numprovider={num_provider}_{distribution}_numelement={num_elements}_distributed_sketchD_provider={provider_id}.pkl')
    result_fu = np.array([0 for iter_num in range(max_fre + 1)])
    result_f = [0 for iter_num in range(max_fre)]
    for row_num in range(np.array(sketch_C).shape[0]):
        desion_sample_sketch = sketch_C[row_num]
        for column_num in range(1, np.array(sketch_C).shape[1] + 1):
            if desion_sample_sketch[-column_num] != 0:
                break
        if int(sketch_A[row_num][-column_num]) * int(sketch_A[row_num][-column_num]) == int(
                sketch_B[row_num][-column_num]) * int(sketch_C[row_num][-column_num]):
            if sketch_D[row_num][-column_num] < 15:
                result_fu[int(sketch_D[row_num][-column_num])] += 1
            else:
                result_fu[15] += 1
        else:
            result_fu[0] += 1
    sum_f = sum(result_fu) - result_fu[0]
    print('the num of sampling is')
    print(sum_f)
    result_fu[0] = sum_f

    for provider_iter in range(num_provider):
        noise_list = []
        for noise_iter in range(16):
            noise = sample_dgauss(sigma2)
            noise_list.append(noise)
            f.write(str(noise))
            if provider_iter == num_provider - 1:
                if noise_iter == 16 - 1:
                    break
            f.write(' ')
        if provider_iter == 0:
            guassian_noise_array = np.array(noise_list)
        else:
            guassian_noise_array = guassian_noise_array + np.array(noise_list)
    print("The result of sampling without noise is:")
    print(result_fu)
    print(sum(result_fu))

    result_fu = result_fu + guassian_noise_array
    print("The result of sampling with noise is:")
    print(result_fu)

    for j_iter in range(len(result_f)):
        result_f[j_iter] = result_fu[j_iter + 1] / result_fu[0]
    num_prop_pre = np.array(num_prop_pre)
    result_f = np.array(result_f)
    error_f = np.abs(result_f - num_prop_pre)
    max_error_all.append(np.max(error_f))
    shuffle_distance_all.append(0.5 * np.sum(error_f))
    print('The ground-truth frequency histogram is')
    print(num_prop_pre)
    print('The estimated frequency histogram is')
    print(result_f)
    print('The shuffle distance is')
    print(0.5 * np.sum(error_f))
    return 0


def create_ID_data_set(num_unique, num_sample):
    universe_set = np.array([number for number in range(1, num_unique + 1)])
    numbers = np.random.choice(universe_set, num_sample, replace=False)
    numbers_df = pd.DataFrame(numbers)
    return numbers_df

def create_homogeneous_frequency_set(num_sample):
    frequency_set = poisson.rvs(mu=1, size=num_sample) + 1
    frequency_df = pd.DataFrame(frequency_set)
    return frequency_df



def create_heterogeneous_frequency_set(num_sample):
    frequency_set = nbinom.rvs(n=1, p=0.5, size=num_sample) + 1
    frequency_df = pd.DataFrame(frequency_set)
    return frequency_df



if __name__ == "__main__":


    proportion = 0.1
    distribution = 'homogeneous'
    num_provider = 10
    #distribution = 'heterogeneous'
    m_bit = 14
    epsilon = 0.1
    delta = pow(10, -12)
    time_consuming = []
    num_element_sample = pow(10, 5)
    experiment_MPS_plus(num_element_sample, int(num_element_sample * proportion), distribution,epsilon,delta,m_bit,num_provider=num_provider)

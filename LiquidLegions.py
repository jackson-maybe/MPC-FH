from scipy.stats import poisson, nbinom
from math import log, exp, floor
import hashlib
import numpy as np
import pandas as pd

m = int(100*pow(2,10))

a = 12
division_number = pow(2,64)

f_max = 15

def hash_256(x):
    x = str(x)
    h = hashlib.sha256(x.encode("gb2312"))
    hex_ = h.hexdigest()
    return int(hex_, base=16)

def noise_sampling(epsilon, delta, sensitivity, T):

    epsilon_eta = 0.35*epsilon
    delta_eta = 0.2*delta
    r = 1/T
    s = epsilon_eta/sensitivity
    middle = 2*T*sensitivity*(1+exp(epsilon_eta))/delta_eta
    mu = floor(log(middle)/s)
    X1 = nbinom.rvs(n=r, p=1-exp(-s), size=1)
    X2 = nbinom.rvs(n=r, p=1-exp(-s), size=1)

    while X1>mu or X2>mu:
        a=1
    X = X1-X2
    return X
def LL_sketch_gen(numbers, numbers_repetition):
    count_array = np.zeros(m)
    key_array = np.zeros(m)
    decision_array = np.zeros(m)

    for number_index,x in enumerate(numbers):
        f = hash_256(x)
        u = float(f% division_number)/division_number
        z = 1-log(exp(a)+u*(1-exp(a)))/a
        insert_index = floor(z*m)
        count_array[insert_index] += numbers_repetition[number_index]
        if decision_array[insert_index] == 0:
            key_array[insert_index] = f
            decision_array[insert_index] = 1
        elif decision_array[insert_index] == 1:
            if f != key_array[insert_index]:
                decision_array[insert_index] = 2
    return [count_array, key_array, decision_array]
def LL_sketch_merge(count_array1, key_array1, decision_array1,count_array2, key_array2, decision_array2):
    for index in range(m):
        count_array1[index] += count_array2[index]
        if decision_array1[index]!=0 and decision_array2[index]!=0 and (not(decision_array1[index] == decision_array2[index] and key_array1[index] == key_array2[index])):
            decision_array1[index] = 2
        elif decision_array1[index] == 0:
            key_array1[index] = key_array2[index]
            decision_array1[index] = decision_array2[index]
    return count_array1, key_array1,decision_array1
def frequency_estimation_noise(count_array, decision_array, noise_array):
    frequency_array = np.zeros(f_max)
    for index in range(m):
        if decision_array[index] == 1:
            frequency_index = int(min(count_array[index], f_max)-1)
            frequency_array[frequency_index] +=1
    sample_num = float(np.sum(frequency_array))
    for f_index in range(f_max):
        frequency_array[f_index] = frequency_array[f_index] + noise_array[f_index+1]
    sample_num = sample_num + noise_array[0]
    for f_index in range(f_max):
        frequency_array[f_index] = float(frequency_array[f_index])/sample_num
    return frequency_array,


def experiment_LiquidLegions(num_elements, provider_data_num,distribution,epsilon,delta,num_provider = 2):
    com_column = []
    max_error_all = []
    shuffle_distance_all = []
    error_f_all = []
    result_f_all = []
    num_prop_pre_all = []
    max_fre = 15
    sensitivity = 2
    T = 3

    if distribution == 'homogeneous':
        raw_data_frequency = [[create_ID_data_set(num_elements, provider_data_num),
                               create_homogeneous_frequency_set(provider_data_num)] for index in
                              range(num_provider)]
    else:
        raw_data_frequency = [[create_ID_data_set(num_elements, provider_data_num),
                               create_heterogeneous_frequency_set(provider_data_num)] for index in
                              range(num_provider)]
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
        sketch_list = LL_sketch_gen(data_pre[:,0],data_pre[:,1])
        count_array_middle = sketch_list[0]
        key_array_middle = sketch_list[1]
        decision_array_middle = sketch_list[2]
        if provider_id == 0:
            count_array = count_array_middle
            key_array = key_array_middle
            decision_array = decision_array_middle
        else:
            count_array,key_array,decision_array = LL_sketch_merge(count_array,key_array,decision_array,count_array_middle,key_array_middle, decision_array_middle)

    noise_array = np.zeros(f_max + 1)
    for f_iter in range(f_max):
        for noise_iter in range(T):
            noise_sample = noise_sampling(epsilon, delta, sensitivity, T)
            noise_array[f_iter + 1] += noise_sample
            noise_array[0] += noise_sample
    result_f= frequency_estimation_noise(count_array, decision_array, noise_array)
    num_prop_pre = np.array(num_prop_pre)
    result_f = np.array(result_f)
    error_f = np.abs(result_f - num_prop_pre)
    error_f_all.append(error_f)
    result_f_all.append(result_f)
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

proportion = 0.1
num_provider = 10
distribution = 'homogeneous'
#distribution = 'heterogeneous'


if __name__=="__main__":
    epsilon = 0.1
    delta = pow(10,-12)
    num_element_sample = pow(10,5)
    experiment_LiquidLegions(num_element_sample,  int(num_element_sample * proportion),distribution,epsilon,delta, num_provider=num_provider)


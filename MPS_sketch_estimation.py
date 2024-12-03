import numpy as np
from MPS_sketch_generation import MPS
import pandas as pd
import multiprocessing as mp
from queue import Queue
from discretegauss import sample_dgauss, searching_sigma

m_bits = [15360]
process_num = 5
proportion = 0.1
distribution = 'homogeneous'
num_provider = 5
#distribution = 'heterogeneous'





def merge_sketch_small(sketch_index_1, sketch_A_1, sketch_B_1, sketch_C_1, sketch_D_1, sketch_index_2, sketch_A_2, sketch_B_2, sketch_C_2, sketch_D_2, row_num):
    """Merge MPS sketches"""
    for row_index in range(row_num):
        if sketch_index_1[row_index] == sketch_index_2[row_index]:
            sketch_A_1[row_index] = sketch_A_1[row_index] + sketch_A_2[row_index]
            sketch_B_1[row_index] = sketch_B_1[row_index] + sketch_B_2[row_index]
            sketch_C_1[row_index] = sketch_C_1[row_index] + sketch_C_2[row_index]
            sketch_D_1[row_index] = sketch_D_1[row_index] + sketch_D_2[row_index]
        elif sketch_index_1[row_index] < sketch_index_2[row_index]:
            sketch_index_1[row_index] = sketch_index_2[row_index]
            sketch_A_1[row_index] = sketch_A_2[row_index]
            sketch_B_1[row_index] = sketch_B_2[row_index]
            sketch_C_1[row_index] = sketch_C_2[row_index]
            sketch_D_1[row_index] = sketch_D_2[row_index]
    return sketch_index_1, sketch_A_1, sketch_B_1, sketch_C_1, sketch_D_1




def experiment_MPS(num_elements, num_sample, data_provider_num, distribution, num_trials=1,
                                   sketch_type='MPS', m_bit=10, experiment_number=1, queue=[], num_provider=2):
    """Use the datasets from publishers to construct the MPS sketch, merge these sketches, and use the merged MPS sketch to estimate the frequency histogram"""
    max_fre = 15
    com_column = []
    provider_data_num = data_provider_num
    sample_num_all = []
    if num_elements < pow(10, 8) + 10:
        num_provider_loop = num_provider
    else:
        num_provider_loop = max(int(10 * num_provider * provider_data_num / num_sample), num_provider)

    sigma2 = searching_sigma(epsilon, delta, sensitivity, num_provider)

    for trial in range(num_trials):
        # load the files of user IDs and their local frequencies
        raw_data_frequency = [[pd.DataFrame(np.array(pd.read_pickle(
            f'./dataset/data_user_ID/datatype=equal_independent_universeset={num_elements}_numsample={num_sample}_sequence={index + trial}.pkl')).reshape(
            -1, 1)[0:provider_data_num, 0]),
                                pd.DataFrame(np.array(pd.read_pickle(
                                    f'./dataset/data_f_{distribution}/frequencytype={distribution}_universeset={num_elements}_numsample={num_sample}_sequence={index + trial}.pkl')).reshape(
                                    -1, 1)[0:provider_data_num, 0])]
                                for index in range(num_provider_loop)]

        data_frequency_distributed = [pd.concat(raw_data_frequency[id], axis=1) for id in
                                      range(len(raw_data_frequency))]

        result_pre = pd.concat(data_frequency_distributed, axis=0)
        result_pre_array = np.array(result_pre)
        result_pre.columns = ['data', 'frequency']

        result_pre = np.array(result_pre.groupby(by=['data'])['frequency'].sum().reset_index())[:, 1]
        frequency_analysis = pd.value_counts(result_pre).sort_index()


        # the ture frequency histogram
        fh_true = np.zeros(max_fre)
        num_sum_pre = np.sum(np.array(frequency_analysis.values))
        index = 1
        for index_fre in frequency_analysis.index:
            if index_fre < max_fre:
                fh_true[index_fre - 1] = frequency_analysis.values[index - 1] / num_sum_pre
            else:
                fh_true[max_fre - 1] = frequency_analysis.values[index - 1] / num_sum_pre
            index = index + 1

        com_column.append(trial)

        for provider_id in range(num_provider):
            provider_data_per = int(provider_data_num / process_num)
            process_list = []
            data_pre = result_pre_array[provider_id * provider_data_num: (provider_id + 1) * provider_data_num, :]
            if sketch_type == 'MPS':
                # Publishers construct their own MPS sketch
                for process_iter in range(process_num):
                    index = data_pre[provider_data_per * process_iter:provider_data_per * (process_iter + 1), 0]
                    label = data_pre[provider_data_per * process_iter:provider_data_per * (process_iter + 1), 1]
                    process_list.append(mp.Process(target=MPS, args=(index, label, m_bit, queue, process_iter)))

                for process_iter in range(process_num):
                    process_list[process_iter].start()
                for process_iter in range(process_num):
                    process_list[process_iter].join()

                for process_iter in range(process_num):
                    sketch_list = queue.get()
                    if process_iter == 0:
                        distributed_sketch_index = sketch_list[0]
                        distributed_sketch_A = sketch_list[1]
                        distributed_sketch_B = sketch_list[2]
                        distributed_sketch_C = sketch_list[3]
                        distributed_sketch_D = sketch_list[4]
                    else:
                        distributed_sketch_index, distributed_sketch_A, distributed_sketch_B, distributed_sketch_C, distributed_sketch_D = merge_sketch_small(distributed_sketch_index, distributed_sketch_A, distributed_sketch_B, distributed_sketch_C, distributed_sketch_D, sketch_list[0], sketch_list[1], sketch_list[2], sketch_list[3], sketch_list[4], m_bit)
            else:
                print("ERROR!")
            if provider_id == 0:
                sketch_index = distributed_sketch_index
                sketch_A = distributed_sketch_A
                sketch_B = distributed_sketch_B
                sketch_C = distributed_sketch_C
                sketch_D = distributed_sketch_D
            else:
                # Merge sketches from different publishers
                sketch_index, sketch_A, sketch_B, sketch_C, sketch_D = merge_sketch_small(sketch_index, sketch_A, sketch_B, sketch_C, sketch_D, distributed_sketch_index, distributed_sketch_A, distributed_sketch_B, distributed_sketch_C, distributed_sketch_D, m_bit)
        
        # Estimate the overall frequency histogram with DP noise
        fh_value_estimate = np.array([0 for iter_num in range(max_fre + 1)])
        fh_estimate = [0 for iter_num in range(max_fre)]

        # Estimate the overall frequency histogram without DP noise
        fh_value_noise_estimate = np.array([0 for iter_num in range(max_fre + 1)])
        fh_noise_estimate = [0 for iter_num in range(max_fre)]
        for row_num in range(m_bit):
            if int(sketch_A[row_num]) * int(sketch_A[row_num]) == int(sketch_B[row_num]) * int(sketch_C[row_num]):
                if sketch_D[row_num] < 15:
                    fh_value_estimate[int(sketch_D[row_num])] += 1
                else:
                    fh_value_estimate[15] += 1
            else:
                fh_value_estimate[0] += 1

        for provider_iter in range(num_provider):
            noise_list = []
            for noise_iter in range(16):
                noise = sample_dgauss(sigma2)
                noise_list.append(noise)
                if provider_iter == num_provider - 1:
                    if noise_iter == 16 - 1:
                        break

            if provider_iter == 0:
                guassian_noise_array = np.array(noise_list)
            else:
                guassian_noise_array = guassian_noise_array + np.array(noise_list)

        print('testing{}'.format(trial))
        sum_f = sum(fh_value_estimate) - fh_value_estimate[0]
        print('the number of sampling is')
        print(sum_f)
        sample_num_all.append(sum_f)
        for j_iter in range(len(fh_estimate)):
            fh_estimate[j_iter] = fh_value_estimate[j_iter + 1] / sum_f
        fh_true = np.array(fh_true)
        fh_estimate = np.array(fh_estimate)
        error_f = np.abs(fh_estimate - fh_true)
        print('Without DP noise, the shuffle distance is')
        print(0.5 * np.sum(error_f))

        fh_value_noise_estimate = fh_value_estimate + guassian_noise_array
        sum_f_noise = sum(fh_value_noise_estimate) - fh_value_noise_estimate[0]
        for j_iter in range(len(fh_noise_estimate)):
            fh_noise_estimate[j_iter] = fh_value_noise_estimate[j_iter + 1] / sum_f_noise
        error_f_noise = np.abs(fh_noise_estimate - fh_true)
        print('With DP noise, the shuffle distance is')
        print(0.5 * np.sum(error_f_noise))
        print("")
    
    return 0








if __name__ == "__main__":

    
    queue = mp.Manager().Queue(process_num)
    experiment_number = 'acc1_0'
    sketch_type = 'MPS'
    num_element_samples = [pow(10,5)]
    num_data_flies = [50]
    epsilon = 0.1
    delta = pow(10, -12)
    sensitivity = 1
    
    for iternum in range(len(num_element_samples)):
        for m_bit_iter in m_bits:
            num_element_sample = num_element_samples[iternum]
            experiment_MPS(num_element_sample, int(num_element_sample * 0.2),
                                           int(num_element_sample * proportion), distribution, 30, sketch_type,
                                           m_bit_iter, experiment_number, queue, num_provider=num_provider)
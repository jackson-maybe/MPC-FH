import mmh3


def hash(x, seed=9):
    """The hash function used in the MPS sketch"""
    x = str(x)
    h = mmh3.hash(x, seed=seed)
    return h



def rightmost_binary_1_position(num, column):
    """The (number of trailing zeroes in the binary representation of num) + 1"""
    i = 0
    while (num >> i) & 1 == 0:
        i += 1
        if i == column-1:
            break
    return i + 1



def MPS(numbers, Fu, bits_for_bucket_index, queue,process_iter, seed_bias=0):
    """Insert data into the MPS sketch"""
    bucket_count = bits_for_bucket_index
    seed_bias = seed_bias * bucket_count
    buckets_index = [0  for row_index in range(bucket_count) ]
    buckets_A = [0  for row_index in range(bucket_count) ]
    buckets_B = [0  for row_index in range(bucket_count) ]
    buckets_C = [0  for row_index in range(bucket_count) ]
    buckets_D = [0  for row_index in range(bucket_count) ]

    # Set up the data for "stochastic averaging"
    for v_num, v in enumerate(numbers):
        hash_row = hash(v, seed=bucket_count + seed_bias+15360)
        i = hash_row & (bucket_count - 1)
        w = rightmost_binary_1_position(hash(v, seed=i + seed_bias+15360), 36)
        if w == buckets_index[i]:
            buckets_A[i] += v
            buckets_B[i] += int(v) * int(v)
            buckets_C[i] += 1
            buckets_D[i] += Fu[v_num]
        elif w > buckets_index[i]:
            buckets_index[i] = w
            buckets_A[i] = v
            buckets_B[i] = int(v) * int(v)
            buckets_C[i] = 1
            buckets_D[i] = Fu[v_num]

    queue.put([buckets_index, buckets_A, buckets_B, buckets_C, buckets_D])

    return 0


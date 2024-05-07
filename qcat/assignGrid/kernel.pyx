import numpy as np
cimport numpy as np
import math

def generate_mnl(np.ndarray[np.float64_t, ndim=2] alpha, double adjust_rcut):
    cdef int l_min, l_max, n_min, n_max, m_min, m_max
    cdef int l, n, m
    cdef double l_core, n_core_p, n_core_n, m_core_p, m_core_n
    cdef list mnl_list = []

    l_core = adjust_rcut / alpha[2, 2]
    l_min = int(math.floor(min(l_core, -l_core)))
    l_max = int(math.ceil(max(l_core, -l_core)))

    for l in range(l_min, l_max + 1):
        n_core_p = (adjust_rcut - l * alpha[2, 1]) / alpha[1, 1]
        n_core_n = n_core_p - 2.0 * adjust_rcut / alpha[1, 1]
        n_min = int(math.floor(min(n_core_p, n_core_n)))
        n_max = int(math.ceil(max(n_core_p, n_core_n)))

        for n in range(n_min, n_max + 1):
            m_core_p = (adjust_rcut - l * alpha[2, 0] - n * alpha[1, 0]) / alpha[0, 0]
            m_core_n = m_core_p - 2.0 * adjust_rcut / alpha[0, 0]
            m_min = int(math.floor(min(m_core_p, m_core_n)))
            m_max = int(math.ceil(max(m_core_p, m_core_n)))

            for m in range(m_min, m_max + 1):
                mnl_list.append([m, n, l])
    return mnl_list

__author__ = 'carlos.ginestra'

from scipy.sparse import csr_matrix
from data_preparation import DataDAO

class TestingFactory():

    @staticmethod
    def get_test_data(limit=0):
        matrix = []
        test_data = DataDAO.get_test_vector()
        for ind, vector in enumerate(test_data):
            if limit != 0 and ind >= limit:
                break
            matrix.append(vector)
        # sparse = csr_matrix(scaled_matrix)

        return matrix
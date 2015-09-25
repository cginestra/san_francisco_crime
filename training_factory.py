__author__ = 'carlos.ginestra'

import numpy
from scipy.sparse import csr_matrix
from data_preparation import DataDAO
from sklearn.preprocessing import scale
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import MinMaxScaler


CATEGORIES = ["ARSON","ASSAULT","BAD CHECKS","BRIBERY","BURGLARY","DISORDERLY CONDUCT",
              "DRIVING UNDER THE INFLUENCE","DRUG/NARCOTIC","DRUNKENNESS","EMBEZZLEMENT",
              "EXTORTION","FAMILY OFFENSES","FORGERY/COUNTERFEITING","FRAUD","GAMBLING",
              "KIDNAPPING","LARCENY/THEFT","LIQUOR LAWS","LOITERING","MISSING PERSON",
              "NON-CRIMINAL","OTHER OFFENSES","PORNOGRAPHY/OBSCENE MAT","PROSTITUTION",
              "RECOVERED VEHICLE","ROBBERY","RUNAWAY","SECONDARY CODES","SEX OFFENSES FORCIBLE",
              "SEX OFFENSES NON FORCIBLE","STOLEN PROPERTY","SUICIDE","SUSPICIOUS OCC","TREA",
              "TRESPASS","VANDALISM","VEHICLE THEFT","WARRANTS","WEAPON LAWS"]

class TrainingFactory():

    @staticmethod
    def get_category_vector(category):

        ## utilizar MultiLabelBinarizaer()

        if category not in CATEGORIES:
            print("Unknown category")
            raise Exception
        dict_cat = {}
        for ind, item in enumerate(CATEGORIES):
            dict_cat[item]=ind

        cat_vector = numpy.zeros(len(CATEGORIES))
        cat_vector[dict_cat[category]] = 1
        return cat_vector

    @staticmethod
    def build_sparse_matrix_input(limit=0):
        matrix = []
        datos = DataDAO.get_train_vector()
        for ind, vector in enumerate(datos):
            if limit != 0 and ind >= limit:
                break
            matrix.append(vector)


        scaled_matrix = scale(matrix, axis=0)
        sparse = csr_matrix(scaled_matrix)

        return sparse


    @staticmethod
    def build_sparse_matrix_target(limit=0):
        targets_vector = DataDAO.get_targets()
        targets_matrix = []
        for ind, item in enumerate(targets_vector):
            if limit != 0 and ind >= limit:
                break
            multiclass_vector = TrainingFactory.get_category_vector(item)
            targets_matrix.append(multiclass_vector)

        sparse = csr_matrix(targets_matrix)
        return sparse

    @staticmethod
    def build_target_vector_by_category(category, limit=0):
        targets_vector = DataDAO.get_targets()
        targets_cat = []
        for ind, item in enumerate(targets_vector):
            if limit != 0 and ind >= limit:
                break
            if item == category:
                targets_cat.append(1)
            else:
                targets_cat.append(0)

        return targets_cat

    @staticmethod
    def get_training_data_by_category(category, limit=0):
        limit_pos = limit*0.2
        limit_neg = limit*0.8
        N_pos = DataDAO.count_training_data_by_category(category)
        if N_pos < limit_pos:
            limit_pos = N_pos
            limit_neg = N_pos*5

        training_data = []
        training_target = []
        positive = DataDAO.get_training_data_by_category(category)
        for ind, sample in enumerate(positive):
            if limit != 0 and ind >= limit_pos:
                break
            training_data.append(sample)
            training_target.append(1)
        negative = DataDAO.get_training_data_by_other_categories(category)
        for ind, sample in enumerate(negative):
            if limit != 0 and ind >= limit_neg:
                break
            training_data.append(sample)
            training_target.append(0)

        scaler = MinMaxScaler()
        training_data_scaled = scaler.fit_transform(training_data)

        # training_data_scaled = scale(training_data,axis=0)
        tr_data_sparse = csr_matrix(training_data_scaled)

        return tr_data_sparse, training_target, scaler


if __name__ == "__main__":

    # TrainingFactory.build_sparse_matrix_input()
    TrainingFactory.build_sparse_matrix_target()
    print TrainingFactory.get_category_vector("BRIBERY")
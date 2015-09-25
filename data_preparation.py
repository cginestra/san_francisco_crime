__author__ = 'carlos.ginestra'

import csv
import datetime
import numpy

class DataDAO():

    """
    Train example:
    ['Dates', 'Category', 'Descript', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y']
    ['2015-05-13 23:53:00', 'WARRANTS', 'WARRANT ARREST', 'Wednesday', 'NORTHERN', 'ARREST, BOOKED', 'OAK ST / LAGUNA ST', '-122.425891675136', '37.7745985956747']


    Test example:
    ['Id', 'Dates', 'DayOfWeek', 'PdDistrict', 'Address', 'X', 'Y']
    ['0', '2015-05-10 23:59:00', 'Sunday', 'BAYVIEW', '2000 Block of THOMAS AV', '-122.39958770418998', '37.7350510103906']

    """

    @staticmethod
    def get_data_from_csv(filename):
        skip = 1
        path = 'data/'+filename
        with open(path,'rb') as csvfile:
            data = csv.reader(csvfile)
            for ind, row in enumerate(data):
                if row[0] != 'Id' and row[0] != 'Dates' and ind % skip == 0:
                    # print row
                    # print DataDAO.time_features_test(row)
                    yield row

    @staticmethod
    def time_features_test(row):

        date_str = row[1]
        dayofweek = DataDAO.weekday_to_number(row[2])
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        day_time_seconds = date.hour*3600+date.minute*60+date.second
        day_year = date.timetuple().tm_yday

        return day_year, day_time_seconds, dayofweek

    @staticmethod
    def time_features_train(row):

        date_str = row[0]
        dayofweek = DataDAO.weekday_to_number(row[3])
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        day_time_seconds = date.hour*3600+date.minute*60+date.second
        day_year = date.timetuple().tm_yday

        return day_year, day_time_seconds, dayofweek

    @staticmethod
    def weekday_to_number(weekday):
        if weekday == "Monday":
            return 1
        elif weekday == "Tuesday":
            return 2
        elif weekday == "Wednesday":
            return 3
        elif weekday == "Thursday":
            return 4
        elif weekday == "Friday":
            return 5
        elif weekday == "Saturday":
            return 6
        elif weekday == "Sunday":
            return 7
        else:
            raise Exception

    @staticmethod
    def get_coordinates_tr(row):
        return float(row[7]), float(row[8])

    @staticmethod
    def get_coordinates_test(row):
        return float(row[5]), float(row[6])

    @staticmethod
    def get_coordinates_tr_by_category(category):
        output = []
        data = DataDAO.get_data_by_category(category)
        for row in data:
            coord = DataDAO.get_coordinates_tr(row)
            output.append(coord)
        return numpy.array(output)

    @staticmethod
    def show_data_sample(filename, limit=10):
        data = DataDAO.get_data_from_csv(filename)
        count = 0
        while count < limit:
            print data.next()
            count += 1

    @staticmethod
    def get_train_vector():
        data = DataDAO.get_data_from_csv('train.csv')
        for row in data:
            time = DataDAO.time_features_train(row)
            coord = DataDAO.get_coordinates_tr(row)
            yield time + coord

    @staticmethod
    def get_test_vector():
        data = DataDAO.get_data_from_csv('test.csv')
        for row in data:
            time = DataDAO.time_features_test(row)
            coord = DataDAO.get_coordinates_test(row)
            yield time + coord

    @staticmethod
    def get_targets():
        targets = []
        data = DataDAO.get_data_from_csv('train.csv')
        for row in data:
            targets.append(row[1])
        return targets

    @staticmethod
    def get_training_data_by_category(category):
        data = DataDAO.get_data_from_csv('train.csv')
        for row in data:
            if row[1]==category:
                time = DataDAO.time_features_train(row)
                coord = DataDAO.get_coordinates_tr(row)
                yield time + coord

    @staticmethod
    def count_training_data_by_category(category):
        data = DataDAO.get_data_from_csv('train.csv')
        count = 0
        for row in data:
            if row[1] == category:
                count += 1
        return count

    @staticmethod
    def get_training_data_by_other_categories(category):
        data = DataDAO.get_data_from_csv('train.csv')
        for row in data:
            if row[1]!=category:
                time = DataDAO.time_features_train(row)
                coord = DataDAO.get_coordinates_tr(row)
                yield time + coord

    @staticmethod
    def get_time_features_by_category(category):
        output = []
        data = DataDAO.get_data_by_category(category)
        for row in data:
            time = DataDAO.time_features_train(row)
            output.append(time)
        return numpy.array(output)

    @staticmethod
    def get_data_by_category(category):
        data = DataDAO.get_data_from_csv('train.csv')
        for row in data:
            if row[1] == category:
                yield row

if __name__ == "__main__":
    # DataDAO.get_data_from_csv('test.csv')
    DataDAO.show_data_sample(filename='test.csv')
    DataDAO.show_data_sample(filename='train.csv')

    # DataDAO.get_train_vector()
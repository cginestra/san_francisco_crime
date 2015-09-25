__author__ = 'carlos.ginestra'

from data_preparation import DataDAO
import matplotlib.pyplot as plt
import numpy


class Visualization():

    @staticmethod
    def plot_crimes_by_coordinates():
        category = "DRUNKENNESS"
        data = DataDAO.get_data_from_csv('train.csv')

        # x=numpy.array([])
        # y=numpy.array([])
        x = []
        y = []
        sf_coordinates_x = [-122.6, -122.35]
        sf_coordinates_y = [35, 40 ]


        for ind,raw in enumerate(data):
            if raw[0] != 'Dates' and raw[1] == category:
                coord = DataDAO.get_coordinates_tr(raw)
                if sf_coordinates_x[0] < coord[0] < sf_coordinates_x[1] and sf_coordinates_y[0] < coord[1] < sf_coordinates_y[1]:
                    x.append(coord[0])
                    y.append(coord[1])
                # numpy.append(x,[coord[0]])
                # numpy.append(y,[coord[1]])
                # if ind > 100000:
                #     break

        #heat map

        # heatmap, xedges, yedges = numpy.histogram2d(x, y, bins=50)
        # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        # plt.subplot(2,1,2)
        # plt.imshow(heatmap, extent=extent)

        #scatter plot
        # plt.subplot(2,1,1)
        plt.scatter(x,y,s=1)

        plt.show()




    @staticmethod
    def plot_crime_category_by_date_time(category):
        datos = DataDAO.get_time_features_by_category(category)

        dia_year = datos[:,0]
        segundo_del_dia = datos[:,1]
        dia_semana = datos[:,2]

        plt.figure()
        plt.suptitle(category)

        plt.subplot(3,1,1)
        plt.hist(dia_semana, bins=7)
        plt.title("Crime by day of week")

        plt.subplot(3,1,2)
        plt.hist(dia_year, bins=12)
        plt.title("Crime by month")

        plt.subplot(3,1,3)
        plt.hist(segundo_del_dia, bins=24)
        plt.title("Crime by hour")

        plt.show()
        return plt






if __name__ == "__main__":
    Visualization.plot_crimes_by_coordinates()
    # Visualization.plot_crime_category_by_date_time("DRUNKENNESS")
    # Visualization.plot_crime_category_by_date_time("FAMILY OFFENSES")
    # Visualization.plot_crime_category_by_date_time("VEHICLE THEFT")
__author__ = 'carlos.ginestra'


# This shows how to read the text representing a map of Chicago in numpy, and put it on a plot in matplotlib.
# This example doesn't make it easy for you to put other data in lat/lon coordinates on the plot.
# Hopefully someone else can add an example showing how to do that? You'll need to know the bounding box of this map:
#    ll.lon     ll.lat   ur.lon     ur.lat
#    -122.52469 37.69862 -122.33663 37.82986

import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap
from osmapi import OsmApi


def plot_sf_void_map():

    mapdata = np.loadtxt("./data/sf_map_copyright_openstreetmap_contributors.txt")
    plt.imshow(mapdata, cmap = plt.get_cmap('gray'))
    plt.savefig('map.png')

    plt.show()




def example_maps():

    # read in topo data (on a regular lat/lon grid)
    etopo = np.loadtxt('./etopo20data.gz')
    lons  = np.loadtxt('./etopo20lons.gz')
    lats  = np.loadtxt('./etopo20lats.gz')
    # create Basemap instance for Robinson projection.
    m = Basemap(projection='robin',lon_0=0.5*(lons[0]+lons[-1]))
    # compute map projection coordinates for lat/lon grid.
    x, y = m(*np.meshgrid(lons,lats))
    # make filled contour plot.
    cs = m.contourf(x,y,etopo,30,cmap=plt.cm.jet)
    m.drawcoastlines() # draw coastlines
    m.drawmapboundary() # draw a line around the map region
    m.drawparallels(np.arange(-90.,120.,30.),labels=[1,0,0,0]) # draw parallels
    m.drawmeridians(np.arange(0.,420.,60.),labels=[0,0,0,1]) # draw meridians
    plt.title('Robinson Projection') # add a title
    plt.show()

def prueba_osmapi():

    MyApi = OsmApi()
    print MyApi.NodeGet(123)




if __name__ == "__main__":

    # example_maps()
    prueba_osmapi()
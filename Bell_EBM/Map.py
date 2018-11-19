# Author: Taylor Bell
# Last Update: 2018-11-19

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from . import H2_Dissociation_Routines as h2

class Map(object):
    """A map.

    Attributes:
        lat (ndarray, optional): The unique latitude values in degrees.
        latGrid (ndarray): The latitude grid in degrees.
        lon (ndarray, optional): The unique longitude values in degrees.
        lonGrid (ndarray): The longitude grid in degrees.
        nside (int): A parameter that sets the resolution of the map.
        pixArea (ndarray): The area of each pixel.
        time (float): Time of map in days.
        useHealpix (bool): Whether the planet's map uses a healpix grid.
        values (ndarray): The temperature map values.
    
    """
    
    def __init__(self, nside=16, values=None, time=0, useHealpix=False):
        """Initialization funciton.

        Args:
            nside (int, optional): A parameter that sets the resolution of the map.
            values(ndarray, optional): The temperature map values.
            time (float, optional): Time of map in days.
            useHealpix (bool, optional): Whether the planet's map uses a healpix grid.

        """
        
        self.time = time
        self.useHealpix = useHealpix
        if not self.useHealpix:
            self.nside = np.rint(nside).astype(int)
            self.npix = self.nside*(2*self.nside)

            self.dlat = 180/self.nside
            self.lat = np.linspace(-90+self.dlat/2, 90-self.dlat/2, self.nside)
            latTop = self.lat+self.dlat/2
            latBot = self.lat-self.dlat/2

            self.dlon = 360/(self.nside*2)
            self.lon = np.linspace(-180+self.dlon/2, 180-self.dlon/2, self.nside*2)
            lonRight = self.lon+self.dlon/2
            lonLeft = self.lon-self.dlon/2

            latArea = np.abs(2*np.pi*(np.sin(latTop*np.pi/180)-np.sin(latBot*np.pi/180)))
            areas = latArea.reshape(1,-1)*(np.abs(lonRight-lonLeft)/360).reshape(-1,1)
            latGrid, lonGrid = np.meshgrid(self.lat, self.lon)

            self.pixArea = areas.reshape(1, -1)
            self.latGrid = latGrid.reshape(1, -1)
            self.lonGrid = lonGrid.reshape(1, -1)
        else:
            # Tuck this away in here so it isn't a required package for those that don't want to use it
            import healpy as hp
            global hp
            
            self.nside = np.rint(nside).astype(int)
            self.npix = hp.nside2npix(self.nside)
            self.pixArea = np.array([hp.nside2pixarea(self.nside)])

            coords = np.empty((self.npix,2))
            for i in range(self.npix):
                coords[i,:] = np.array(hp.pix2ang(self.nside, i))*180/np.pi
            lon = coords[:,1]
            lat = coords[:,0]-90
            self.latGrid = lat.reshape(1, -1)
            self.lonGrid = lon.reshape(1, -1)
        
        if values is not None:
            values = np.array([values]).reshape(-1)
            if values.size < self.npix:
                print('Error: Too few map values ('+str(values.size)+'!='+str(self.npix)+')')
                return None
            elif values.size > self.npix:
                print('Error: Too many map values ('+str(values.size)+'!='+str(self.npix)+')')
                return None
            else:
                self.values = values
        else:
            self.values = np.zeros(self.npix)
    
    def set_values(self, values, time=None):
        """Set the temperature map.
        
        Args:
            values (ndarray): The map temperatures (in K) with a size of self.npix.
            time (float, optional): Time of map in days.
        
        """
        
        values = np.array([values]).reshape(-1, order='F')
        if values.size < self.npix:
            print('Error: Too few map values ('+str(values.size)+' < '+str(self.npix)+')')
            return None
        elif values.size > self.npix:
            print('Error: Too many map values ('+str(values.size)+' > '+str(self.npix)+')')
            return None
        else:
            if time is not None:
                self.time = time
            self.values = values
            
    def load_custom_map(self, values, time=None, lat=None, lon=None, latGrid=None, lonGrid=None, pixArea=None):
        """Set the whole map object.
        
        Args:
            values (ndarray): The map temperatures (in K) with a size of self.npix.
            time (float, optional): Time of map in days.
            lat (ndarray, optional): The unique latitude values in degrees.
            lon (ndarray, optional): The unique longitude values in degrees.
            latGrid (ndarray, optional): The latitude of every pixel in degrees.
            lonGrid (ndarray, optional): The longitude of every pixel in degrees.
            pixArea (ndarray, optional): The angular area of each pixel in steradians.
        
        """
        
        values = np.array([values]).reshape(-1, order='F')
        if lat is None and lon is None and latGrid is None and lonGrid is None:
            npix = self.npix
        elif lat is not None and lon is not None:
            npix = len(lon)*len(lat)
        elif latGrid is not None and lonGrid is not None:
            npix = latGrid.size
        
        if values.size < npix:
            print('Error: Too few map values ('+str(values.size)+' < '+str(npix)+')')
            return None
        elif values.size > npix:
            print('Error: Too many map values ('+str(values.size)+' > '+str(npix)+')')
            return None
        else:
            self.values = values
            if time is not None:
                self.time = time
            else:
                self.time = 0
            if lat is not None:
                self.lat = lat
            if lon is not None:
                self.lon = lon
            if latGrid is not None:
                self.latGrid = latGrid
            if lonGrid is not None:
                self.lonGrid = lonGrid
            if pixArea is not None:
                self.pixArea = pixArea
            
            if lat is None and lon is None and latGrid is None and lonGrid is None:
                self.dlat = 180/self.nside
                self.lat = np.linspace(-90+self.dlat/2, 90-self.dlat/2, self.nside)
                latTop = self.lat+self.dlat/2
                latBot = self.lat-self.dlat/2

                self.dlon = 360/(self.nside*2)
                self.lon = np.linspace(-180+self.dlon/2, 180-self.dlon/2, self.nside*2)
                lonRight = self.lon+self.dlon/2
                lonLeft = self.lon-self.dlon/2

                latArea = np.abs(2*np.pi*(np.sin(latTop*np.pi/180)-np.sin(latBot*np.pi/180)))
                areas = latArea.reshape(1,-1)*(np.abs(lonRight-lonLeft)/360).reshape(-1,1)
                latGrid, lonGrid = np.meshgrid(self.lat, self.lon)

                self.pixArea = areas.reshape(1, -1)
                self.latGrid = latGrid.reshape(1, -1)
                self.lonGrid = lonGrid.reshape(1, -1)
            elif lat is not None and lon is not None and latGrid is None and lonGrid is None:
                self.dlat = 180./len(self.lat)
                latTop = self.lat+self.dlat/2
                latBot = self.lat-self.dlat/2

                self.dlon = 360./len(self.lon)
                lonRight = self.lon+self.dlon/2
                lonLeft = self.lon-self.dlon/2

                latArea = np.abs(2*np.pi*(np.sin(latTop*np.pi/180)-np.sin(latBot*np.pi/180)))
                areas = latArea.reshape(1,-1)*(np.abs(lonRight-lonLeft)/360).reshape(-1,1)
                latGrid, lonGrid = np.meshgrid(self.lat, self.lon)

                self.pixArea = areas.reshape(1, -1)
                self.latGrid = latGrid.reshape(1, -1)
                self.lonGrid = lonGrid.reshape(1, -1)
            elif lat is None and lon is None and latGrid is not None and lonGrid is not None and not self.useHealpix:
                self.lat = np.unique(self.latGrid)
                self.lon = np.unique(self.lonGrid)

                if pixArea is None:
                    self.dlat = 180./len(self.lat)
                    latTop = self.lat+self.dlat/2
                    latBot = self.lat-self.dlat/2

                    self.dlon = 360./len(self.lon)
                    lonRight = self.lon+self.dlon/2
                    lonLeft = self.lon-self.dlon/2

                    latArea = np.abs(2*np.pi*(np.sin(latTop*np.pi/180)-np.sin(latBot*np.pi/180)))
                    areas = latArea.reshape(1,-1)*(np.abs(lonRight-lonLeft)/360).reshape(-1,1)
                    self.pixArea = areas.reshape(1, -1)
    
    
    def plot_map(self, refLon=None):
        """A convenience routine to plot the temperature map
        
        Args:
            refLon (float, optional): The sub-stellar longitude used to de-rotate the map.
        
        Returns:
            figure: The figure containing the plot.
        
        """
        
        if not self.useHealpix:
            tempMap = self.values.reshape((self.lat.size, self.lon.size), order='F')
            if refLon is not None:
                rollCount = -(np.where(np.abs(self.lon-refLon) < self.dlon/2+1e-6)[0][0]-int(self.lon.size/2))
                tempMap = np.roll(tempMap, rollCount, axis=1)

            im = plt.imshow(tempMap, cmap='inferno', extent=(-180,180,-90,90))
            plt.xlabel(r'Longitude', fontsize='large')
            plt.ylabel(r'Latitude', fontsize='large')
            plt.xticks([-180,-90,0,90,180], fontsize='large')
            plt.yticks([-90,-45,0,45,90], fontsize='large')
            cbar = plt.colorbar(orientation='vertical', fraction=0.05, pad = 0.05, aspect=9)
        else:
            current_cmap = matplotlib.cm.get_cmap('inferno')
            current_cmap.set_bad(color='white')
            if refLon is None:
                refLon = 0
            im = hp.orthview(self.values, flip='geo', cmap='inferno', min=0,
                             rot=(refLon, 0, 0), return_projected_map=True, cbar=None)
            plt.clf()
            plt.imshow(im, cmap='inferno')
            plt.xticks([])
            plt.yticks([])
            plt.setp(plt.gca().spines.values(), color='none')
            cbar = plt.colorbar(orientation='horizontal',fraction=0.075, pad = 0.05)
        
        cbar.ax.tick_params(labelsize='large')
        cbar.set_label('Temperature (K)', fontsize='x-large')
        return plt.gcf()
    
    def plot_dissociation(self, refLon=None):
        """A convenience routine to plot the H2 dissociation map.
        
        Args:
            refLon (float, optional): The sub-stellar longitude used to de-rotate the map.
        
        Returns:
            figure: The figure containing the plot.
        
        """
        
        if not self.useHealpix:
            dissMap = h2.dissFracApprox(self.values.reshape((self.lat.size, self.lon.size), order='F'))*100.
            if refLon is not None:
                rollCount = -(np.where(np.abs(self.lon-refLon) < self.dlon/2+1e-6)[0][0]-int(self.lon.size/2))
                dissMap = np.roll(dissMap, rollCount, axis=1)

            plt.imshow(dissMap, cmap='inferno', extent=(-180,180,-90,90), vmin=0)
            plt.xlabel(r'Longitude', fontsize='large')
            plt.ylabel(r'Latitude', fontsize='large')
            plt.xticks([-180,-90,0,90,180], fontsize='large')
            plt.yticks([-90,-45,0,45,90], fontsize='large')
            cbar = plt.colorbar(orientation='vertical', fraction=0.05, pad = 0.05, aspect=9)
        else:
            current_cmap = matplotlib.cm.get_cmap('inferno')
            current_cmap.set_bad(color='white')
            if refLon is None:
                refLon = 0
            im = hp.orthview(h2.dissFracApprox(self.values)*100., flip='geo', cmap='inferno', min=0,
                             rot=(refLon, 0, 0), return_projected_map=True, cbar=None)
            plt.clf()
            plt.imshow(im, cmap='inferno', vmin=0)
            plt.xticks([])
            plt.yticks([])
            plt.setp(plt.gca().spines.values(), color='none')
            cbar = plt.colorbar(orientation='horizontal', fraction=0.075, pad = 0.05)
        
        cbar.ax.tick_params(labelsize='large')
        cbar.set_label('Dissociation Fraction (%)', fontsize='x-large')
        return plt.gcf()

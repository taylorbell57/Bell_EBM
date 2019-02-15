# Author: Taylor Bell
# Last Update: 2019-02-15

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from . import H2_Dissociation_Routines as h2

class Map(object):
    """A map.

    Attributes:
        dissValues (ndarray): The H2 dissociation fraction values for the map.
        lat (ndarray, optional): The unique latitude values in degrees.
        lat_radians (ndarray, optional): The unique latitude values in radians.
        latGrid (ndarray): The latitude grid in degrees.
        latGrid_radians (ndarray): The latitude grid in radians.
        lon (ndarray, optional): The unique longitude values in degrees.
        lon_radians (ndarray, optional): The unique longitude values in radians.
        lonGrid (ndarray): The longitude grid in degrees.
        lonGrid_radians (ndarray): The longitude grid in radians.
        nlat (int, optional): The number of latitudinal cells to use for rectangular maps.
        nlon (int, optional): The number of longitudinal cells to use for rectangular maps.
        nside (int, optional): A parameter that sets the resolution of healpy maps.
        pixArea (ndarray): The area of each pixel.
        time (float): Time of map in days.
        useHealpix (bool): Whether the planet's map uses a healpix grid.
        values (ndarray): The temperature map values.
    
    """
    
    def __init__(self, values=None, dissValues=None, time=0., nlat=16, nlon=None):
        """Initialization funciton.

        Args:
            
            values(ndarray, optional): The temperature map values.
            dissValues(ndarray, optional): The H2 dissociation fraction values for the map.
            time (float, optional): Time of map in days.
            nlat (int, optional): The number of latitudinal cells to use for rectangular maps.
            nlon (int, optional): The number of longitudinal cells to use for rectangular maps.
                If nlon==None, uses 2*nlat.
            

        """
        
        self.time = time
        
        self.nlat = int(nlat)
        if nlon==None:
            self.nlon = int(2*self.nlat)
        else:
            self.nlon = int(nlon)
        self.npix = int(self.nlat*self.nlon)

        self.dlat = 180./self.nlat
        self.lat = np.linspace(-90.+self.dlat/2., 90.-self.dlat/2., self.nlat, endpoint=True)
        latTop = self.lat+self.dlat/2.
        latBot = self.lat-self.dlat/2.

        self.dlon = 360./self.nlon
        self.lon = np.linspace(-180.+self.dlon/2., 180.-self.dlon/2., self.nlon, endpoint=True)
        lonRight = self.lon+self.dlon/2.
        lonLeft = self.lon-self.dlon/2.

        latArea = np.abs(2.*np.pi*(np.sin(latTop*np.pi/180.)-np.sin(latBot*np.pi/180.)))
        areas = latArea.reshape(-1,1)*(np.abs(lonRight-lonLeft)/360.).reshape(1,-1)
        lonGrid, latGrid = np.meshgrid(self.lon, self.lat)

#         self.pixArea = areas.reshape(1, -1)
#         self.latGrid = latGrid.reshape(1, -1)
#         self.lonGrid = lonGrid.reshape(1, -1)
        self.pixArea = areas#[np.newaxis,:]
        self.latGrid = latGrid#[np.newaxis,:]
        self.lonGrid = lonGrid#[np.newaxis,:]
        
        self.lat_radians = self.lat*np.pi/180
        self.lon_radians = self.lon*np.pi/180
        self.latGrid_radians = self.latGrid*np.pi/180.
        self.lonGrid_radians = self.lonGrid*np.pi/180.
        
        
        if values is not None:
            if values.size < self.npix:
                print('Error: Too few map values ('+str(values.size)+'!='+str(self.npix)+')')
                return None
            elif values.size > self.npix:
                print('Error: Too many map values ('+str(values.size)+'!='+str(self.npix)+')')
                return None
            else:
                self.values = values
        else:
            self.values = np.zeros_like(self.lonGrid)
            
        if dissValues is not None:
            if dissValues.size < self.npix:
                print('Error: Too few map values ('+str(dissValues.size)+'!='+str(self.npix)+')')
                return None
            elif dissValues.size > self.npix:
                print('Error: Too many map values ('+str(dissValues.size)+'!='+str(self.npix)+')')
                return None
            else:
                self.dissValues = dissValues
        else:
            self.dissValues = np.zeros_like(self.lonGrid)
    
    def set_values(self, values, time=None, dissValues=None):
        """Set the temperature map.
        
        Args:
            values (ndarray): The map temperatures (in K) with a size of self.npix.
            time (float, optional): Time of map in days.
            dissValues(ndarray, optional): The H2 dissociation fraction values for the map.
        
        """
        
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
        
        if dissValues is not None:
            if dissValues.size < self.npix:
                print('Error: Too few map values ('+str(dissValues.size)+'!='+str(self.npix)+')')
                return None
            elif dissValues.size > self.npix:
                print('Error: Too many map values ('+str(dissValues.size)+'!='+str(self.npix)+')')
                return None
            else:
                self.dissValues = dissValues
    
    def plot_map(self, refLon=None):
        """A convenience routine to plot the temperature map
        
        Args:
            refLon (float, optional): The centre longitude used to rotate the map.
        
        Returns:
            figure: The figure containing the plot.
        
        """
        
        tempMap = self.values
        if refLon is not None:
            rollCount = -(np.where(np.abs(self.lon-refLon) < self.dlon/2.+1e-6)[0][-1]-int(self.lon.size/2.))
            tempMap = np.roll(tempMap, rollCount, axis=1)

        plt.imshow(tempMap, cmap='inferno', extent=(-180,180,-90,90), origin='lower')
        
        plt.xlabel(r'$\rm Longitude$')
        plt.ylabel(r'$\rm Latitude$')
        plt.xticks([-180,-90,0,90,180])
        plt.yticks([-90,-45,0,45,90])
        
        cbar = plt.colorbar(orientation='vertical', fraction=0.05, pad = 0.05, aspect=9)
        cbar.set_label(r'$\rm Temperature~(K)$')

        return plt.gcf()
    
    def plot_H2_dissociation(self, refLon=None):
        """A convenience routine to plot the H2 dissociation map.
        
        Args:
            refLon (float, optional): The centre longitude used to rotate the map.
        
        Returns:
            figure: The figure containing the plot.
        
        """
        
        dissMap = self.dissValues*100.
        if refLon is not None:
            rollCount = -(np.where(np.abs(self.lon-refLon) < self.dlon/2.+1e-6)[0][-1]-int(self.lon.size/2.))
            dissMap = np.roll(dissMap, rollCount, axis=1)

        plt.imshow(dissMap, cmap='inferno', extent=(-180,180,-90,90), vmin=0, origin='lower')
        
        plt.xlabel(r'$\rm Longitude$')
        plt.ylabel(r'$\rm Latitude$')
        plt.xticks([-180,-90,0,90,180])
        plt.yticks([-90,-45,0,45,90])
        
        cbar = plt.colorbar(orientation='vertical', fraction=0.05, pad = 0.05, aspect=9)
        cbar.set_label(r'$\rm Dissociation~Fraction~(\%)$')
        
        return plt.gcf()

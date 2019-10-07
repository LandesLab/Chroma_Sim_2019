# General imports for math
import numpy.random as rnd
import numpy as np
import pandas as pd

# This segment is here to deal with the segfault happening on command line using column_runner.py requires that this code block be active.
#import matplotlib as mpl
#mpl.use('TkAgg') 
####

import matplotlib.pyplot as plt
from numba import jit

# Imports to make animation work
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.path as path
from IPython.display import HTML

# Imports the necessary data to make the graphs look nice.
import matplotlib.mlab as mlb
from matplotlib import rc

# Imports all the Scipy methods to do math
from scipy import optimize as opt
from scipy.interpolate import interp1d, splrep, sproot, splev
from scipy import stats as stats
from sklearn.preprocessing import normalize

# General Diagnostics and other handy items
import math
import time
import os
import timeit
from timeit import default_timer as timer

# Imports to allow for saving
import sys
import resource
import json 

# Required limit raise to allow the full column to be saved via Pickle
max_rec = 0x100000
sys.setrecursionlimit(max_rec)

################################ Helper Methods #########################################
def makeCoords(n, x_low = 0.0, x_high = 1.0, y_low = 0.0, y_high = 1.0, x_step = 0.0, y_step = 0.0, x_set = [], y_set = [], space = '', zero_val = 0.0, norm = True, distrib = rnd.uniform, d_options={}):
    """
    The purpose of this function is to create n tuples representing random coordinates of the adsorption probability histogram
    n = number of output tuples
    *_low, *_high = upper and lower bounds for each coordinate
    *_step = minimum distance between points. Default is 0.0
    *_set = predetermined set sizes should the user wish to define their own coordinates.
    norm = Boolean dictating if the final PDF is normalized such that the sum = 1
    """
    coords = np.zeros((n+2,2))
    if space == 'even':
        coords = np.zeros((n+1,2))
        coords[:, 0] = np.linspace(x_low, x_high, n+1, endpoint = True)
        coords[1:-1, 1] = distrib(low = y_low, high = y_high, size = n-1)
    
    else:
        if len(x_set) is 0:
            if x_step == 0.0:
                coords[0,0] = 0.0
                coords[-1,0] = 1.0
                coords[1:-1,0] = np.sort(distrib(low = x_low, high = x_high, size = n))
            else:
                return 0
        else:
            coords[0, 0] = 0.0
            coords[-1, 0] = 1.0
            coords[1:-1,0] = x_set
            
        if len(y_set) is 0:
            if y_step == 0.0:
                coords[1:-1,1] = distrib(low = y_low, high = y_high, size = n)
            else:
                return 0
        else:
            coords[1:-1,1] = y_set

    # Normalize y coords
    if norm:
        totals = np.sum(coords, axis = 0)
        coords[:, 1] /= totals[1]
    
    # Sort by x coords
    return coords

def update_hist(num, cc, pop = 0):
    """
    The update hist method is used for animating the output of an elution histogram.
    """
    for i in range(10):
        done = cc.timeStep()
    plt.cla()
    data = cc.currentProfile()[pop]
    uniques = len(np.unique(data))
    if uniques is 0:
        uniques = 1
    plt.hist(data, bins = uniques)

class MultiplePeaks(Exception): pass
class NoPeaksFound(Exception): pass

def ExpAsym(x, y, per = 0.10, k=10):
    """
    Determine full-with-half-maximum of a peaked set of points, x and y.

    Assumes that there is only one peak present in the datasset.  The function
    uses a spline interpolation of order k.
    """

    decriment = np.amax(y)*per
    center = np.argmax(y)
    s = splrep(x, y - decriment, k=k)
    roots = sproot(s)

    if len(roots) > 2:
        raise MultiplePeaks("The dataset appears to have multiple peaks, and "
                "thus the FWHM can't be determined.")
    elif len(roots) < 2:
        raise NoPeaksFound("No proper peaks were found in the data set; likely "
                "the dataset is flat (e.g. all zeros).")
    else:
        return abs((roots[0] - center)/(roots[1] - center)), center, roots[0], roots[1]
###########################################################################################
################################ Stationary Phase #########################################
###########################################################################################

"""
The stationaryPhase class which is home to all axial slices.
"""
class stationaryPhase:
    """
    This is the class definition for the statonary phase. It essentially binds together many axial slices.
    """
    chroma_column = None
    def __init__(self, name = 'stat_phase', res = (10,10), buildIt = True, contPDFs = False,
                 c_column = None, hop_dist_array = None):
        self.slice_ptr = 0                          # Pointer to the end of the column
        self.slices = {}                            # Dictionary to hold all axial slices
        self.name = name                            # Name of the column for printing
        self.res = res                              # A tuple of the form (num slices, points per slice)
        self.col_max = 0                            # Where the highest adsorption probability is
        self.col_max_pos = 0                        # What slice contains the highest probability
        self.elution_paths = np.zeros((1,res[0]))   # Initial holder for every elution path that has been made.
        self.num_elu_paths = 0                      # Counter enumerating how many elution paths there are
        self.hop_paths = np.zeros((1,res[0]))       # Initial holder for every hopping path that has been made
        self.num_hop_paths = 0                      # Counter enumerating how many hopping paths there
        self.site_types = dict()                    # A dictionary listing all the types of site.
        self.chroma_column = c_column               # Pointer to the master column
        self.num_sites = 0                          # Counter for the number of sites in the column
        self.site_norm = 0                          # Normalizer for site prevalence
        self.hop_dist_array = hop_dist_array        # Collection of pdfs to determine correlated adsorption
        self.site_names = dict()                    # Collection of site key to name relationships

        # Script to auto-construct a full column; Allows for fast implementation
        if buildIt:                                 
            for i in np.arange(res[0]):
                self.addSlice(nomen = str(i), contPDF = contPDFs)
        
    def __str__(self):
        """
        General printing method to test if the stationary phase exists
        """
        return self.name + ' : Resolution: '+ str(self.res) + ' : Elution Paths : '+ str(self.num_elu_paths)
    
    def getSlice(self, i):
        """
        Returns the axial slice at position i
        """
        return self.slices[i]
    
    def addSlice(self, nomen = '0', contPDF = False):
        """
        Adds an axial slice to the end of the column depending on slice_ptr
        Primarily a helper method for the column, not necessarily meant to be available for the outside observer
        """
        h_d = None
        # Assign hopping distribution based on input data.
        if self.hop_dist_array is not None:
            rows = self.hop_dist_array.shape[0]
            # If there is only one row, choose that one
            if rows is 1 or len(self.hop_dist_array.shape) == 1:
                try:
                    h_d = self.hop_dist_array[0,:]
                except:
                    h_d = self.hop_dist_array[:]
            # If there is many rows, choose one randomly
            # TODO Make these values have weights someday.
            else:
                h_d = self.hop_dist_array[rnd.choice(rows, 1)[0]]
        newSlice = axialSlice(name = nomen, stat_phase = self, pts = self.res[1], hop_distrib = h_d)
        
        # Switch to make the PDF continuous
        if contPDF:                             
            newSlice.makePdfContinuous()
        self.slices[self.slice_ptr] = newSlice
        self.slice_ptr += 1
    
    def getMostLikelyPath(self):
        """
        Returns the highest likelihood from each slice of the column
        """
        m_path = np.zeros(self.slice_ptr)
        for i in np.arange(self.slice_ptr):
            m_path[i] = self.getSlice(i).getMaxProb()[0]
        return m_path
    
    def getSiteTypes(self):
        """
        Returns the dictionary of all site types
        """
        return self.site_types
    
    def getColLength(self):
        """
        Returns the length of the column
        """
        return self.res[0]

    def modSlice(self, pos, new_slice):
        """
        User method to replace the slice at pos with a new slice _created by the user_
        """
        self.slices[pos] = new_slice
        
    def elute(self):
        """
        Generates a prospective path down the length of the column
        """
        t_path = np.zeros(self.slice_ptr)
        for i in np.arange(self.slice_ptr):
            t_path[i] = self.slices[i].sample()
        self.elution_paths = np.array([self.elution_paths, t_path])
        self.num_elu_paths += 1
        return t_path
    
    def getHops(self):
        """
        Generate a history of hops for a molecule
        """

        # Check to see if any distributions were given
        if self.hop_dist_array is None:
            return 1
        h_path = np.zeros(self.slice_ptr)
        for i in np.arange(self.slice_ptr):
            h_path[i] = self.slices[i].sampleHopDistribution()
        self.hop_paths = np.array([self.hop_paths, h_path])
        self.num_hop_paths += 1
        return h_path

    def multiElute(self, n):
        """
        Repeatedly call elute to make a larger number of paths
        """
        t_paths = np.zeros((n, self.slice_ptr))
        for i in np.arange(n):
            t_paths[i,:] = self.elute()
        return t_paths
    
    def newSiteType(self, site_type, distrib, d_options, prevalence, comp):
        """
        Creation method to generate a new type of site
        """
        # Auto-update and normalize all prevalences
        self.site_norm += prevalence
        # Creates a new site type and sets it in the stationary phase. 
        self.site_types[self.num_sites] = site(site_type = site_type, distrib = distrib,
                                               d_options = d_options, prevalence = prevalence,
                                               comp = comp, stat_phase = self)
        # Increment total number of site types
        self.site_names[site_type] = self.num_sites
        self.num_sites += 1
        
    def giveSiteType(self):
        """
        Sample method that lets an outside system call for a new site given the sites prevalence
        in the column. This call wraps around oddly, 
        """
        key_iterator = 1
        name_keys = list(self.site_names.keys())          # Get the full list of keys
        this_site = self.site_names[name_keys[0]]
        if len(name_keys) == 0:                     # Check to be sure that sites do exist
            return None
        chance = rnd.uniform()                      # roll dice to pick site type
        limit = self.site_types[this_site].getPrevalence()/self.site_norm
        while chance >= limit:
            this_site = self.site_names[name_keys[key_iterator]]
            limit += self.site_types[this_site].getPrevalence()/self.site_norm
            key_iterator += 1
        return self.site_types[this_site]
        
    def getSite(self, ax_pos):
        """
        Returns a site from the axial slice located at ax_pos
        """
        return self.slices[ax_pos].getSite()
    
    def modifySite(self, site_key, prev = None, distrib = None, d_options = None, comp = None):
        """
        Function that pushes a request to change the variables within an entire site type. This does not
        do site specific changes in an axial slice but rather changes the variables in an entire site type
        """
        # Check if a name is given
        if type(site_key) is not int:
            num_site_key = self.site_names[site_key]

        if prev is not None:
            self.site_norm += prev - self.site_types[num_site_key].getPrevalence()
            if prev == 0:
                del self.site_types[num_site_key]
                del self.site_names[site_key]
            else:
                self.site_types[num_site_key].setPrevalence(prev)
            
        if distrib is not None:
            self.site_types[site_key].setDistribution(distrib)
        if d_options is not None:
            self.site_types[site_key].setDOptions(d_options)
        if comp is not None:
            self.site_types[site_key].setComp(comp)

    def columnMatrix(self):
        """
        The idea behind this function is to represent the whole of the stationary phase in a clever
        way that would allow different stationary phases to be compared easily.
        """
        col_mat = np.zeros((self.slice_ptr, ))
        
    def eluteNshow(self, save = False):
        """
        Randomly generate an elution and graph it. Primarily a diagnostic method.
        """
        pathFig = plt.figure()
        plt.plot(self.elute())
        plt.title(self.name+' - Sample Elution')
        plt.xlabel('Slice \#')
        plt.ylabel(r'P(f(x))')
        if save:
            pathFig.savefig(directory+'_'+self.name+'_path_'+str(pathNum)+'.jpg', bbox_inches = 'tight')
        else:
            plt.show()
                            
    def plotMLP(self, save = False):
        """
        Quickly grab and graph the most likely path through the column.
        """
        MLPfig = plt.figure()
        plt.plot(self.getMostLikelyPath())
        plt.title(self.name+'- Most Likely Path')
        plt.xlabel('Slice \#')
        plt.ylabel(r'P(f(x))')
        if save:
            MLPfig.savefig(directory+'_'+self.name+"_MLP.jpg", bbox_inches = 'tight')
        else:
            plt.show()

###########################################################################################
################################ Axial Slice ##############################################
###########################################################################################
"""
The axialSlice class which holds a pdf and several access/statistics methods.
"""
class axialSlice:
    """
    Class definition for the axial slice of a column
    """
    stat_phase = None
    def __init__(self, name = '0', disc = True, pts = 10, space = 'even', num_sites = 0, 
                 site_limit = float('inf'), stat_phase = None, hop_distrib = None):
        """
        Method of self creation
        """
        self.num_feat = 0                   # Holder variable for number of features in the column
        self.name = name                    # Name variable to help distinguish between slices
        self.disc = disc                    # Variable dictating if PDFs are discrete or continuous
        self.space = space                  # Dictates the spacing between points on the PDF; Are they even or randomly placed?
        self.pdf = self.makeSlcPdf(pts)     # Make an initial pdf for the slice and store it.
        self.num_sites = 0                  # A listing of the number of sites available
        self.site_limit = site_limit        # The max number of sites available in this slice.
        self.stat_phase = stat_phase        # Link the the master stationary phase in the column.
        self.hop_distribution = hop_distrib # 1D numpy array of the distribution of hops
        if not disc:
            self.makePdfContinuous()
    
    def __str__(self):
        """
        Printable string method to state all the necessary variables in an axialSlice
        """
        return "Axial Name - "+ str(self.name) + " with probability avg of " + str(self.pdfAvg()) + "number of sites " + str(self.num_sites)
    
    def makeSlcPdf(self, pts = 10):
        """
        Returns a basic PDF, usually reserved for diagnostic purposes.
        """
        return makeCoords(pts, space = self.space)
    
    def setPdf(self, pdf_in):
        """
        This allows a user to build a PDF and directly assign it to an axialSlice
        """
        self.pdf = pdf_in
    
    def setHopDistribution(self, distrib_in):
        """
        This method let's a user set the hop distribution for this slice
        """
        self.hop_distribution = distrib_in

    def getPdf(self):
        """
        Easy method to grab and examine a pdf for a specific slice.
        """
        return self.pdf
    
    def getSite(self):
        """
        Returns one of the possible sites held by the stationary phase.
        """
        this_site = self.stat_phase.giveSiteType()

        max_reached = self.num_sites >= self.site_limit
        if not max_reached:
            self.num_sites += 1
            return this_site
        elif max_reached:
            # TODO: Need to implement this whenever we decide we want to look at competitive adsorption
            return None
    
    def getHopDistribution(self):
        """
        Retrieves hop distribution
        """
        return self.hop_distribution

    def sampleHopDistribution(self):
        """
        Sample the hop distribution to get the number of times a molecule jumps on the surface
        """

        # Normalize the array real quick
        self.hop_distribution /= np.sum(self.hop_distribution)
        if self.hop_distribution is not None:
            # Add one to the number drawn to ensure that the molecule at least attempts to adsorb
            s = rnd.choice(self.hop_distribution.shape[0], 1, p = self.hop_distribution) + 1
            return s[0]
        else:
            return 1

    def nSampleHopDistribution(self,n):
        """
        Take many samples from the hop distribution to get the number of times a molecule jumps on the surface
        """
        # Need to add one else the first option is zero and the molecule never stops.
        s = rnd.choice(self.hop_distribution.shape[0], n, p = self.hop_distribution) + 1
        return s

    def makePdfContinuous(self, mode = 'cubic', spacing = 600, normalize = True):
        """
        General method to make the PDF in a slice continuous rather than discrete.
        """
        xnew = np.linspace(0, 1, num=spacing, endpoint=True)
        p = self.pdf
        f2 = interp1d(p[:,0], p[:,1], kind = mode, fill_value = 0)
        c_pdf = f2(xnew)
        c_pdf[c_pdf<0] = 0
        if normalize:
            norm = np.sum(c_pdf)
            c_pdf /= norm
        # Set pdf and inform slice of change
        self.pdf = np.transpose([xnew, c_pdf])
        self.disc = False
        
    def pdfAvg(self):
        """
        This method is absolute shit in implementation but it is straightforward.
        """
        return np.mean(self.nSample(500))
    
    def getMaxProb(self):
        """
        Returns a tuple containing the maximum probability of adsorption for this slices pdf.
        """
        my_max = np.max(self.pdf[:,1])
        return (self.pdf[self.pdf[:,1] == my_max, 0][0], my_max)
    
    def sayMyName(self):
        # Return what the slice's name is.
        return self.name
    
    def sample(self):
        # Return a single sample selected from the slice.
        s = rnd.choice(self.pdf[:,0], 1, p = self.pdf[:,1])
        return s[0]
    
    def nSample(self, n):
        # Return n samples from this slice.
        s = rnd.choice(self.pdf[:,0], n, p = self.pdf[:,1])
        return s
    
    def visSlice(self):
        # Graphing script to view the axial slice.
        if self.disc:
            plt.plot(self.pdf[:,0], self.pdf[:,1], 'X', markersize=15 )
        else:
            plt.plot(self.pdf[:,0], self.pdf[:,1])
        plt.xlabel('f(x)')
        plt.ylabel('P(f(x))')
        plt.title('Axial Slice - ' + self.name)
        plt.xlim([0,1])

###########################################################################################
################################# Sites ###################################################
###########################################################################################
"""
The site class which contains all necessary statistics for an adsorption site.
"""
class site:
    # Class defining an adsorption site.
    stat_phase = None
    def __init__(self, stat_phase = stat_phase, site_type = 'normal', comp = False, 
                 shift_time = False, distrib = rnd.normal, d_options = {}, prevalence = 1):
        self.site_type = site_type # Type tag
        self.stat_phase = stat_phase
        self.comp = comp # Boolean stating if the site undergoes competitive binding
        self.limit_list = np.zeros(stat_phase.getColLength())  # List of the max number of each site type at a slice.
        self.adsorb_count = np.zeros(stat_phase.getColLength()) # List the current number of adsorptions at each location.
        self.prevalence = prevalence
        self.distribution = distrib
        self.d_options = d_options  
        
    def __str__(self):
        return self.site_type + ": Prevalence - " + str(self.prevalence) + " : Comp - "+ str(self.comp) + " : Distribuiton - " + str(self.distribution) + ' : D_Options - ' + str(self.d_options)
    
    def isOcc(self, ax_num):
        return self.limit_list[ax_num]
    
    def getTotalAdsCounts(self):
        return np.sum(self.adsorb_count)

    def getAdsCountList(self):
        return self.adsorb_count

    def setSpeed(self, new_speed):
        # Eventual development for exit speed of a molecule from the site.
        self.speed = new_speed

    def getPrevalence(self):
        return self.prevalence

    def setPrevalence(self, new_prev):
        self.prevalence = new_prev

    def setDistribution(self, new_distrib):
        self.distribution = new_distrib

    def setDOptions(self, new_d_opts):
        self.d_options = new_d_opts

    def setComp(self, new_comp):
        self.comp = new_comp

    def adsorb(self, ax_num): # Signify the adsorption of a molecule and indicate that
        # TODO: Introduce competition by incorporating site limits
        maxed = False 
        if not maxed:
            self.adsorb_count[ax_num] += 1
            s_time = self.getAdsTime()
            return s_time
        elif self.comp and maxed :
            # Do something for competition
            return None
        else: # Already occupied, do nothing
            return 0
    
    def desorp(self): # Clear the site for newer binding
        self.occ = False
        self.mol = None
        
    def getAdsTime(self):
        # Call Chroma column instead.
        return self.distribution(**self.d_options)
        
    def timeStep(self):
        if self.occ is True:
            self.tot_ads_time += 1

    def dumptoJSON(self):
        this_dict = dict()
        this_dict['site_type'] = self.site_type
        this_dict['comp'] = self.comp
        this_dict['prevalence'] = self.prevalence
        this_dict['d_opts'] = self.d_options 
        this_dict['distribution'] = self.distribution.__name__
        return this_dict

###########################################################################################
################################# Molecule ################################################
###########################################################################################

class molecule:
    """
    The molecule class which contains an adsorption pathway and all statistics pertinent
    to a molecule.
    """
    chroma_column = None # Holder for master chroma column
    def __init__(self, c_column = None, speed = 1, state = 'moving', name = '0', path = 0.5, mol_type = 0, hops = 1, competitive = False, takeHistory = False):
        """
        speed - the number of steps moved by the car each time cycle
        state - dictates whether the car is moving or parked
        name  - a given name for the car
        path  - the path the car follows dictating whether the car will park or not. If it
                is an int, then that is the number of steps in the column. If it is an
                numpy array, then that is the probability of the car parking.
        mol_type - a tag to differentiate one type of molecule from another. Allows for inclusion of contaminant 
        """
        self.speed = speed                  # Number of axial slices passed per time step
        self.state = state                  # Adsorbed or moving?
        self.mol_type = mol_type            # The molecular type
        self.name = name                    # A diagnostic name
        self.max_speed = speed              # A base limitation for how fast an analyte can go. Can be altered later.
        self.chroma_column = c_column       # Establish a connection to the column. Shared by all molecules
        self.path = path # The adsorption probability at each step in the column; May be constant
        self.competitive = competitive
        self.takeHistory = takeHistory
        ############################################# Everything in this section needs to be reset ###############################################
        if takeHistory:
            self.history = list() # List of the column position of a molecule at a set time.
        self.stop_time = 0 # Amount of time left for this particle to remain motionless
        self.distance = 0 # How far the molecule has already moved
        self.real_distance = 0 # Real distance moved to include stoppage time
        self.ad_count = 0 # Number of times the molecule has already adsorbed
        self.finished_eluting = False # Tag to state whether the molecule has fully eluted
        self.clock = 0 # Internal clock for time tracking
        self.overage = 0 # Sometimes a molecules does not have a discrete value for ads time. Here we take the overage.
        self.last_adsorption = 0 # Number of steps since the last adsorption undergone by the molecule.
        self.ads_limit_list = np.zeros(c_column.getColLength())
        ##########################################################################################################################################

        ########################################### Calculated Values ############################################################################

        self.hops = hops
        if type(hops) is int:
            self.curr_num_hops = self.hops
        elif type(hops) is np.ndarray:
            self.curr_num_hops = self.hops[0]


        # Calculate average ads probability
        if type(self.path) is int or type(self.path) is float: 
            self.path_avg = self.path
        else:
            this_sum = 0
            for i in range(self.chroma_column.getColLength()):
                this_sum += self.path[i]
            this_sum /= self.chroma_column.getColLength()
            self.path_avg = this_sum

    def __str__(self):
        """
        Default printing method for diagnostic testing
        """
        return self.name + ': '+ self.state + ': Distance Traveled = ' + str(self.distance)
    
    def setPath(self, new_path):
        """
        Setter method to create a new path if necessary.
        """
        self.path = new_path
    
    def getDistance(self):
        """
        Access method to grab the distance.
        """
        return self.distance

    def getRealDistance(self):
        """
        Access method to grab the molecule plus its adjusted distance due to stoppage time
        """
        return self.distance + self.real_distance

    def getTime(self):
        """
        Access method to grab the time.
        """
        return self.clock + self.overage - self.getAdCount()
    
    def getType(self):
        """
        Access method to grab the molecular type
        """
        return self.mol_type
    
    def getAdCount(self):
        """
        Access method to grab the number of adsorptions
        """
        return self.ad_count
    
    def getPathAverage(self):
        """
        Return the raw average adsorption across the path of this molecule.
        """
        return self.path_avg


    def getHopPath(self):
        """
        Return the hopping path for this molecule
        """
        return self.hops

    def getHistory(self):
        """
        Returns the history of a molecule in the column
        """
        if self.takeHistory:
            return self.history
        else:
            print("No histories were collected")
            
    def getAdsLimitList(self):
        return self.ads_limit_list

    def isAdsorbed(self):
        """
        Boolean method to state if the molecule is currently adsorbed
        """
        return self.state == 'adsorbed'
    
    def isEluted(self):
        """
        Boolean method to state if the molecule has passed enough distance to exit the column.
        """
        return self.finished_eluting
    
    def reset(self):
        """
        Reset method to return the molecule to the "top" of the column. Does not reset the path.
        """
        self.stop_time = 0              # Amount of time left for this particle to remain motionless
        self.distance = 0               # How far the molecule has already moved
        self.ad_count = 0               # Number of times the molecule has already adsorbed
        self.finished_eluting = False   # Tag to state whether the molecule has fully eluted
        self.clock = 0                  # Internal clock for time tracking
        self.overage = 0                # Overage Time
        self.last_adsorption = 0        # Number of steps since the last adsorption undergone by the molecule.
        if self.takeHistory:
            self.history = list()

        if type(self.hops) is int:
            self.curr_num_hops = self.hops
        else:
            self.curr_num_hops = self.hops[0] # Number of hops to take in the first step.
        
    def doIadsorb(self, site = None):
        """
        Boolean method that dictates if the molecule has adsorbed to a site. This can be given a site 
        (for diagnostic purposes) or can ask the column to return a site based on the amount of distance the molecule has traveled.
        The method returns a boolean indicating whether it eventually adsorbed.
        """
        # Check to see if you have finished eluting.
        if self.finished_eluting:
            return False
        
        # Update my hop count
        self.curr_num_hops -= 1

        # Roll a dice
        chance = rnd.uniform()

        # Probability as given by the stat_phase pdf.
        # Add if statement here to check if path is just one value; If so, just take that value.
        if type(self.path) is int or type(self.path) is float:
            path_contrib = self.path
        else:
            path_contrib = self.path[self.distance-1] # To be expanded on later

        limit = path_contrib
        # Adjustment for correlated adsorption events.
        corr_contrib = 0
        if self.chroma_column.corr_ads is not None:
            diff_dist = self.distance - self.last_adsorption
            # If we _just_ desorbed
            if diff_dist is 0:
                corr_contrib = 0
            elif diff_dist > 0:
                corr_contrib = self.chroma_column.corr_ads(diff_dist)
                #print corr_contrib
        """
        End of P(ads) section
        """

        limit = (path_contrib + corr_contrib)
        self.ads_limit_list[self.getDistance()] = limit

        # Test if the change falls below the limit
        if chance <= limit:
            if site is None: # Were we handed a site?
                site = self.chroma_column.getSite(dist = self.distance)
                if site is not None: # Was a site available? Only triggers under limited sites
                    s_time = site.adsorb(self.getDistance())
                    self.adsorb(s_time)
        return chance <= limit
    
    def accelerate(self, new_spd = None, spd_dif = None):
        """
        Modifier method that is used to introduce velocity changes. Will be used at a later date.
        """
        if new_spd is None and spd_dif is not None:
            self.speed += spd_dif
        else:
            self.speed = new_spd
        
    def adsorb(self, s_time):
        """
        adsorb is called assuming that doIadsorb returned true. It flags that he molecule has adsorbed,
        brings in a stop time (s_time) provided by the site and increments the adsorption count.
        """
        if self.competitive:
            self.state = 'adsorbed'
            self.stop_time = s_time
            self.ad_count += 1 # Might want to build in a diagnostic for capturing the number of events for each different site.
        else:
            self.overage += s_time + 1 # There's a lot that goes into why this plus one is here
            self.ad_count += 1

        """
        Why the plus one:
        In a competitive system, the molecule is treated as not moving for the full tick as long as _some_ stoppage time remained. This, in turn, causes it to lag behind one extra tick from the rest of the population as the other molecules still arrive 'first' when looking at adsorption sites.
        """
    def timeStep(self):
        """
        Central method for moving the molecule down the column. It analyzes motion based on the molecular tag,
        alters stop time as necessary and evaluates if the molecule has spent long enough to exit the column.
        """
        if not self.finished_eluting:
            self.clock += 1
            if self.state == 'adsorbed':
                self.stop_time -= 1
                if self.stop_time <= 0:
                    self.overage -= self.stop_time # Stop time is negative so need to subtract to make it positive
                    self.real_distance -= self.stop_time
                    self.stop_time = 0
                    self.state = 'moving'
                    self.last_adsorption = 0
            else:
                if self.curr_num_hops <= 0.0:
                    self.distance += self.speed
                    if type(self.hops) is int:
                        self.curr_num_hops = self.hops
                    else:
                        self.curr_num_hops = self.hops[self.distance-1]
                if self.distance >= self.chroma_column.getColLength():
                    self.finished_eluting = True

                else:
                    self.doIadsorb()
        return self.finished_eluting

###########################################################################################
################################# Chroma Column ###########################################
###########################################################################################

class ChromaColumn:
    """
    The complete shell class for the chromatographic column. This maintains access to all pertinent structures and has hold of several methods to grab statistics of interest.
    """
    def __init__(self, name = None, 
                 stat_dict = {'name' : 'stat_phase', 'res' : (10,10), 'buildIt' : True, 'contPDFs' : False, 'hop_dist_array': None}, 
                 molecule_limit = 1000, make_paths = False, cons_ads = None, directory = '', competitive = False):
        """
        Inputs:
        name - column specific name which will be used to save the column when pickled.
        
        stat_dict - dictionary of all necessary items to generate the stationary phase.
        
        molecule_limit - The (soft)max number of molecules to be generated upon initialization. More can be added
                         after initialization.
        
        make_paths - boolean dictating whether elution pathways are generated upon molecule creation.
        
        cons_ads - if make_paths is False, cons_ads sets each molecule to have a constant adsorption (0 < x < 1)
        
        competitive - boolean variable dictating whether adsorption at sites is modeled competitively

        Class variables:
        stat_phase - a pointer to the stationary phase associated with the column. Initialized upon creation but can be replaced via the setStatPhase() method
        molecule_limit - The (soft) max number of molecules to be generated upon initialization. More can be added
                         after initialization.
        molecules - dictionary containing all molecules in the column. Can be used to pull specific molecules and
                    gather pertinent statistics.
        name - column specific name which will be used to save the column when pickled.
        
        num_molecules - total number of molecules in the column at that instant.
        
        all_eluted - boolean indicating if all molecules in the column have reached the end of the column.
        
        clock - hand of the internal clock tracking the motion of the molecules.

        col_length - the full length of the column in time steps
        
        """
        
        self.stat_phase = stationaryPhase(**stat_dict)      # Associated stationary phase with all options imported in.
        self.stat_dict = stat_dict                          # Retain stat_dict for saving
        self.molecule_limit = molecule_limit                # Max number of molecules to make during the initial run.
        self.molecules = dict()                             # Master dictionary containing all molecules in the column
        self.name = name                                    # Name of the column (for printing/saving)
        self.num_molecules = 0                              # Current number of molecules in the column.
        self.all_eluted = False                             # Boolean evaluating if all molecule have eluted.
        self.clock = 0                                      # Clock hand that will track time.
        self.col_length = stat_dict['res'][0]               # Store the native column length
        self.run_num = 0                                    # Calculate the number of elutions that have been performed.
        self.base_dir = directory                           # Base directory where all the data will be saved
        self.competitive = competitive                      # Boolean determining if adsorption events occur competitively
        self.start_time = timeit.default_timer()            # Timer for evaluating performance
        self.conditionals_active = False                    # Boolean to indicate if freezeFrame conditionals have been set.
        self.corr_ads = None
        # Make a storage directory for all pertinent data
        self.directory = directory+self.name+'/'
        # Indicate a name conflict and stop
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        # Make all the molecules in the column.
        for i in range(self.molecule_limit):
            self.makeMolecule(make_path = make_paths, path = cons_ads, comp = competitive)
            if i%round(molecule_limit/10,0) == 0:
                print(str(i) + ' molecules made! Total time elapsed: ' + str(round(timeit.default_timer() - self.start_time, 3)) + 's')

        # Make folder for first run of column. Check to see if there is conflicting information.
        self.run_directory = self.directory+str(self.run_num)+'/'
        while(os.path.exists(self.run_directory)):
            self.run_num += 1
            self.run_directory = self.directory+str(self.run_num)+'/'

        # Should be true but should check to make sure.
        if not os.path.exists(self.run_directory):
            os.makedirs(self.run_directory)
        
            
    def __str__(self):
        """
        Printing method to all for easy diagnostic of column creation.
        """
        return self.name + ' with ' + str(self.num_molecules) + ' molecules.'
            
    def setStatPhase(self, new_stat_phase):
        """
        A setter method to assign a new stationary phase to the column.
        """
        self.stat_phase = new_stat_phase
    
    def getStatPhase(self):
        """
        Retrieve the stationary phase associated with this column.
        """
        return self.stat_phase
    
    def getColLength(self):
        """
        Retrieve the full length of the column
        """
        return self.col_length
    
    def getSite(self, dist):
        """
        TODO: This method is not currently complete and needs refinement of purpose.
        Retrieves a site based on a call from molecule.
        """
        return self.stat_phase.getSite(dist)
    
    def getMolecule(self, num = 0):
        """
        Retieve the molecule at num from the full dictionary in the column.
        """
        return self.molecules[num]
    
    def getClockTime(self):
        """
        Retrieve the current clock time.
        """
        return self.clock
    
    def getAxialSlice(self, slice_num):
        """
        Return the axialSlice located at slice_num
        """
        return self.stat_phase.getSlice(slice_num)
    
    def sayMyName(self):
        return self.name

    def setMyName(self, new_name):
        self.name = new_name
        # Make a storage directory for all pertinent data
        self.directory = self.base_dir+self.name+'/'

        # Indicate a name conflict and stop
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.run_directory = self.directory+str(self.run_num)+'/'
        self.run_num = 0
        while(os.path.exists(self.run_directory)):
            self.run_num += 1
            self.run_directory = self.directory+str(self.run_num)+'/'

        # Should be true but should check to make sure.
        if not os.path.exists(self.run_directory):
            os.makedirs(self.run_directory)

    def sayCurrentDirectory(self):
        """
        Returns the directory stub given to the column
        """
        return self.directory

    def sayCurrentRunDirectory(self):
        """
        Returns the directory with the run number appended
        """
        return self.run_directory

    def newSiteType(self, name = '0', distrib = rnd.exponential, d_options = {'scale': 1}, prevalence = 1, comp = False):
        """
        Column level method to generate a new site type. The distribution parameters are
        handed directly to the stationary phase via distrib and d_options.
        """
        self.stat_phase.newSiteType(name, distrib, d_options, prevalence, comp)
    
    def listActiveSites(self):
        """
        TODO: Method is currently incomplete.
        Quickly produce a list of all types of sites currently in the column.
        """
        this_list = self.stat_phase.getSiteTypes()
        for i in this_list:
            print(this_list[i])

    def modifySite(self, site_key, prev = None, distrib = None, d_options = None, comp = None):
        self.stat_phase.modifySite(site_key, prev = prev, distrib = distrib, d_options = d_options, comp = comp)
    
    def makeMolecule(self, make_path = False, path = None, hops = None, comp = False):
        """
        ChromaColumn level molecule creation method. Molecules can have a path generated by the associated 
        stationary phase or be handed a path from an outside entity.
        """
        # Check to see if there is a path that has been handed over
        if make_path:
            this_path = self.stat_phase.elute()
        else:
            this_path = path

        # Figure out the hopping behavior
        these_hops = self.stat_phase.getHops()
        
        self.molecules[self.num_molecules] = molecule(name = str(self.num_molecules), c_column = self, path = this_path, hops = these_hops, competitive = comp)
        self.num_molecules += 1
    
    def currentProfile(self):
        """
        Returns the position and state of all molecules in the column in three numpy arrays.
        ful_prof - position of all molecules being eluted
        mob_prof - position of all molecules currently in motion
        ads_prof - position of all molecules currently adsorbed to the stationary phase
        
        These will need to be histogrammed after output.
        TODO: Eventually expand current profile to handle velocity differences
        """
        ful_prof = np.zeros((self.num_molecules))
        mob_prof = np.zeros((self.num_molecules))
        ads_prof = np.zeros((self.num_molecules))
        j, k = 0, 0

        # Set the way we are measuring distance of molecules in the column
        if self.competitive: # Measure distance by the slice ptr
            mol_pos = lambda i, cc: cc.getMolecule(i).getRealDistance()
        else: # If we aren't competitive, measure molecule position by time in column (*assumes constant velocity)
            mol_pos = lambda i, cc: cc.getMolecule(i).getTime()

        for i in range(len(self.molecules)):
            if not self.molecules[i].isEluted():
                if self.molecules[i].isAdsorbed():
                    ads_prof[j] = mol_pos(i, self)
                    j += 1
                else:
                    mob_prof[k] = mol_pos(i, self)
                    k += 1
                
                if self.competitive:
                    ful_prof[i] = self.molecules[i].getDistance()
                else:
                    ful_prof[i] = self.molecules[i].getTime()

        ful_prof = ful_prof[ful_prof != 0]
        return ful_prof, np.trim_zeros(mob_prof), np.trim_zeros(ads_prof)
            
    def timeStep(self):
        """
        Performs a time step to advance the clock for both the column and all molecules in the column.
        all_eluted is updated during this step and will be set to True if all molecules have reached the
        end of the column.
        """
        if not self.all_eluted:
            self.all_eluted = True
            for i in range(len(self.molecules)):
                this_mol = self.molecules[i].timeStep()
                self.all_eluted = self.all_eluted and this_mol
            self.clock += 1
            if self.conditionals_active:
                self.evaluateMolecules()
        return self.all_eluted
        
    def performElution(self):
        """
        Perform the timeStep method until all molecules are eluted.
        TODO: Expand method to optionally record the currentProfile for every n steps.
        """
        while self.all_eluted is False:
            self.timeStep()
            if self.clock % 1000 == 0:
                    print(str(self.clock) + ' ticks have passed! Total time elapsed: ' + str(round(timeit.default_timer() - self.start_time, 3)) + 's')
        return self.clock
    
    def finalProfile(self):
        """
        Produces the final elution profile for the column after every molecule has eluted. Is essentially the
        final product of an experimental elution.
        """
        if self.all_eluted:
            time_set = np.zeros(self.num_molecules)
            for i in range(len(self.molecules)):
                    time_set[i] = self.molecules[i].getTime()
            return time_set
        else:
            return None

    def plotFinalProfile(self, bins = 40, save = False, directory = ''):
        """
        Calculate the final elution profile after all molecules have exited the column.
        Can additionally print/save a figure for the final chromatogram.
        """
        if self.all_eluted:
            fig = plt.figure()
            plt.hist(self.finalProfile(), bins = bins)
            if save:
                plt.savefig(self.sayCurrentRunDirectory()+'_finalProfile')
                plt.close()
        else:
            print('Not all molecules have eluted!')
            return None

    def storeFinalProfileData(self):
        """
        Capture all the pertinent statistics for this chromatographic separation.
        """
        vals = self.finalProfile()
        if vals is None:
            print('Elution was not finished, skipping profile storage.')
            return None
        results = pd.DataFrame({'Final Profile' : self.finalProfile()})
        results.to_csv(self.sayCurrentRunDirectory()+"_ElutionData")

    def currentStats(self, pop = 0):
        """
        Call intermediate stats as necessary
        """
        if self.all_eluted:
            data = self.finalProfile()
        else:
            data = self.currentProfile()[pop]
        return stats.describe(data)
    
    def writeCurrentStats(self, pop = 0, directory = None, save = False):
        results = pd.DataFrame({'Stats': ['pop #', 'min/max', 'mean', 'variance', 'skewness', 'kurtosis'],
                        self.sayMyName() : self.currentStats(pop)})
        results = results.set_index('Stats')
        if save:
            results.to_csv(self.sayCurrentRunDirectory()+"_stats")

    def reset(self):
        """
        Resets all molecules in the column back to an un-eluted state. Useful for repeating the experiment
        without needing to recreate all molecules.

        TODO: Reset the site data
        """
        for i in range(len(self.molecules)):
            self.molecules[i].reset()
        self.run_num += 1
        self.all_eluted = False
        self.clock = 0
        if self.conditionals_active:
            for i in self.conditionals_list:
                i.completion_bool = False
        self.start_time = timeit.default_timer()
        self.run_directory = self.directory+str(self.run_num)+'/'
        if not os.path.exists(self.run_directory):
            os.makedirs(self.run_directory)
    
    def saveColumn(self, directory = None):
        """
        DEFUNCT - SWITCHED TO JSON METHODOLOGY
        This method saves a column to a .pkl file given a target directory. The file can be loaded using the following
        commands. Note that the max recursion limit needs to be raised for this to work successfully.
        output = open(directory + 'pickled_column.pkl', 'rb')
        c2 = pickle.load(output)
        """
        """
        directory = self.directory
        output = open(directory + self.name+'.pkl', 'wb')
        # Pickle dictionary using max protocol.
        pickle.dump(self, output, -1)
        output.close()
        """
        pass
        
    def axialSliceMatrix(self, start = 0, x = 2, y = 2, save = False, name = 'temp'):
        """
        Generates a matrix of the pdfs of a series of consecutive axial slices. Probability is retained on the left and 
        the x-axis is maintained on the bottom row. 
        x - total number of rows of the matrix
        y - total number of columns of the matrix
        save - Boolean to turn saving on/off
        name - name that will be used to save the figure. Assumes directory information is passed
        """
        r_i = x
        r_j = y
        f, plots = plt.subplots(r_i,r_j)
        num = start
        for i in range(r_i):
            for j in range(r_j):
                this_slice = self.stat_phase.getSlice(num).getPdf()
                plots[i,j].plot(this_slice[:,0], this_slice[:,1])
                plt.setp(plots[i,j].get_yticklabels(), visible=False)
                plt.setp(plots[i,j].get_yticklines(), visible=False)
                if i is not 2:
                    plt.setp(plots[i,j].get_xticklabels(), visible=False)
                num += 1
        if save:
            plt.savefig(name, '.jpg')
            
    def getMolAdsorptionProfile(self):
        """
        Grab the adsorption count for each molecule in the column
        """
        ads = np.zeros(self.num_molecules)
        for i in range(len(self.molecules)):
            ads[i] = self.molecules[i].getAdCount()
        return ads

    def plotMolAdsorptionProfile(self, directory = '', save = False):
        fig = plt.figure()
        plt.hist(self.getMolAdsorptionProfile(), bins = np.unique(self.getMolAdsorptionProfile()))
        if save:
            plt.savefig(self.sayCurrentRunDirectory()+'_molAdsorptionProfile.png')
            plt.close()
        else:
            plt.show()

    def storeMolAdsorptionProfileData(self):
        results = pd.DataFrame({'Number of Adsorptions' : self.getMolAdsorptionProfile()})
        results.to_csv(self.sayCurrentRunDirectory()+"_adsData")

    def getMolAvgProbProfile(self):
        holder = np.zeros(self.num_molecules)
        for i in range(self.num_molecules):
            holder[i] = self.getMolecule(i).getPathAverage()
        return holder

    def plotMolAvgProbProfile(self, directory = '', save = False, bins = 100):
        fig = plt.figure()
        plt.hist(self.getMolAvgProbProfile(), bins = self.getMolAvgProbProfile())
        if save:
            plt.savefig(self.sayCurrentRunDirectory()+'_molAvgProbProfile.png')
            plt.close()
        else:
            plt.show()

    def storeMolAvgProbProfileData(self):
        results = pd.DataFrame({'Avg. Ads. Prob' : self.getMolAdsorptionProfile()})
        results.to_csv(self.sayCurrentRunDirectory()+"_avgProbData")

    def dumpParameterstoJSON(self):
        # This method needs to dump the parameters for the sites as well.
        this_dir = self.sayCurrentRunDirectory()

        check_hops = type(self.stat_dict['hop_dist_array'])
        try:
            if  check_hops is not None and check_hops is not list:
                self.stat_dict['hop_dist_array'] = self.stat_dict['hop_dist_array'].tolist()
        except KeyError:
            pass
        this_dict = dict()
        this_dict['molecule_limit'] = self.molecule_limit
        this_dict['name'] = self.name
        this_dict['num_molecules'] = self.num_molecules
        this_dict['col_length'] = self.col_length
        this_dict['competitive'] = self.competitive
        this_dict['directory'] = self.directory

        with open(this_dir+'params.json', 'w') as fp1:
            json.dump(self.stat_dict, fp1)
            json.dump(this_dict, fp1)
            for i in self.stat_phase.site_types:
               site_dict = self.stat_phase.site_types[i].dumptoJSON()
               json.dump(site_dict, fp1)
            fp1.close()

    def gatherData(self):
        # Default data gathering method. Will break if handed a None value after the column resets.
        self.plotFinalProfile(save = True)
        self.storeFinalProfileData()
        self.plotMolAdsorptionProfile(save = True)
        self.storeMolAdsorptionProfileData()
        self.plotMolAvgProbProfile(save = True)
        self.storeMolAvgProbProfileData()
        self.writeCurrentStats(save = True)
        print("All data saved at "+ str(self.sayCurrentRunDirectory()))
    """
    def makeVideo(self, directory = './', number_of_frames = 200, pop = 0, save = False, writer = None):
        def update_hist(num, cc):
            for i in range(10):
                done = cc.timeStep()
                plt.cla()
                data = cc.currentProfile()[pop]
                uniques = len(np.unique(data))
                if uniques is 0:
                    uniques = 1
                plt.hist(data, bins = uniques)

        fig = plt.figure()
        plt.hist(self.currentProfile()[pop], normed = False)
        ani = animation.FuncAnimation(fig, update_hist, number_of_frames, fargs=(self, ))
        plt.show()
        if save:
            ani.save(directory+ self.myDir() + 'elutionOfPop_' + str(pop), writer=writer)
    """
# Freeze Frame methods
    """
    freezeFrame is a nested class that only exists inside of ChromaColumn. The purpose of freezeFrame is to contain the specifics for when a freeze frame should occur and to appropriatly store the state of the column when that condition is met. As of now, freezeFrame only records the postion/time of passing of a molecule given a certain distance metric. It should be expanded to include other surface/molecule based metrics in the future and be capable of storing information that exists outside of molecule position.

    Freeze frame should do one of two things: A. It should be able to pass back a way for ChromaColumn to view itself or B. Be able to ask for information from ChromaColumn to trigger a freeze frame.

    Some condition types:
    Pass-the-Post: Pick a position in the column and evaluate the state of the column when the first/last/nth molecule passes this position.
    Stopwatch: Wait for a certain amount of time to elapse, then capture state of the column

    Some return types:
    In space - Return where the molecules are in the column -> Can occur instantaneously once the condition has been met.
    In time - Return the time that each molecule achieved the specified condition -> Must occur in-situ and be checked for each molecule.

    Assume that all statistics that we would want to check can be performed in post processing via ChromatogramProcessor.

    freezeFrame is not meant to store full molecule data: It is handed a molecule, evaluates it, then takes what data it needs. No more than that.

    Only one freeze frame should exist and it should hold all of our conditions. The evalate method is a general call from chromaColumn. When a molecule hits the correct flag structure, it will trigger a recall of all relevant data by sending a signal to chromaColumn and using the captureFrame method.

    This way we can mix both temporal and spatial conditions to grab all the information we might want.
    """
    def setUpMovie(self, framesPerFreeze = 100, compress = False):
        """
        Method to set the system to make a 'movie' file by repeatedly grabbing frames after a set number of ticks.

        Inputs:
        framesPerFreeze - Number of clock ticks that should pass before a freeze frame is triggered
        compress        - boolean to state if the data should be binned before being saved. This should save a HUGE amount of disk space but might make it hard to extract other bits of data.
        """
        self.initConditions()
        self.movieBool = True
        self.addCondition('n-wait', self.getClockTime()+framesPerFreeze, stable_aux = framesPerFreeze)

    def initConditions(self):
        """
        Initialization method for all for freezeFrame. This triggers automatically
        """
        if self.competitive is False:
            print('The column must run in competitive mode in order to freeze frames appropriately.')
            return -1
        # Static listing of all accepted spatial types
        self.accepted_spatial_conditions = set(['n-past', 's-past'])
        """
        n-past - Record state of the system after n molecules pass by some point in the column. Limits temporal information if n != all molecules in column
        s-past - Record state of the system after some specific molecule passes. Lmits temporal information if s is not the last molecule past a point.
        """
        # Static listing of all accepted temporal types
        self.accepted_temporal_conditions = set(['n-wait'])
        """
        n-wait - Record state of the system after n time steps have passed. Can only report spatial information as all molecules share the clock.
        """
        self.conditionals_active = True
        self.movieBool = False
        self.conditionals_list = list()

    def addCondition(self, condition_type, condition_ref, aux = None, stable_aux = None, volatile_aux = None):
        """
        Method to add a condition to the total freeze frame condition list.
        
        Inputs:
        condition_type - Codification that dictates how the freeze frame should trigger.
        condition_ref  - Codification that dictates when the freeze frame should trigger.
        num_molecules  - ChromaColumn given number of molecules that freezeFrame should evaluate to clear the condition.
        """

        # One way of doing this.
        if condition_type in self.accepted_spatial_conditions:
            c = conditional(condition_type, condition_ref, aux = self.num_molecules, stable_aux = np.zeros(self.num_molecules), volatile_aux = 0)
            self.conditionals_list.append(c)
            print('Successfully added spatial conditional '+condition_type+' with ref of '+str(condition_ref))
        elif condition_type in self.accepted_temporal_conditions:
            c = conditional(condition_type, condition_ref, stable_aux = stable_aux)
            self.conditionals_list.append(c)
            print('Successfully added temporal conditional '+condition_type+' with ref of '+str(condition_ref))
        else:
            print('The given condition does not exist within the accepted conditions list.')
            return -1

    def sayConditions(self, inDepth = False):
        """
        Method to debug freezeFrame conditions by printing their information to screen.
        """
        if self.conditionals_list is None:
            print("No active conditions")
        print('I have ', str(len(self.conditionals_list)),' conditions running.')
        if inDepth:
            print('These are their specifics:')
            print('Type \t\t Ref \t Aux')
            for i in self.conditionals_list:
                print(i.giveType(), '\t\t', i.giveRef(), '\t', i.giveAuxVars())
    
    def evaluateMolecules(self):
        """
        This method takes every molecule and evaluates it against all freeze frame conditions that are currently active.
        """
        for c in self.conditionals_list:
            if c.checkCompBool() is False:
                eval_func = self.getEvalFunction(c.giveType())
                for m in range(len(self.molecules)):
                    running_bool = eval_func(c, self.molecules[m], tag = m)
                if running_bool:
                    self.captureFrame(c)
                    c.completeSelf()
                else:
                    c.reset()

    def getEvalFunction(self, conditional_type):
        if conditional_type == 'n-past':
            return self.npast
        elif conditional_type == 's-past':
            return self.spast
        elif conditional_type == 'n-wait':
            return self.nwait
        else:
            print("No evaluation function could be found")

    def npast(self, c, mol, tag):
        post = c.giveRef()
        n = c.giveAuxVars()
        already_passed = c.giveVolatileAuxOutputs()

        if mol.getDistance() >= post:
            time_store = c.giveStableAuxOutputs()
            if time_store[tag] == 0:
                time_store[tag] = self.getClockTime()
            already_passed += 1
            c.updateVolatileAuxOutputs(already_passed)
            c.updateStableAuxOutputs(time_store)
      
        if already_passed >= n:
            return True
        else:
            return False

    def spast(self, c, mol, tag):
        post = c.giveRef()
        s = c.giveAuxVars()
        n = c.giveAuxVars()
        already_passed = c.giveVolatileAuxOutputs()
        
        if s == -1:
            s = len(self.molecules)
        elif s == 0:
            s = 1
        else:
            s = np.ceil(s * len(self.molecules))
        
        if mol.getDistance() >= post:
            time_store = c.giveStableAuxOutputs()
            if time_store[tag] == 0:
                time_store[tag] = self.getClockTime()
            already_passed += 1
            c.updateVolatileAuxOutputs(already_passed)
            c.updateStableAuxOutputs(time_store)

        if already_passed >= s:
            return True
    
    def nwait(self, c, mol, tag):
        time = c.giveRef()
        if self.getClockTime() == time:
            if self.movieBool and not c.isComplete():
                c.completeSelf()
                freezePerFrame = c.giveStableAuxOutputs()
                self.addCondition('n-wait', self.getClockTime()+freezePerFrame, stable_aux = freezePerFrame)
            return True
        return False

    # Maybe consider a method that looks at when the peak of the chromatogram passes through?

    def captureFrame(self, c):
        """
        This method should summate all the data we want and package it nicely into a folder inside of the chromaColumn data storage area.
        """
        this_dir = self.sayCurrentRunDirectory()+'freezeFrames/'+str(c.giveType())+'/'+str(c.giveRef())+'/'
        print("Freezing current frame in the following directory: ", this_dir)
        os.makedirs(this_dir)
        
        # Save position data
        position_data = self.currentProfile()[0]
        results = pd.DataFrame({'Spatial Profile' : position_data})
        results.to_csv(this_dir + 'spatial_data.csv')

        if c.giveType() in self.accepted_spatial_conditions:
            time_data = c.giveStableAuxOutputs()
            results = pd.DataFrame({'Temporal Profile' : time_data})
            results.to_csv(this_dir + 'temporal_data.csv')

        print('All information has been saved for freezeFrame ', str(c.giveType()), ' with reference ', str(c.giveRef()))
        """
        Future ideas for information to pull:
        Molecular adsorption info - Site types/counts per site type
        """

###########################################################################################
################################# Conditionals ############################################
###########################################################################################

class conditional:
    def __init__(self, con_type, con_ref, aux = None, stable_aux = None, volatile_aux = None):
        self.con_type = con_type    # Stores the typing of the conditional
        self.con_ref = con_ref      # Stores the reference information
        self.aux_vars = aux         # Defined by what conditional we are using. Could be a list if necessary
        self.completion_bool = False # States if the condition has been met
        self.stable_aux_outputs = stable_aux # Set of variables that does not get reset from conditional resets
        self.original_volatile_aux = volatile_aux # Should be zeros/null values that define structure
        self.volatile_aux_outputs = volatile_aux # Setting these variables to their initial state
    """
    aux_vars patterns
    """
    def giveType(self):
        return self.con_type
    def giveRef(self):
        return self.con_ref
    def giveAuxVars(self):
        return self.aux_vars
    def checkCompBool(self):
        return self.completion_bool
    def giveVolatileAuxOutputs(self):
        return self.volatile_aux_outputs
    def giveStableAuxOutputs(self):
        return self.stable_aux_outputs
    def completeSelf(self):
        self.completion_bool = True
    def isComplete(self):
        return self.completion_bool
    def updateVolatileAuxOutputs(self, update):
        # This probably needs to be written so that it is condition dependent
        self.volatile_aux_outputs = update
    def updateStableAuxOutputs(self, update):
        self.stable_aux = update    
    def reset(self):
        self.volatile_aux_outputs = self.original_volatile_aux

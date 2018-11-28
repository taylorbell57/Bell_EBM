.. Bell_EBM documentation master file, created by
   sphinx-quickstart on Fri Nov 2 15:29:29 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Bell_EBM's documentation!
====================================

Bell_EBM is an object-oriented and flexible Energy Balance Model (EBM) that can be used to model the temperature of exoplanet atmospheres and observations of those planets. A wide range of planet compositions can be modelled: rocky planets, ocean worlds, and gas atmospheres. This is done by assuming there is a single, fully mixed layer which absorbs all of the incident radiation. The depth of this layer and the layer's heat capacity will change depending on the type of planet composition you are modelling. In the future, the hope is to have multiple layers to allow for simultaneous modelling of an atmosphere and an ocean or rock covered surface. At its core though, this is just an EBM, and no north-south (meridional) flows are modelled (e.g. no Hadley cells), and only manually selected solid-body rotation is permitted for east-west (zonal) flows (e.g. no jets).

Package Usage
=============

Check out the `Quickstart Tutorial <docs/Bell_EBM_Tutorial.html>`_ to get an idea of the capabilities of this EBM, and explore the API for a more detailed description of each of the functions and objects. But the simplest, default usage of the model is:

.. code-block:: python

   import Bell_EBM as ebm
   
   planet = ebm.Planet() # Many planetary parameters can be passed as arguments
   star = ebm.Star() # Basic stellar parameters can be passed as arguments
   system = ebm.System(star, planet)


.. toctree::
   :maxdepth: 2
   :caption: API Table of Contents:

   Bell_EBM


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

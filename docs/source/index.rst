.. C4dynamics documentation master file, created by
   sphinx-quickstart on Thu Nov 23 16:12:22 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. title:: Home 

.. raw:: html

  <div class = "text-center">
  <img  src = "_static/c4dlogo.svg" 
    class = "mx-auto my-4 dark-light" 
    style = "width: 200%; max-width: 200px;">
  <h1 class="display-1">C4DYNAMICS</h1>
  <p class="fs-4"><strong>(Tsipor Dynamics)</strong></p>
  <p class="fs-4"><strong>Elevating your algorithms</strong></p>
  </div>


``c4dynamics`` is a Python framework for developing algorithms in space and time.

`Source repository <https://github.com/C4dynamics/C4dynamics>`_

**c4dynamics** is designed to 
simplify the development of algorithms for dynamic systems, 
using state space representations. 
It offers engineers and researchers a systematic approach to model, 
simulate, and control systems in fields like ``robotics``, 
``aerospace``, and ``navigation``.

The framework introduces ``state objects``, which are foundational 
data structures that encapsulate state vectors and provide 
the tools for managing data, simulating system behavior, 
and analyzing results. 

With integrated modules for sensors, 
detectors, and filters, 
c4dynamics accelerates algorithm development 
while maintaining flexibility and scalability.




.. raw:: html

  <h2>Quickstart</h2>
   

.. code-block:: bash

  pip install c4dynamics     

Copy and run the following snippet in a Python script or a Jupyter notebook:  

.. code:: 

  >>> import c4dynamics as c4d   

.. code:: 

  >>> s = c4d.state(y = 1, vy = 0.5)   # define object of two variables in the state space (y, vy) with initial conditions. 
  >>> s.store(t = 0)                   # store the state  
  >>> F = [[1, 1],                     # transition matrix 
  ...      [0, 1]]              
  >>> s.X += F @ s.X                   # propogate the state through transition matrix  
  >>> s.store(t = 1)                   # store the new state 

.. code:: 

  >>> print(s)  
  [ y  vy ]
  >>> s.X 
  [2.5  1]
  >>> s.data('y')                      # extract stored data for the state varable 'y'. other option: s.plot('y')    
  ([0,  1], [1,  2.5])




.. raw:: html

  <h2>State Objects</h2>

At the core of c4dynamics are **state objects** — 
data structures that encapsulate the variables 
defining a system's state. 
State objects offer a clear interface for 
getting and setting the state vector and 
performing state space transformations.

For instance, with a state of two variables :math:`y` and :math:`v_y`, 
a state object `s` can be defined as follows:

.. code:: 

  >>> s = c4d.state(y = 1, vy = 0.5)
  >>> print(s)
  [ y  vy ]

The state vector `X` represents the current snapshot of 
the system's state variables:

.. code:: 

  >>> print(s.X)
  [1  0.5]



Use Cases for State Objects:

- **Researchers and Developers**: Working on dynamic systems in control theory, 
  robotics, guidance, etc.
- **Educators and Learners**: Exploring and experimenting 
  with state-based modeling, Kalman filtering, and sensor fusion.
- **Engineers**: Implementing practical applications in navigation, 
  estimation, and control systems.


For more details refer to :mod:`states <c4dynamics.states>`.


.. raw:: html

  <h2>Workflow</h2>

The following flowchart outlining a 
typical workflow with c4dynamics:



.. figure:: /_architecture/workflow.png 

  **Typical workflow with c4dynamics**




.. raw:: html

  <h2>Modules</h2>
 

- **State Data Structures**: 
  Custom and predefined state objects 
  facilitate efficient 
  data management and mathematical operations. 
- **Sensors**: 
  Simulate sensor models to provide realistic inputs.
- **Detectors**: 
  API and data structure designed for reading inputs from objects detection models.
- **Filters**: 
  Kalman filters for real-time tracking and prediction.
- **Utilities**: 
  Additional tools for system development and analysis.




.. raw:: html

  <h2>Documentation Sections</h2>
 
The documentation is organized as follows:

- **Installation**: Simple instructions for setting up `c4dynamics`. 

- **Getting Started**: An overview of the core concepts and features of c4dynamics, with beginner-friendly examples to help you get started quickly.

- **Programs**: Real-world use cases showcasing how c4dynamics can be applied in fields such as robotics, aerospace, and computer vision.

- **API Reference**: Comprehensive documentation for technical users, covering all the classes, functions, and modules available in c4dynamics.


.. toctree::
  :maxdepth: 1
  :hidden:

  Installation <installation>
  Getting Started <gettingstarted>
  API Reference <api/index> 


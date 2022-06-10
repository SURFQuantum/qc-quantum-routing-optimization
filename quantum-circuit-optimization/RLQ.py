import plotting
from collections import defaultdict


V = {(10,12,True):2,(10,11,True):1,(10,13,True):3,(10,14,True):4,(10,15,False):5,(10,16,False):6,(10,17,False):7,(10,18,False):8,
     (10,12,True):2,(10,11,True):1,(10,13,True):3,(10,14,True):4,(10,15,False):5,(10,16,False):6,(10,17,False):7,(10,18,False):8,
     (10,12,True):2,(10,11,True):1,(10,13,True):3,(10,14,True):4,(10,15,False):5,(10,16,False):6,(10,17,False):7,(10,18,False):8,
     (10,12,True):2,(10,11,True):1,(10,13,True):3,(10,14,True):4,(10,15,False):5,(10,16,False):6,(10,17,False):7,(10,18,False):8,
     (10,12,True):2,(10,11,True):1,(10,13,True):3,(10,14,True):4,(10,15,False):5,(10,16,False):6,(10,17,False):7,(10,18,False):8,
     (10,12,True):2,(10,11,True):1,(10,13,True):3,(10,14,True):4,(10,15,False):5,(10,16,False):6,(10,17,False):7,(10,18,False):8,
     (10,12,True):2,(10,11,True):1,(10,13,True):3,(10,14,True):4,(10,15,False):5,(10,16,False):6,(10,17,False):7,(10,18,False):8,
     (10,12,True):2,(10,11,True):1,(10,13,True):3,(10,14,True):4,(10,15,False):5,(10,16,False):6,(10,17,False):7,(10,18,False):8,
     (10,12,True):2,(10,11,True):1,(10,13,True):3,(10,14,True):4,(10,15,False):5,(10,16,False):6,(10,17,False):7,(10,18,False):8,
     (10,12,True):2,(10,11,True):1,(10,13,True):3,(10,14,True):4,(10,15,False):5,(10,16,False):6,(10,17,False):7,(10,18,False):8}

plotting.plot_value_function(V, title="80 Simulations")
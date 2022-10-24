import sys
import srepar.examples.utils
    
if __name__ == '__main__':
    srepar.examples.utils.run('scanvi', sys.modules[__name__].__package__)

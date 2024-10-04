import argparse
import matplotlib.pyplot as plt
import json
import pathlib
import numpy as np
import re
import copy
from distutils.util import strtobool

def str2num(datum):
    """ Convert string to integer or float"""

    try:
        return int(datum)
    except:
        try:
            return float(datum)
        except:
            try:
                return strtobool(datum)
            except:
                return datum
            
def get_details(str_in_bench):
    matches = re.findall(r'<(.*?)>', str_in_bench)[0].split(',')

    precision, layout, param = matches
    layout = layout.split('::')[-1]
    param = param.replace(" ", "")

    #params = param.split('_')[-3:]
    #_param = [param_dict[key][param] for key, param in zip(param_keys, params)]
    
    return precision, layout, param_dict[param]
    
def parse():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-dirname', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default='./', \
                        type=str, \
                        choices=None, \
                        help='directory of inputfile', \
                        metavar=None
                       )
    
    parser.add_argument('--filename', \
                        action='store', \
                        nargs='?', \
                        const=None, \
                        default='gesv_gpu_bench.json', \
                        type=str, \
                        choices=None, \
                        help='input file name', \
                        metavar=None
                       )
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    
    # Read json file
    filename = pathlib.Path(args.dirname) / args.filename
    with open(filename, 'r') as f:
        data = json.load(f);
        
    benchmarks = data.get('benchmarks')
    
    # Reconfigure a dict
    param_dict = {'param_s': "Static",
                  'param_d2': "Dynamic",
                 }
    param_keys = ['uplo', 'trans', 'diag']
    keys = ['real_time', 'GB/s']
    layouts = ['LayoutLeft', 'LayoutRight'] 
    results = {}
    for bench in benchmarks:
        print(bench)
        full_name = bench.get('name')
        list_of_names = full_name.split('/')
        name, N, batch, _ = list_of_names
        
        def get_number(str_in_bench):
            str = str_in_bench.split(':')[-1]
            return str2num(str)
        
        N = get_number(N)
        batch = get_number(batch)
        
        precision, layout, pivoting = get_details(name)
        print(precision, layout, pivoting)
        
        result = {key: bench.get(key) for key in keys}
        result['precision'] = precision
        result['layout'] = layout
        result['pivoting'] = pivoting
        result['N'] = N
        result['batch'] = batch
        result['tag'] = name
        print(result)
        results[full_name] = result
    
    def to_list(key):
        _list = [result_dict[key] for result_dict in results.values()]
        _list = set(_list) # remove overlap
        _list = sorted(list(_list)) # To ascending order
        
        return _list
    
    tags = to_list('tag')
    mat_sizes = to_list('N')
    
    #for tag in tags:
    #    precision, layout, pivoting = get_details(tag)
    #    print(tag)
    
    # Plot
    for mat_size in mat_sizes:
        fig, ax = plt.subplots(figsize=(8, 8))
        for key in param_dict.keys():
            time = [result_dict['real_time'] for result_dict in results.values() 
                    if ((key in result_dict['tag']) and ("Left" in result_dict['tag'])) and result_dict['N'] == mat_size]
            batch = [result_dict['batch'] for result_dict in results.values() 
                    if ((key in result_dict['tag']) and ("Left" in result_dict['tag'])) and result_dict['N'] == mat_size]
            ax.plot(batch, np.asarray(time) * 1.0e-6, marker='*', markersize=10, label=param_dict[key])
            
            #ax.set_ylim([0, max_bandwidth])
        ax.set_xscale('log')
        ax.set_xlabel('Batch size')
        ax.set_ylabel('time [s]')
        ax.legend()
        ax.grid()
        fig.savefig(f'N_{mat_size}.png')
        plt.close('all')
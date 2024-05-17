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

    params = param.split('_')[-3:]
    uplo, trans, diag = [param_dict[key][param] for key, param in zip(param_keys, params)]
    
    return precision, layout, uplo, trans, diag
    
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
                        default='tbsv_bench.json', \
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
    param_dict = {'uplo': {'l': 'Lower', 'u': 'Upper'},
                  'trans': {'n': 'NoTrans', 't': 'Trans', 'c': 'ConjTrans'},
                  'diag': {'n': 'NonUnit', 'u': 'Unit'},
                 }
    param_keys = ['uplo', 'trans', 'diag']
    keys = ['real_time', 'GB/s']
    layouts = ['LayoutLeft', 'LayoutRight'] 
    results = {}
    for bench in benchmarks:
        full_name = bench.get('name')
        list_of_names = full_name.split('/')
        name, N, batch, _ = list_of_names
        
        def get_number(str_in_bench):
            str = str_in_bench.split(':')[-1]
            return str2num(str)
        
        N = get_number(N)
        batch = get_number(batch)
        
        precision, layout, uplo, trans, diag = get_details(name)
        
        result = {key: bench.get(key) for key in keys}
        result['precision'] = precision
        result['layout'] = layout
        result['uplo'] = uplo
        result['trans'] = trans
        result['diag'] = diag
        result['N'] = N
        result['batch'] = batch
        result['tag'] = name
        results[full_name] = result
    
    def to_list(key):
        _list = [result_dict[key] for result_dict in results.values()]
        _list = set(_list) # remove overlap
        _list = sorted(list(_list)) # To ascending order
        
        return _list
    
    tags = to_list('tag')
    mat_sizes = to_list('N')
    
    for tag in tags:
        precision, layout, uplo, trans, diag = get_details(tag)
        
        all_bandwidth = []
        def replace_tag(tag, new_layout):
            new_tag = copy.copy(tag)
            for _layout in layouts:
                if _layout in tag:
                    new_tag = new_tag.replace(_layout, new_layout)
                    
            return new_tag
        
        for _layout in layouts:
            tag_with_layout = replace_tag(tag, _layout)
            for mat_size in mat_sizes:
                bandwidth = [result_dict['GB/s'] for result_dict in results.values() if result_dict['tag'] == tag_with_layout and result_dict['N'] == mat_size]
            
                all_bandwidth.append(bandwidth)
        
        max_bandwidth = np.max(all_bandwidth)
        
        # Plot
        fig, axes = plt.subplots(figsize=(16, 6), ncols=2)
        for ax, _layout in zip(axes, layouts):
            tag_with_layout = replace_tag(tag, _layout)
            for mat_size in mat_sizes:
                bandwidth = [result_dict['GB/s'] for result_dict in results.values() 
                             if result_dict['tag'] == tag_with_layout and result_dict['N'] == mat_size]
                batch = [result_dict['batch'] for result_dict in results.values()
                         if result_dict['tag'] == tag_with_layout and result_dict['N'] == mat_size]
                     
                ax.plot(batch, bandwidth, marker='o', markersize=5, label=f'N={mat_size}')
        
            ax.set_xscale('log')
            ax.set_xlabel('Batch size')
            ax.set_ylabel('[GB/s]')
            ax.set_ylim([0, max_bandwidth])
            ax.set_title(f'{_layout}, {uplo}, {trans}, {diag}')
            ax.legend()
            ax.grid()
        fig.savefig(f'{tag}.png')
        plt.close('all')
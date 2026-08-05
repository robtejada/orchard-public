'''
short little script that serves as a "cli" to APPLE, to cut down on time spent
reloading eos module

Usage:
python orchard_cli.py

Then type in the name of the parameters file you want to run

It will reread the params file, reload all modules in RELOAD_MODULES,
evolution.py, then rerun the code as if you'd typed
`python evolution.py --config <XXX>`

You can also manually reload a file by typing "reload transport" or "reload
transport.py" (with or without file extension is fine), in case you changed code


Provided as-is, it may not work super well lol
'''
# try to provide support for up-arrow etc., fail quietly if not
try:
    import readline
except:
    print("cannot load readline, up-arrow and search may not work; try `pip install readline` if you'd like these features")
    pass
import traceback
import os
import sys
import evolution
import importlib

# reloaded in order for every specified params file (evolution.py will always be
# reloaded at end)
RELOAD_MODULES = [
                'utils.common', 'initial', 'transport', 'hydrostatic', 'utils.mantle_core_props',
                'atm_bc',
                'utils.profiles',
                'misc.gupta_misc',
                'misc.gupta_misc_lowT',
                'misc.gilmore_misc',
                'utils.hdf5_evol_saver',
                  ]

MODULE_ALIASES = {
                'common': 'utils.common',
                'profiles': 'utils.profiles',
                'mantle_core_props': 'utils.mantle_core_props',
                'hdf5_evol_saver': 'utils.hdf5_evol_saver',
                'gupta_misc': 'misc.gupta_misc',
                'gupta_misc_lowT': 'misc.gupta_misc_lowT',
                'gilmore_misc': 'misc.gilmore_misc',
                }

inline = ''
while True:
    inline = input('\nName of params file (with path) [type "help" for usage]: ')
    inline = inline.strip()

    prev = None
    if inline == 'break':
        break
    elif inline == 'ls':
        folder = input('Enter folder to list files: ').strip()
        if os.path.isdir(folder):
            lsret = os.listdir(folder)
            print('Files include:')
            for file in lsret:
                if not file.startswith('.'):
                    print(f'- {file}')
        else:
            print(f'Folder "{folder}" does not exist.')
    elif inline == 'help':
        print('\tFuzzy-search for parameters file to run using evolution.py')
        print('\t- Keywords include: ["break", "ls", "help"]')
    elif inline.startswith('reload'):
        # reload by filename (with or without ".py"; just in case...)
        module_name = inline.split(' ')[1].replace('.py', '')
        module_name = MODULE_ALIASES.get(module_name, module_name)
        importlib.reload(importlib.import_module(module_name))
    else:
        print('Trying to run for filename', inline)
        try:
            # allow fuzzy search for filenames
            folder, filename = os.path.split(inline)
            if not folder:
                folder = '.'
            lsret = os.listdir(folder)
            searchfn = None
            for file in lsret:
                if file.startswith(filename):
                    if searchfn is None:
                        searchfn = file
                    else:
                        print(f'Multiple files found starting with {filename} in list:', lsret)
                        print('Please give a more specific filename')
                        break
            else:
                if searchfn is not None:
                    print('Did not find for', filename)
                    inline = os.path.join(folder, searchfn)

            sys.argv = [sys.argv[0], '--config', inline]
            for m in RELOAD_MODULES:
                importlib.reload(importlib.import_module(m))
            importlib.reload(evolution)
            evolution.run()
        except:
            print(traceback.print_exc())
        prev = inline

import os, sys
workdir = os.path.abspath(os.getcwd())
rundir = '.venv\\Scripts\\python.exe'
command = f'{os.path.join(workdir, rundir)} -m {" ".join(sys.argv[1:])}'
print(f'run: {command}'); os.system(command)
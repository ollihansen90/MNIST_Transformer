docker run --gpus all --name dl -it --rm -v $(pwd):/workingdir -v $HOME:/data --user $(id -u):$(id -g) dl_workingdir_hohansen python3 main.py >>output.txt 2>&1
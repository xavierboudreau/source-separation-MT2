# source-separation-MT2


Using the neural network design of https://github.com/sigsep/open-unmix-pytorch, this project trains a model to separate vocals from a track of music. It also includes a web app to present the model interactively.

Initially create conda environment using:
`> conda env create -f environment-cpu-osx.yml`

To activate this environment, use:
`> source activate source-separation-MT2`

To deactivate an active environment, use:
`> source deactivate`

To update the yml file (when new packages were installed through conda)
`> conda env export > environment-cpu-osx.yml`

# source-separation-MT2
Using the neural network design of [UMX](https://github.com/sigsep/open-unmix-pytorch), this project trains a model to separate vocals from a file of music. It also includes a [web app](https://github.com/xavierboudreau/source-separation-MT2-website) to present the model interactively.
## Demo of client usage
[![Demo video](http://img.youtube.com/vi/2bpY5lM7T-A/0.jpg)](http://www.youtube.com/watch?v=2bpY5lM7T-A)

## Environment
Clone the repo with  
`> git clone https://github.com/xavierboudreau/source-separation-MT2`  
Be sure you have [miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/), then create conda environment using:   
`> conda env create -f environment-cpu-osx.yml`  
To activate this environment, use:  
`> source activate source-separation-MT2`  
To deactivate an active environment, use:  
`> source deactivate`  
To update the yml file (when new packages were installed through conda)  
`> conda env export > environment-cpu-osx.yml`
## Server
The server for the web app can be run with  
`> python app.py`  
Two directories should be made to store data from the client  
`> mkdir client-files`  
`> mkdir client-separation-results`
### Endpoints
`/graphql`  
Get Epoch objects representing statistics during training  
`/upload`  
Upload a `MP3`, `WAV`, or `M4A` to `client-files`  
`/my-files`  
Get filenames in `client-files`  
`/my-models`  
Get model names in `intermediate_models`  
`/separate`  
Given a filename and model name, estimate the vocal track
## Training
Uses the network design from UMX and the data from [MUSDB](https://sigsep.github.io/datasets/musdb.html) to train the model. I add [early stopping](https://en.wikipedia.org/wiki/Early_stopping) and record the training and validation loss for each epoch in `model_stats.csv`. For each epoch I sample a random sequence of 13 seconds of each track from the training and validation sets. The short sample duration makes training this model feasible on my 4GB machine.

To train run  
`> python train.py`  
Note if you don't have access to full dataset you must instead run  
`> python train.py -demo`  
## Testing
To estimate the audio of a file on disk run  
`> python test.py FILEPATH SAVEPATH MODELPATH`

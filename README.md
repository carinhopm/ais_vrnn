# VRNN Model for AIS Data

This is a VRNN implementation for Global Maritime Surveillance using AIS messages.

## Installation in HPC

To build the Python environment with all the necessary packages to run the model is necessary to execute the _setup.sh_ bash script.

This script can be found inside the _code_ folder: https://github.com/carinhopm/ais_vrnn/blob/main/code/setup.sh

Note that all the instructions listed here are being executed from the _code_ folder.

## Usage

Once the Python environment is configured, the job can be sent to the HPC cluster executing the line above in the terminal:

```bash
bsub < run_model_hpc.sh
```

## Contributing

This repository has been created by:

Carlos Parra Marcelo
Ashwinth Mathivanan

The code is an updated version of the implementation provided by Kristoffer V. Olesen.

Preterm transient burst detector: a guide
============================================

This repo is to serve as a guide on how to use the fully trained preterm transient burst detector xgboost model.

This model corresponds to the 2020 EMBC paper titled 'Detection of Transient Bursts in the EEG of Preterm Infants using
Time–Frequency Distributions and Machine Learning'



---

[Requirements](#requirements) | [Use](#use) | [Files](#files) | [Test computer
setup](#test-computer-setup) | [Licence](#licence) | [References](#references) |
[Contact](#contact)

## Requirements
Python 3.7 or newer with the following packages installed
| Package       | Version installed  |
| ------------- |:------------------:|
| pandas        | 0.25.1             |
| numpy         | 1.17.1             |
| xgboost       | 0.90               |
| mne           | 0.18.2             |
| scipy         | 1.3.1              |


The current version of the code requires MATLAB to compute the time-frequency distributions. The [memeff_TFDs](https://github.com/otoolej/memeff_TFDs)
MATLAB package is what is used to generate the time-frequency distributions.
Please download this package change the variable `path_to_TFD_package` to the location of the package.

To use MATLAB commands in python the following link provides details on how to install the MATLAB engine [LINK](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).

**Note**: A future release will be all python when the MATLAB functions are converted into python.


## Use 

Set path `Preterm_transient_burst_detector` to the folder location of the downloaded files. 
Then the following will load the main functions:
```
  import sys
  sys.path.append(Preterm_transient_burst_detector)
  import use_model
```

Example when using the demo_data.edf file
```
eeg_ts_data, tfd_data_df, y_preds_probs = main(path_to_TFD_package='path_to_memeff_TFDs_package')
```
**Note**: To use a different edf file call function like so
```
eeg_ts_data, tfd_data_df, y_preds_probs = use_model.main(file_name='Own_edf_file.edf', path_to_TFD_package='path_to_memeff_TFDs_package')
```


Example when the eeg data is saved in a csv file (assuming it is bi-polar order, has been filtered and downsampled to 64 Hz)
```
eeg_data = pd.read_csv('demo_data.csv')
eeg_df, tfd_data_df, y_preds_probs = use_model.main(eeg_data=eeg_data, Fs=64, file_name='File_1', channel='F3–C3', path_to_TFD_package='path_to_memeff_TFDs_package')
```
**Note**: A user can load the data whatever way they like as long as it is passed into the main function as a DataFrame, 
where the column names are the bi-polar channel names. If the data is not sampled at 64 Hz please filter and downsample 
to 64 Hz. This can be done either before passing the data in or it can be coded in the `prepare_EEG_data` function.






## Files
The `use_model` python file (.py file) have a description and an example in the header of each function. To read this
header, type `help(use_model.main)` in the console after importing (`import use_model`), to get information about the
main function.  Directory structure is as follows: 
```
├── CHANGELOG.md          # changelog file
├── demo_data.csv         # example csv data
├── demo_data.edf         # example edf data
├── LICENSE.md            # license file 
├── README.md             # readme file describing project
└── use_model.py          # File showing how to use the model
```

**NOTE**: demo_data.edf was created using [NEURAL_py_EEG_feature_set](https://github.com/BrianMur92/NEURAL_py_EEG_feature_set).


## Test computer setup
- hardware:  Intel Core i7-8700K @ 3.2GHz; 32GB memory.
- operating system: Windows 10 64-bit
- software: python 3.7


## Licence

```
Copyright (c) 2020, Brian M. Murphy, University College Cork
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  Neither the name of the University College Cork nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```


## References



## Contact

Brian M. Murphy

Neonatal Brain Research Group,  
[INFANT Research Centre](https://www.infantcentre.ie/),  
Department of Paediatrics and Child Health,  
Room 2.18 UCC Academic Paediatric Unit, Cork University Hospital,  
University College Cork,  
Ireland

- email: Brian.M.Murphy AT umail dot ucc dot ie 
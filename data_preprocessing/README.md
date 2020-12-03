# Data Pre-processing

Run `getdata.sh` under that directory to obtain and pre-process the data. This script will download and process the official data from UD. For CTB5, CTB6, CTB7, and CTB9, you need to obtain the official data yourself, and then put the raw data folder under the `data_preprocessing` directory. The folder name for the CTB datasets should be:

* CTB5: LDC05T01
* CTB6: LDC07T36
* CTB7: LDC10T07
* CTB9: LDC2016T13

All processed data will appear in `data` directory organized by the datasets, where each of them contains the files with the same file names in the `sample_data` folder.

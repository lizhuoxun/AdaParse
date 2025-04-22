# AdaParse
> This repository is official implementation for AdaParse: Personalized Fingerprinting for Visual Generative Model Reverse Engineering.
##  Training/testing

### Training
Run:
```bash
python adaparse.py --savedir <your model dir> --test 1 --device 0
```
- To modify the train/test split, simply adjust the parameter ‘test’
- The accepted range includes [1,2,3,4]
- Each value corresponds to one of the four settings described in the paper

### Testing
- During training, folders are automatically generated with timestamped names
- Rename them to match the experimental setting names, e.g., 'set1' corresponds to the '--test' parameter set to 1

Run:
```bash
python adaparse_test.py --test 1 --device <your gpu id> --model_dir <your model dir>+'/set1/model/'
```

- You will receive test results from all 20 epochs in a setting 
- Change the four best epoch numbers obtained from the four experimental settings of the 'model_list=[1,18,9,15]' line in adaparse_subval.py, in the corresponding order

Run:
```bash
python adaparse_average.py --test 1 --model_dir <your model dir>
```
- You will obtain the average results across all four experimental settings

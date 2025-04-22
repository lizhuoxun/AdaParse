# AdaParse
##  Training/testing

### Training
run
```bash
python adaparse.py --savedir <your model dir> --test 1 --device 0
```

### Testing
run
```bash
python adaparse_val.py --test 1 --device <your gpu id> --model_dir <your model dir>+'/set1/model/'
```
If you want to modify the train/test split, simply adjust the parameter ‘test’. The accepted range includes [1,2,3,4]. Each value corresponds to one of the four settings described in the paper.

run
```bash
python adaparse_subval.py --test 1 --model_dir <your model dir>
```

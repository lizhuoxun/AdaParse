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

# CS420-SJTU-Project

## Usage

### Unet

#### Training stage

```
python train.py --root='../dataset/membrane' --save_path='unet_membrane.hdf5'
```

#### Test stage

```
python test.py --root='../dataset/membrane' --model_path='unet_membrane.hdf5'
```



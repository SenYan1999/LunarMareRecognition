# Lunar Mare Recognition

This repo is the code for my CS dissertation which aims to recognize mare given a picture of lunar area.

## Installation
```shell script
pip install -r requirements.txt
```

## Running
1. Fetch the data from BaiduNetDisk(to be continued)

2. Train model

    - **One Card**
        ``` shell script
        python main.py --train --fp16
        ```
    - **Distributed Training**
        ```shell script
        python -m torch.distributed.launch --nproc_per_node 3 main.py --train --distributed --fp16
        ```

## Results
To be continued
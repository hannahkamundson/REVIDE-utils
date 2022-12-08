# README

The REVIDE dataset originates from here: http://xinyizhang.tech/revide/
The data can be downloaded here: https://www.kaggle.com/datasets/hannahkamundson/revide-indoor
Based off of the following repos: https://github.com/csbhr/SGID-PFF and https://github.com/hannahkamundson/Video-Dehazing-Extension

## How To Install
You can install this from the git repo.
```sh
pip install git+https://github.com/hannahkamundson/REVIDE-utils
```

## Example Folder Structure
This is the expected folder structure:
```
| train
    | hazy
        | video 1
            | frame 1 image of video 1
            | frame 2 image of video 1
            | ...
        | video 2
            | frame 1 image of video 2
            | frame 2 image of video 2
            | ...
        | ...
    | clear
        | video 1
            | frame 1 image of video 1
            | frame 2 image of video 1
            | ...
        | video 2
            | frame 1 image of video 2
            | frame 2 image of video 2
            | ...
        | ...
| test
    | hazy
        | video 3
            | frame 1 image of video 3
            | frame 2 image of video 3
            | ...
        | video 4
            | frame 1 image of video 4
            | frame 2 image of video 4
            | ...
        | ...
    | clear
        | video 3
            | frame 1 image of video 3
            | frame 2 image of video 3
            | ...
        | video 4
            | frame 1 image of video 4
            | frame 2 image of video 4
            | ...
        | ...
```

## Example Usage
This is based on the above folder structure example
```python
from torch.utils.data import DataLoader
from revide_dataset import RevideDataset

revide_dataset=RevideDataset(train_directory='Train', 
                             test_directory='Test', 
                             clear_folder='clear', 
                             hazy_folder='hazy')
loader=DataLoader(
        dataset=revide_dataset,
        batch_size=batch_size,
        shuffle=is_train,
        pin_memory=True,
        num_workers=number_of_threads
```
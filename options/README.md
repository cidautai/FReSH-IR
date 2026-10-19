## Configuration files

This subfolder contains all the configuration files needed for training/testing/predicting your models. The whole repository is intended to work around these configuration files. 

> You can have as many keys written in the configs, that they will only be used the ones needed to work. For example, if you want to add a new network architecture, you'll surely need to add new keys to define them at [/archs/__init__.py](/archs/__init__.py). The existing keys don't need to be removed if they are unused. The requested keys will be handled automatically

## Train configurations

The train configurations may follow the structure in [train](/options/train/Baseline.yml). The keys are divided into 5 different topics:

- The **device** information, that gathers the most important information on the environment training setup.

- The **dataset** section, where information related to the used dataset for training is requested.

    > **Important!** If you are **not** using SCAYLE environment then the `train_path` and `test_path` need to be absolute routes. In the case of SCAYLE, ayou just need to write the relative path from the `$SCRATCH` directory.
- **network** section stablishes the parameters of the network architecture. It also decides if you resume or you use guiding losses.

- **optimization** section is the one related to the whole optimization process. Information about optimizer, scheduler, epochs, steps and used losses (per step) is introduced here.

- Finally, **wandb** is related to the uploading of results to wandb.

> Note that in the **dataset** and **optimization** sections, there are a few keys that need to receive a list of element (whether they are bools, ints, floats or lists). All these elements that need a list are related with a posible multi-step training that can be performed within this framework. The `optimization.steps` key is the one that stablishes how many stages in the training will be used. If you set `optimization.steps=1`, then only the first element of the mention keys will be used. Please pay attention to the len of these key lists (the ones with a comment of `STEP`), that it should be the same as the number of steps. If only one step is performed, it has been tested that the framework will work properly if the elements of this keys are lists, e.g. `optimization.pixel_flag: [True]`. Otherwise, It won't work.

> There are some keys that also need to be introduced as lists, but don't have the `STEP` comment on them. They won't change through the stages.

## Test and Predict configurations

These two types of config files have fewer number of keys, as all the training don't need to be done and it is not expected to upload the results to wandb. Some features of these configs:

- Eventhough during testing you won't be using the training dataset, it is requested in this framework. So, the paths regarding this dataset need to be present in the test config file.
- In the case of predict (inference) an additional `Resize` key is added to handle large scale images.


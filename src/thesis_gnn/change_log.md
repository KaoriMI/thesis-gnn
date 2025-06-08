# Change Summary

## Description

1. **inference.py**\
   Fixed errors caused by invalid arguments "avg_tps" and "precrec".


2. **model_settings.json**\
   Add parameters for directed GIN model(dir_gin) at the bottom.


3. **models.py**\
   Newly added DirGINe class, which pass message from both incoming and outgoing [to be updated].


6. **training.py**\
　　Add reference to DirGIN.


6. **util.py**\
   Updated the parser for "unique_name" to allow users to add a unique name to the title of trained model when it's saved.
   It will be saved as "checkpoint_[specified name].tar".


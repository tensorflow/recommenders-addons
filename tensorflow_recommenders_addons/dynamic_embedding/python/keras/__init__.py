from tensorflow_recommenders_addons.dynamic_embedding.python.keras import layers
from tensorflow_recommenders_addons.dynamic_embedding.python.keras import callbacks
from tensorflow_recommenders_addons.dynamic_embedding.python.keras import models

setattr(models, 'save_model', models.de_save_model)
setattr(callbacks, 'ModelCheckpoint', callbacks.DEHvdModelCheckpoint)
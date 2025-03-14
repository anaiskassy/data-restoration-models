from data_restoration.evaluation import *
from data_restoration.base_models import *
from data_restoration.load_data import *

# load head cats
data = load_data_head_small()
data_train = data[:-100]
data_test = data[-100:]

# initilization models
generator,gen_opti,discriminator,disc_opti = init_base_model()

# train base model and save weights
history_gen, history_disc , predictions_finales , progressive_output = run_base_model(
    data_train=data_train,
    generator=generator,
    gen_opti=gen_opti,
    discriminator=discriminator,
    disc_opti=disc_opti,
    n_epochs=int(N_EPOCHS),
    batch_size=int(BATCH_SIZE),
    reload_w=RELOAD_W,
    checkpoint=int(CHECKPOINT),
    workbook=False
)

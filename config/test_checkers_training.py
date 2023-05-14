# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-shakespeare-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 20
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'


gradient_accumulation_steps = 1
batch_size = 12


# can't be adjusted yet, need to change that. Hardcoded to zero when loading othello model
dropout = 0.0

# load pretrained othello model
init_from = "othello"

# Set this to True to add extra embeddings when training using checkers dataset
train_checkers = True

# Assuming we should finetune at a constant rate based on other finetuning configs
decay_lr = False
learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 1000
lr_decay_iters = 2000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model

# lora config
do_lora = False
r = 8
alpha = 8
lora_dropout = 0.05
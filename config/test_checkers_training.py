# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-checkers'
ckpt_name = 'ckpt.pt'
eval_interval = 200 # keep frequent because we'll overfit
eval_iters = 20
log_interval = 10 # don't print too too often

# Haven't set up model registry on wandb so for now just going to 
# log runs and then re train with the best hyperparams
always_save_checkpoint = False
never_save_checkpoint = True

wandb_log = False # override via command line if you like
wandb_project = 'nanogpt-checkers'
wandb_run_name = 'test-run'


gradient_accumulation_steps = 1 # single gpu training
batch_size = 12
acc_games = 2*batch_size


# can't be adjusted yet, need to change that. Hardcoded to zero when loading othello model
dropout = 0.0

# load pretrained othello model
init_from = "othello"

# othello model params
n_layer = 8
n_head = 8
n_embd = 512
block_size = 59

# Set this to True to add extra embeddings when training using checkers dataset
train_checkers = True
add_annotation_tokens = False

# Assuming we should finetune at a constant rate based on other finetuning configs
decay_lr = False
learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 2000
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
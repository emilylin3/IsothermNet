# Directories
jobPath = "C:/Users/opbir/CGCNN/cgcnn_att/"

# Loading checkpoints/data
load_checkpoint = False
load_hp = True
run_dataProcess = False

# Hyperparameter tuning/optimization
max_evals = 40

# Training/validation/testing parameters
optim = "adam"
num_epoch = 500
train_patience = 50
train_batchSize = 64
val_batchSize = train_batchSize
test_batchSize = train_batchSize
train = True
test = True

# Dataset parameters
indicies = [6]
indx_str = f"_{indicies[0]}"
pos = torch.tensor(indicies[0])

num = [0, 5394]

# Constants
R = 0.008314;              # Universal gas constant [kJ/mol-K]

# Plotting
font = {"family" : "Arial",
        "weight" : "medium",
        "size"   : 10,
        "style"  : "normal"}
font_title = {"family" : "Arial",
              "weight" : "semibold",
              "size"   : 12,
              "style"  : "normal"}
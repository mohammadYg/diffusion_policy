# visualize the model's network architecture using tensorboard
import torch
from torch.utils.tensorboard import SummaryWriter
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from torchviz import make_dot



# Model configuration
action_dim = 2
obs_dim = 20
obs_step_dim = 2
horizon = 16
local_cond_dim = obs_dim
global_cond_dim = obs_dim * obs_step_dim
diffusion_step_embed_dim = 256
down_dims = [256, 512, 1024]
kernel_size = 5
n_groups = 8
cond_predict_scale = True

# Initialize model
model = ConditionalUnet1D(
    input_dim=action_dim,
    local_cond_dim=local_cond_dim,
    global_cond_dim=global_cond_dim,
    diffusion_step_embed_dim=diffusion_step_embed_dim,
    down_dims=down_dims,
    kernel_size=kernel_size,
    n_groups=n_groups,
    cond_predict_scale=cond_predict_scale
)

model.eval()   # set to eval mode (required for graph trace)

# Dummy inputs for graph tracing
dummy_input = torch.randn(1, horizon, action_dim)    # (batch, horizon, action_dim)
dummy_timestep = torch.tensor([10])                  # (batch,)
dummy_local = torch.randn(1, horizon, local_cond_dim)                                   # model supports None local cond
dummy_global = torch.randn(1, global_cond_dim)       # (batch, global_cond_dim)

out = model(dummy_input, dummy_timestep, None, dummy_global)
dot = make_dot(out, params=dict(model.named_parameters()))
dot.render("unet_graph", format="png")



# # Create TensorBoard writer
# writer = SummaryWriter(log_dir="data/outputs/model_eval/")
# # Add graph to TensorBoardwich 
# writer.add_graph(
#     model,
#     input_to_model=(dummy_input, dummy_timestep, dummy_local, dummy_global)
# )

# writer.close()

# # Save model
# torch.save(model.state_dict(), 'data/outputs/model_eval/conditional_unet1d.pth')

# print("Model graph saved. Run: tensorboard --logdir data/outputs/model_eval/")
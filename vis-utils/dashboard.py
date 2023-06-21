import torch
import sys
sys.path.append("../")
from config import *
import streamlit as st
from models import get_model_optimizer_and_step
from models.common import load_saved_state
from datasets import get_dataset_chw, get_dataset_info, get_dataset_loaders
import matplotlib.pyplot as plt
from models.common import (
    Functional,
    edl_loss,
    edl_probs,
    gather_samples,
    margin_loss,
)

## params
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["MODEL_NAME"] = "rescaps"
os.environ["MANIFOLD_D"] = "50"
os.environ["CV_MODE"] = "k-fold"
os.environ["CV_FOLDS"] = "10"
os.environ["DATASET_NAME"] = "AMRB2_species.4"

@st.cache_data
def load_stuff():
    config = load_config()
    config.image_chw = get_dataset_chw(
            dataset_name=config.dataset_name,
    )

    config.dataset_info = get_dataset_info(
        dataset_name=config.dataset_name,
        data_root=config.data_dir,
        cv_mode=config.cv_mode,
    )

    train_loader, test_loader = get_dataset_loaders(
            dataset_name = config.dataset_name,
            batch_size_train = 32,
            batch_size_test = 32,
            data_root = "/Users/ramith_1/Datasets/",
            cv_k = config.cv_k,
            cv_folds = config.cv_folds,
            cv_mode = config.cv_mode,
        )

    PATH = "../experiments/ood_flows/AMRB2_species/53.097923/rescaps_model_e100.pth"

    device = torch.device('cpu')
    weights = torch.load(PATH, map_location=device)

    model, _, _  = get_model_optimizer_and_step(config)
    model.load_state_dict(weights)
    model.to(device)

    encoder = model["encoder"]
    classifier = model["classifier"]
    decoder = model["decoder"]
    
    return encoder, classifier, decoder, train_loader, test_loader


encoder, classifier, decoder, train_loader, test_loader = load_stuff()

st.sidebar.header("Settings")
mode = st.sidebar.selectbox('Choose mode', ['train', 'test'])
filter_only_correct = st.sidebar.checkbox('Filter only correct predictions')

if mode == 'train':
    loader = train_loader
else:
    loader = test_loader
    
tensors = [] 
ys = []

for idx, (x,y) in enumerate(loader):

    z_x = encoder(x.float())
    y_z, sel_z_x = classifier(z_x)
    pY, uY = edl_probs(y_z.detach())
    
    
    pos_to_extract = y.argmax(dim = 1).unsqueeze(-1).unsqueeze(-1).expand(-1, 50, -1)
    
    target_prototypes = z_x.gather(dim = 2, index = pos_to_extract).detach().cpu()
    
    
    #append
    if(filter_only_correct):
        correct_indices = pY.argmax(dim = 1) == y.argmax(dim = 1)
        
        tensors.append(target_prototypes[correct_indices])
        ys.append(y[correct_indices])
    else:
        tensors.append(target_prototypes)
        ys.append(y)
            
        
    
    if(idx > 2):
        break
    
all_target_prototypes = torch.cat(tensors, dim=0)
all_y = torch.cat(ys, dim = 0 )

class_means = []

# for i in range(0,4):
#     mean_vector = all_target_prototypes[all_y.argmax(dim = 1) == i].mean(axis = 0)
    
#     class_means.append(mean_vector)
    
#     plt.figure(figsize=(5,5))
#     plt.imshow(all_target_prototypes[all_y.argmax(dim = 1) == i].detach().cpu())
#     plt.savefig(f"mode_{mode}_class_{i}_only_correct_{filter_only_correct}.png")
    
#     plt.colorbar()
    
# class_reps = torch.cat(class_means, dim=1).T

# plt.figure(figsize=(10,10))
# plt.imshow(class_reps)
# plt.savefig("class_reps.png")

# pos_variance = torch.var(class_reps, dim = 0).reshape(50,1)

# plt.figure(figsize=(24,4))
# plt.bar(range(pos_variance.shape[0]),pos_variance)

# plt.xlabel('Position')
# plt.ylabel('Variance')
# plt.savefig("varience_plot.png")

# with st.expander("Plots"):  # Collapsible section for plots

for i in range(0,4):
    mean_vector = all_target_prototypes[all_y.argmax(dim = 1) == i].mean(axis = 0)
    class_means.append(mean_vector)
    fig = plt.figure(figsize=(3,3))
    plt.imshow(all_target_prototypes[all_y.argmax(dim = 1) == i].detach().cpu())
    st.pyplot(fig)  # Here we use Streamlit's function to show the plot


class_reps = torch.cat(class_means, dim=1).T

fig2 = plt.figure(figsize=(10,10))
plt.imshow(class_reps)
st.pyplot(fig2)

pos_variance = torch.var(class_reps, dim = 0).reshape(50,1)

fig3 = plt.figure(figsize=(24,4))
plt.bar(range(pos_variance.shape[0]),pos_variance)
plt.xlabel('Position')
plt.ylabel('Variance')
st.pyplot(fig3)

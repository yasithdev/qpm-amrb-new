import torch
import sys
sys.path.append("../")
from config import *
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models import get_model_optimizer_and_step
from models.common import load_saved_state
from datasets import get_dataset_chw, get_dataset_info, get_dataset_loaders
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
st.set_page_config(layout="wide")

@st.cache_data
def load_stuff(PATH):
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


    device = torch.device('cpu')
    weights = torch.load(PATH, map_location=device)

    model, _, _  = get_model_optimizer_and_step(config)
    model.load_state_dict(weights)
    model.to(device)

    encoder = model["encoder"]
    classifier = model["classifier"]
    decoder = model["decoder"]
    
    return encoder, classifier, decoder, train_loader, test_loader

base_dir = "../experiments/ood_flows/AMRB2_species/"
models = os.listdir(base_dir)
models = [model for model in models if model != ".ipynb_checkpoints"]

model_name = st.sidebar.selectbox('Choose model', models)

encoder, classifier, decoder, train_loader, test_loader = load_stuff(base_dir + model_name)

st.sidebar.header("CapsNet Latent Dimension Visualizer")

mode = st.sidebar.selectbox('Choose data split', ['train', 'test'])
n_batches = st.sidebar.slider('Number of batches', 0, 10, 3)
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
            
    if(idx == n_batches):
        break
    
all_target_prototypes = torch.cat(tensors, dim=0)
all_y = torch.cat(ys, dim = 0 )

class_means = []

class_names = ["Acinetobacter", "E. Coli", "K. pneumoniae", "S. aureus"]
fig = make_subplots(rows=2, cols=4, subplot_titles=[f'{name} Heatmap' for name in class_names] + [f'{name} Variance' for name in class_names])
max_variance = 0

for i in range(0,4):
    mean_vector = all_target_prototypes[all_y.argmax(dim = 1) == i].mean(axis = 0)
    class_means.append(mean_vector)
    
    img = all_target_prototypes[all_y.argmax(dim = 1) == i].detach().cpu()
    img = img.reshape(img.shape[0],img.shape[1])
    
    # Add the image as a subplot
    fig.add_trace(
        go.Heatmap(z=img, zmin=-1, zmax=1, showscale=False),
        row=1, col=i+1
    )
    
    ## calculate the variance of the class
    pos_intraclass_variance = torch.var(img, dim = 0).reshape(50,1)
    if(pos_intraclass_variance.max() > max_variance):
        max_variance = pos_intraclass_variance.max()
    
    # Add variance as a subplot
    fig.add_trace(
        go.Bar(y=pos_intraclass_variance.squeeze(), name=f"Variance of class {i}"),
        row=2, col=i+1
    )
    
    
fig.update_layout(height=600)
fig.update_yaxes(range=[0, max_variance], row=2)

st.plotly_chart(fig, use_container_width=True)



with st.expander("Inter-Class Variance w.r.t Position (using :red[mean] vectors)"):
    # Calculate the variance between classes
    st.title('Inter-Class Variance w.r.t Position (using :red[mean] vectors)')
    
    class_reps = torch.cat(class_means, dim=1).T
    pos_variance = torch.var(class_reps, dim = 0).reshape(50,1)
    pos_variance_np = pos_variance.numpy()

    fig3 = go.Figure([go.Bar(x=list(range(pos_variance_np.shape[0])), 
                            y=pos_variance_np.flatten(),
                            name='Variance')])  # use flatten to convert 2D array to 1D

    fig3.update_layout(title_text='Inter-Class Variance w.r.t Position',
                    xaxis_title='Position',
                    yaxis_title='Variance')

    st.plotly_chart(fig3,  use_container_width=True)


    fig2 = px.imshow(class_reps, title="Class Mean Vectors")
    st.plotly_chart(fig2, use_container_width=True)



with st.expander("Inter-Class Variance w.r.t Position (using :blue[all] vectors)"):
    
    st.title('Inter-Class Variance w.r.t Position (using :blue[all] vectors)')
    pos_variance = torch.var(all_target_prototypes, dim = 0).reshape(50,1)
    pos_variance_np = pos_variance.numpy()

    fig3 = go.Figure([go.Bar(x=list(range(pos_variance_np.shape[0])), 
                            y=pos_variance_np.flatten(),
                            name='Variance')])  # use flatten to convert 2D array to 1D

    fig3.update_layout(title_text='Inter-Class Variance w.r.t Position',
                    xaxis_title='Position',
                    yaxis_title='Variance')

    st.plotly_chart(fig3,  use_container_width=True)
    
with st.expander(":blue[Moving] in the latent space"):
    
    st.sidebar.markdown("""---""")
    st.sidebar.subheader(":blue[Moving] in the latent space")
    pos = st.sidebar.slider('Position', 0, 49, 0)
    
    st.write(f"Position {pos}")
    
    
    for idx, (x,y) in enumerate(loader):
        z_x = encoder(x.float())
        y_z, sel_z_x = classifier(z_x)
        
        x_z = decoder(sel_z_x[..., None, None])
        break
    
    num_images = 10
    constants = torch.linspace(-1, 1, 11)
    st.write(sel_z_x.max(), sel_z_x.min())
    fig = make_subplots(rows=num_images, cols = constants.shape[0])

    # Add each image as a Heatmap to the subplots
    
    for i in range(num_images):
        for j, constant in enumerate(constants):
            img_mod = img + constant.item()  # Add the constant
            
            perturbed_sel_z_x = sel_z_x.clone()
            perturbed_sel_z_x[:, pos] = perturbed_sel_z_x[:, pos] + constant.item() # z_x is -> (32, 50), we add noise to just one pos across all images
            
            x_z = decoder(perturbed_sel_z_x[..., None, None])

            img = x_z[i].squeeze().detach().cpu().numpy()  # Remove the channel dimension and convert the image to numpy array
            
            fig.add_trace(
                go.Heatmap(z = img, zmin = 0, zmax = 1, colorscale='Viridis', showscale=False),
                row=i+1, col=j+1
            )
    # Set the overall height of the figure
    fig.update_layout(height=200*num_images)  # Adjust as needed

    st.plotly_chart(fig, use_container_width=True)
            
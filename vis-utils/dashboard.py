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
st.sidebar.markdown("""---""")
st.sidebar.subheader(":blue[Moving] in the latent space")
pos = st.sidebar.slider('Position', 0, 49, 0)

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
    
    st.write(torch.std(all_target_prototypes, dim = 0).reshape(50,1))
    st.write(torch.std(all_target_prototypes, dim = 0).reshape(50,1).max())

    fig3 = go.Figure([go.Bar(x=list(range(pos_variance_np.shape[0])), 
                            y=pos_variance_np.flatten(),
                            name='Variance')])  # use flatten to convert 2D array to 1D

    fig3.update_layout(title_text='Inter-Class Variance w.r.t Position',
                    xaxis_title='Position',
                    yaxis_title='Variance')

    st.plotly_chart(fig3,  use_container_width=True)
    
with st.expander("Inter-Class :green[pair] variance w.r.t. position"):
    
    st.title('Inter-Class :green[pair] variance w.r.t. position')
    selected_classes = st.multiselect('Select classes', [0,1,2,3], default=[0,1])
    
    pair_tensors = []

    for idx, (x,y) in enumerate(loader):

        z_x = encoder(x.float())
        y_z, sel_z_x = classifier(z_x)
        pY, uY = edl_probs(y_z.detach())
        
        
        pos_to_extract = y.argmax(dim = 1).unsqueeze(-1).unsqueeze(-1).expand(-1, 50, -1)
        
        target_prototypes = z_x.gather(dim = 2, index = pos_to_extract).detach().cpu()
        
        
        #append
        
        for i in selected_classes:
            
            #select only the target prototypes of the selected class
            selected_indices = y.argmax(dim = 1) == i
            
            considered = target_prototypes[selected_indices]
            
            #get the corresponding predictions of the selected class
            pred_indices = pY.argmax(dim = 1)[selected_indices]
            
            if(filter_only_correct):
                #filter only the correct predictions
                correct_indices = pred_indices == y.argmax(dim = 1)[selected_indices]
                pair_tensors.append(considered[correct_indices])
            else:
                pair_tensors.append(considered)
                
        if(idx == n_batches):
            break
    
    pair_tensors = torch.cat(pair_tensors, dim=0)
    
    pos_variance = torch.var(pair_tensors, dim = 0).reshape(50,1)
    pos_variance_np = pos_variance.numpy()
    
    st.write(torch.std(pair_tensors, dim = 0).reshape(50,1))
    st.write(torch.std(pair_tensors, dim = 0).reshape(50,1).max())

    fig3 = go.Figure([go.Bar(x=list(range(pos_variance_np.shape[0])), 
                            y=pos_variance_np.flatten(),
                            name='Variance')])  # use flatten to convert 2D array to 1D

    fig3.update_layout(title_text='Inter-Class Variance w.r.t Position',
                    xaxis_title='Position',
                    yaxis_title='Variance')

    st.plotly_chart(fig3,  use_container_width=True)
   
 
with st.expander(":blue[Move] from one bacteria to another"):
    st.header(":blue[Move] from one bacteria to another")
    b_1 = st.selectbox('Select bacteria 1', [0,1,2,3])
    
    rem_bac = [0,1,2,3]
    rem_bac.remove(b_1)
    b_2 = st.selectbox('Select bacteria 2', rem_bac)
    
    bac1_latents = 0
    bac2_latents = 0
    
    b1_labels = 0
    b2_labels = 0
    
    for idx, (x,y) in enumerate(loader):
        z_x = encoder(x.float())
        y_z, sel_z_x = classifier(z_x)
        pY, uY = edl_probs(y_z.detach())
        
        x_z = decoder(sel_z_x[..., None, None])
        
        #select user picked bacteria latent representations
        b1_indices = []
        b1_count = 0
        b2_indices = []
        b2_count = 0 
        
        for i in range(y.shape[0]):
            if(y.argmax(dim = 1)[i] == b_1 and pY.argmax(dim = 1)[i] == b_1):
                b1_indices.append(True)
                b1_count += 1
            else:
                b1_indices.append(False)
                
            if(y.argmax(dim = 1)[i] == b_2 and pY.argmax(dim = 1)[i] == b_2):
                b2_indices.append(True)
                b2_count += 1
            else:
                b2_indices.append(False)
                
        s_length = min(b1_count, b2_count)
        st.write(s_length)
        
        bac1_latents = sel_z_x[b1_indices][:s_length]
        bac2_latents = sel_z_x[b2_indices][:s_length]
        
        b1_labels = y[b1_indices][:s_length]
        b2_labels = y[b2_indices][:s_length]

        break
    
    diff = bac2_latents - bac1_latents
    
    num_images = diff.shape[0]
    sweep = 10
    
    header =  [] #[f"original class {b_1}"] + [f"Δ {i}" for i in range(sweep)] +  [f"mix {b_2}"] +  [f"original class {b_2}"]
    
    # Add each image as a Heatmap to the subplots
    for i in range(num_images):
        for j in range(sweep+3):
            
            if(j == 0):
                perturbed_sel_z_x = bac1_latents.clone()
                labels = b1_labels
            
            if(j > 0 and j < sweep+1):
                perturbed_sel_z_x[:, pos] = perturbed_sel_z_x[:, pos] + diff[:, pos]/sweep*j
            
            if(j == sweep+1):
                perturbed_sel_z_x = bac2_latents.clone()
                labels = b2_labels
                perturbed_sel_z_x[:, pos] = bac1_latents[:, pos] #only position pos is gotten from bacteria 1
            
            if(j == sweep+2):
                perturbed_sel_z_x = bac2_latents.clone()
                labels = b2_labels
            
            x_z = decoder(perturbed_sel_z_x[..., None, None])
            
            z_x_ = encoder(x_z)
            y_z_, _ = classifier(z_x_)
            pY_, uY_ = edl_probs(y_z_.detach())
            
            st.write(x_z.shape)
            
            pred_class = pY_.argmax(dim = 1)[i]
            gt_class = labels.argmax(dim = 1)[i]
            
            header.append(f"pred {pred_class} gt {gt_class}")

    fig_move_in_latent = make_subplots(rows=num_images, cols = sweep+3, subplot_titles = header)

    # Add each image as a Heatmap to the subplots
    for i in range(num_images):
        for j in range(sweep+3):
            
            if(j == 0):
                perturbed_sel_z_x = bac1_latents.clone()
            
            if(j > 0 and j < sweep+1):
                perturbed_sel_z_x[:, pos] = perturbed_sel_z_x[:, pos] + diff[:, pos]/sweep*j
            
            if(j == sweep+1):
                perturbed_sel_z_x = bac2_latents.clone()
                perturbed_sel_z_x[:, pos] = bac1_latents[:, pos] #only position pos is gotten from bacteria 1
            
            if(j == sweep+2):
                perturbed_sel_z_x = bac2_latents.clone()
            
            x_z = decoder(perturbed_sel_z_x[..., None, None])

            img = x_z[i].squeeze().detach().cpu().numpy()  # Remove the channel dimension and convert the image to numpy array
            
            fig_move_in_latent.add_trace(
                go.Heatmap(z = img, colorscale='Viridis', showscale=False),
                row=i+1, col=j+1
            )
    # Set the overall height of the figure
    fig_move_in_latent.update_layout(height=200*num_images)  # Adjust as needed

    st.plotly_chart(fig_move_in_latent, use_container_width=True)
    
    
    
    
with st.expander(":blue[Moving] in the latent space"):
    
    st.write(f"Position {pos}")
    
    
    for idx, (x,y) in enumerate(loader):
        z_x = encoder(x.float())
        y_z, sel_z_x = classifier(z_x)
        
        x_z = decoder(sel_z_x[..., None, None])
        break
    
    num_images = 10
    constants = torch.linspace(-0.5, 0.5, 7)
    st.write(sel_z_x.max(), sel_z_x.min())
    fig = make_subplots(rows=num_images, cols = constants.shape[0])

    # Add each image as a Heatmap to the subplots
    
    for i in range(num_images):
        for j, constant in enumerate(constants):
            
            perturbed_sel_z_x = sel_z_x.clone()
            perturbed_sel_z_x[:, pos] = perturbed_sel_z_x[:, pos] + constant.item() # z_x is -> (32, 50), we add noise to just one pos across all images
            
            x_z = decoder(perturbed_sel_z_x[..., None, None])

            img = x_z[i].squeeze().detach().cpu().numpy()  # Remove the channel dimension and convert the image to numpy array
            
            fig.add_trace(
                go.Heatmap(z = img, colorscale='Viridis', showscale=False),
                row=i+1, col=j+1
            )
    # Set the overall height of the figure
    fig.update_layout(height=200*num_images)  # Adjust as needed

    st.plotly_chart(fig, use_container_width=True)
            
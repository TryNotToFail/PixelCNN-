'''
This code is used to evaluate the classification accuracy of the trained model.
You should at least guarantee this code can run without any error on validation set.
And whether this code can run is the most important factor for grading.
We provide the remaining code, all you should do are, and you can't modify other code:
1. Replace the random classifier with your trained model.(line 69-72)
2. modify the get_label function to get the predicted label.(line 23-29)(just like Leetcode solutions, the args of the function can't be changed)

REQUIREMENTS:
- You should save your model to the path 'models/conditional_pixelcnn.pth'
- You should Print the accuracy of the model on validation set, when we evaluate your code, we will use test set to evaluate the accuracy
'''
from torchvision import datasets, transforms
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
import csv
NUM_CLASSES = len(my_bidict)

#TODO: Begin of your code
def get_label(model, model_input, device):
    # Write your code here, replace the random classifier with your trained model
    # and return the predicted label, which is a tensor of shape (batch_size,)
    B = model_input.size(0)
    predicted_labels = []
    for i in range(B):
        sample_i = model_input[i].unsqueeze(0)  # Shape: [1, C, H, W]
        losses = []
        for possible_class in range(NUM_CLASSES):
            labels_tensor = torch.tensor([possible_class], device=device, dtype=torch.long)
            # Run the model conditioned on the current possible class.
            output = model(sample_i, labels_tensor)
            # Compute the loss for this sample and class.
            loss_val = discretized_mix_logistic_loss(sample_i, output)
            losses.append(loss_val.item())
        # The predicted class is the one with the minimum loss for this sample.
        predicted_labels.append(np.argmin(losses))
    return torch.tensor(predicted_labels, device=device, dtype=torch.long)
# End of your code

def classifier(model, data_loader, device):
    model.eval()
    acc_tracker = ratio_tracker()
    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, categories, _ = item
        model_input = model_input.to(device)
        original_label = [my_bidict[item] for item in categories]
        original_label = torch.tensor(original_label, dtype=torch.int64).to(device)
        answer = get_label(model, model_input, device)
        
        # Print the ground truth and predicted labels for this batch.
        print("Batch", batch_idx)
        print("Ground truth:", original_label.cpu().numpy())
        print("Predicted:   ", answer.cpu().numpy())
        
        correct_num = torch.sum(answer == original_label)
        acc_tracker.update(correct_num.item(), model_input.shape[0])
    
    return acc_tracker.get_ratio()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str,
                        default='validation', help='Mode for the dataset')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':False}

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                            mode = args.mode, 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             **kwargs)

    #TODO:Begin of your code
    #You should replace the random classifier with your trained model
    model = PixelCNN(nr_resnet=3, nr_filters=80, input_channels=3, nr_logistic_mix=15)
    #End of your code
    
    model = model.to(device)
    #Attention: the path of the model is fixed to './models/conditional_pixelcnn.pth'
    # conditional_pixelcnn_big, filter = 100, nr_resnet = 1, nr_logistic_mix = 10 # Accuracy 25.25%
    # pcnn_cpen455_load_model_499_condition_batch_16_resnet_2_filter_80_mix_15 # Accuracy 25.04%
    #You should save your model to this path
    model_path = os.path.join(os.path.dirname(__file__), 'models/pcnn_cpen455_from_scratch_469.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print('model parameters loaded')
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.eval()
    
    acc = classifier(model = model, data_loader = dataloader, device = device)
    print(f"Accuracy: {acc}")
        
import torch.nn as nn
import torch.nn.functional as F
from layers import *

class PixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=0)
                                            for _ in range(nr_resnet)])

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=1)
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul):
        u_list, ul_list = [], []

        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u)
            ul = self.ul_stream[i](ul, a=u)
            u_list  += [u]
            ul_list += [ul]

        return u_list, ul_list


class PixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream  = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=1)
                                            for _ in range(nr_resnet)])

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=2)
                                            for _ in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list):
        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u, a=u_list.pop())
            ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1))

        return u, ul

class AbsolutePositionalEncoding(nn.Module):
    # Taken from PA2, absolute positional encoding class to fuse class labels here
    def __init__(self, num_classes, d_model):
        super().__init__()
        self.W = nn.Parameter(torch.empty((num_classes, d_model)))
        nn.init.normal_(self.W)

    def forward(self, x):
        """
        args:
            x: shape B x D
        returns:
            out: shape B x D
        START BLOCK
        """
        B, D = x.shape
        out = torch.add(x, self.W[:B, :D].to(x.device))
        """
        END BLOCK
        """
        return out

class PixelCNN(nn.Module):
    MAX_LEN = 256
    num_classes = 4
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
                    resnet_nonlinearity='concat_elu', input_channels=3):
        super(PixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu' :
            self.resnet_nonlinearity = lambda x : concat_elu(x)
        else :
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad  = nn.ZeroPad2d((0, 0, 1, 0))
    
        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters,
                                                self.resnet_nonlinearity) for i in range(3)])

        self.up_layers   = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters,
                                                self.resnet_nonlinearity) for _ in range(3)])

        self.downsize_u_stream  = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters,
                                                    stride=(2,2)) for _ in range(2)])

        self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(nr_filters,
                                                    nr_filters, stride=(2,2)) for _ in range(2)])

        self.upsize_u_stream  = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters,
                                                    stride=(2,2)) for _ in range(2)])

        self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(nr_filters,
                                                    nr_filters, stride=(2,2)) for _ in range(2)])

        self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2,3),
                        shift_output_down=True)

        self.ul_init = nn.ModuleList([down_shifted_conv2d(input_channels + 1, nr_filters,
                                            filter_size=(1,3), shift_output_down=True),
                                       down_right_shifted_conv2d(input_channels + 1, nr_filters,
                                            filter_size=(2,1), shift_output_right=True)])

        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None
        
        self.pos_encoding = AbsolutePositionalEncoding(self.MAX_LEN, self.nr_filters)
        # Embedding shape has to account for 4 classes and because we do middle fusion it should match the shape of
        self.pos_embedding = nn.Embedding(self.num_classes, self.nr_filters)

    def forward(self, x, class_labels, sample=False):  
        if self.init_padding is not sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.cuda() if x.is_cuda else padding.to(x.device)

        if sample :
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            padding = padding.cuda() if x.is_cuda else padding.to(x.device)
            x = torch.cat((x, padding), 1)
        
        # # ====== Apply Class Conditioning ======
        # # Compute class embedding; expecting embedding_dim == nr_filters.
        # batch_size = x.size(0)
        # class_embedding = self.embedding(class_labels)
        # class_embedding = class_embedding.view(batch_size, -1, 1, 1)
        # pos_embedding = self.pos_embedding(x)
        
        # # ====== Compute Initial Features ======
        # u0 = self.u_init(x) + pos_embedding
        # ul0 = self.ul_init[0](x) + self.ul_init[1](x) + pos_embedding
        
        # # Add the conditioning embedding to the initial feature maps.
        # u0 = u0 + class_embedding
        # ul0 = ul0 + class_embedding

        # # # Start the upward pass with the conditioned features.
        # u_list  = [u0]
        # ul_list = [ul0]
        
        ###      UP PASS    ###
        x = x if sample else torch.cat((x, self.init_padding), 1)
        u_list  = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        for i in range(3):
            # resnet block
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1])
            u_list  += u_out
            ul_list += ul_out

            if i != 2:
                # downscale (only twice)
                u_list  += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]
        
        # # Flatten the class_embedding to [B, nr_filters].
        # flat_emb = class_embedding.view(batch_size, -1)
        # gamma = self.film_gamma(flat_emb).view(batch_size, self.nr_filters, 1, 1)
        # beta  = self.film_beta(flat_emb).view(batch_size, self.nr_filters, 1, 1)
        # # Modulate the top features from the upward pass.
        # u_list[-1] = u_list[-1] * gamma + beta
        # ul_list[-1] = ul_list[-1] * gamma + beta
        
        label_embeddings = self.pos_embedding(class_labels.to(x.device)).unsqueeze(-1).unsqueeze(-1)
        # print(f"SHAPE OF LE: {label_embeddings.shape}")
        self.add_embedding_to_u_ul(u_list, label_embeddings)
        self.add_embedding_to_u_ul(ul_list, label_embeddings)
        
        ###    DOWN PASS    ###
        u  = u_list.pop()
        ul = ul_list.pop()

        for i in range(3):
            # resnet block
            u, ul = self.down_layers[i](u, ul, u_list, ul_list)

            # upscale (only twice)
            if i != 2 :
                u  = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

        x_out = self.nin_out(F.elu(ul))

        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return x_out
    
    def add_embedding_to_u_ul(self, tensor_list, embedding_tensor):
        # print(f"SHAPE OF u: {tensor_list[0].shape}")
        # print(f"SHAPE OF EMB: {embedding_tensor.shape}")
        for i in range(len(tensor_list)):
            # Add the embedding tensor to the current tensor in-place
            tensor_list[i] += embedding_tensor
    
    
class random_classifier(nn.Module):
    def __init__(self, NUM_CLASSES):
        super(random_classifier, self).__init__()
        self.NUM_CLASSES = NUM_CLASSES
        self.fc = nn.Linear(3, NUM_CLASSES)
        print("Random classifier initialized")
        # create a folder
        if os.path.join(os.path.dirname(__file__), 'models') not in os.listdir():
            os.mkdir(os.path.join(os.path.dirname(__file__), 'models'))
        torch.save(self.state_dict(), os.path.join(os.path.dirname(__file__), 'models/conditional_pixelcnn.pth'))
    def forward(self, x, device):
        return torch.randint(0, self.NUM_CLASSES, (x.shape[0],)).to(device)
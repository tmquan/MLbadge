A boilerplate for image-to-image translation 
***************************

In general, a typical training  has the following format:

.. code-block:: python

    from __future__ import absolute_import
    from __future__ import division
    from __future__ import print_function

    import os
    import sys
    import shutil
    import logging
    import argparse

    #----------------------------------------------------------------------------------
    import cv2
    import numpy as np 

    #----------------------------------------------------------------------------------
    # Using torch
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.nn.utils.weight_norm as weightNorm
    import torch.optim as optim
    from torch.autograd import Variable
    from torch.utils.tensorboard import SummaryWriter
    import torchvision

    #----------------------------------------------------------------------------------
    # An efficient dataflow loading for training and testing
    import tensorpack.dataflow as df
    import tqdm

    #----------------------------------------------------------------------------------
    #
    # Global configuration
    #
    BATCH = 32
    EPOCH = 500
    SHAPE = 256
    NF = 64

    #----------------------------------------------------------------------------------
    #
    # Create the data flow using tensorpack dataflow (independent from tf and pytorch)
    #
    # TODO

    #----------------------------------------------------------------------------------
    #
    # Create the model
    #
    # TODO

    #----------------------------------------------------------------------------------
    #
    # Perform sample
    #
    # TODO


    #----------------------------------------------------------------------------------
    #
    # Main
    #
    if __name__ == '__main__':
        #------------------------------------------------------------------------------
        #
        parser = argparse.ArgumentParser()
        parser.add_argument('--data', help='the image directory')
        parser.add_argument('--load', help='load model')
        parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
        parser.add_argument('--sample', action='store_true', help='run inference')
        args = parser.parse_args()
        
        #------------------------------------------------------------------------------
        # Choose the GPU
        if args.gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
            
        if args.sample:
            #------------------------------------------------------------------------------
            # TODO: Run the inference
            pass
        else   
            #------------------------------------------------------------------------------
            # Initialize the program
            writer = SummaryWriter()
            use_cuda = torch.cuda.is_available()
            xpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            step = 0

            #------------------------------------------------------------------------------
            # TODO
            net = CustomNetwork()
            optimizer = optim.Adam(net.parameters(), lr=3e-6)
            criterion = nn.L1Loss()
        
            #
            # Train from scratch or load the pretrained network
            #
            # TODO: Load the pretrained model
            if args.load:
                pass
                

            # Create a dataflow of training and validation
            # TODO
            ds_train = CustomDataflow(size=100, datadir=args.data) 
            augs = [
                    # imgaug.ResizeShortestEdge(270),
                    imgaug.RandomCrop(SHAPE), 
                    imgaug.Flip(horiz=True), 
                    imgaug.Flip(vert=True), 
                    imgaug.Transpose()
                    ]
            ds_train = AugmentImageComponents(ds_train, augs, (0, 1))
            ds_train = MapData(ds_train, lambda dp: [np.expand_dims(dp[0], axis=0), 
                                                     np.expand_dims(dp[1], axis=0), 
                                                     ])
            ds_train = df.BatchData(ds_train, batch_size=BATCH)
            ds_train = df.PrintData(ds_train)
            # ds_train = df.PrefetchDataZMQ(ds_train, nr_proc=4)

            ds_valid= CustomDataflow(size=100, datadir=args.data)

            #
            # Training loop
            #
            max_step = 10000000
            for epoch in range(EPOCH):
                for mb_train in ds_train.get_data():
                    step = step+1
                    if step > max_step:
                        exit()
                    # print("Step: {}, Epoch {}".format(step, epoch))

                    image = torch.tensor(mb_train[0]).float()
                    label = torch.tensor(mb_train[1]).float()
           
                    net = net.to(xpu)
                    image = image.to(xpu)
                    label = label.to(xpu)

                    # TODO: Forward pass
                    estim = net(image)

                    # Reset the optimizer
                    optimizer.zero_grad()

                    # TODO: Loss calculation
                    loss = criterion(estim, label)
                    loss.backward()
                    optimizer.step()
                    
                    # TODO: Log to tensorboard after n steps
                    writer.add_scalar('train/loss', loss.item(), step)     
                    writer.add_image('train/estim', torch.cat([image, label, estim], 3)[0][0], step, dataformats='HW')
                    

                    # TODO: Valid set after n steps, need to implement as callback
                    if step % 100 == 0:
                        net.eval()
                        pass
                   
                    # TODO: Log to console after n steps, need to implement as callback
                    if True:
                        print('\rStep {} \tLoss: {:.4f}'.format(step, loss.item()), end="")   
                        pass


                    # Customization on learning rate
                    # TODO: Lowering the learning rate after n steps
                    if step < 200000:
                        lr = 1e-4
                    elif step < 400000:
                        lr = 1e-5
                    else:
                        lr = 1e-6
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr
      
                    # TODO: Save the model after n steps, need to implement as callback
                    if step % 10000 == 0:
                        print('\rStep {} \tLoss: {:.4f}'.format(step, loss.item()))
                        torch.save(net.cpu().state_dict(), "driver_snemi.pkl")
                        net = net.to(xpu)


# Streamlined Attention Module
This work was developed in order to propose a model for the task of fine-grained visual recognition under computer vision. This was implemented on CIFAR10, CIFAR100 and FGVC-Aircrafts dataset, where we could prove considerable gain in results for CIFAR10 &amp; CIFAR100 datasets and state-of-the-art results for FGVC-Aircrafts dataset with 99.7%. 

## Attention and its advancements
Attention mechanism in computer vision have been emerging vastly in application perspective. Even though this mechanism was introduced in a work inclined towards natural language processing, researchers were successful enough in inculcating the same in the vision perception as well. This mechanism can be interpreted by taking the prime focus on a context under consideration. In terms of vision, whenever you come across a scenario you get to focus on a certain partial of the whole scene in that scenario and paying attention to that particular partial is what your attention is grabbed towards. 

### Global and Local attention
This mechanism can be approached in two different ways i.e., by considering the global and local attention. Global attention deals with the attention attained from a whole scene where in the attention of the whole scene is captured and the aligned weights and features with respect to the whole are averaged. This mean of weights is considered to be offering the global attention. Global attention can also be interpreted as soft attention, this is moreover considered to be low performing due to its tendency of taking attentive weighted features from the whole image patch. Later, researchers found that the results of attention attained from global attention were inaccurate due to the reason being more complex and computationally expensive.  
                                  Considering this factor, local attention have been introduced depicting the mechanism as it captures the attention over the sub-patches of the portrayed image i.e., learns those weights which are aligned with attentive sub-patches of the considered image. This is depicted as a combination of soft and hard attention which not only proves to capture more attention but also perform better than that of global attention.
                                  ![1_lES94I6Z5jLH2ZsSRpUANg](https://user-images.githubusercontent.com/67636257/149650420-a7b5ec51-eb82-4713-95b7-382ae4af7c70.png)

### Spatial and Channel attention                                 
Similarly, spatial and channel attention are considered to be the recent time adavancements dealing with the attention across the channels obtained via convolutional mechanism which is considered in a general neural network. Here, spatial attention can be interpreted as in the space based encapsulation of the features obtained from a specific spatial channel i.e., the overall refinement of the presented feature map (spatially) for better representation of the succeeding feature maps. Channel attention is something which deals with the attention gain of a specific channel from the group of channels obtained from the corresponding input in a neural network.
In a way we can interpret the combination of both channel and spatial attention as block based module, where in channel attention, the most attentive channel/ feature map is captured and the spatial attention is obtained from the selected feature map proceeded from the channel attention. This helps in gaining a greater level of attention based incentive due to which the produced feature maps are considered to be performing more efficiently.   


### Motivation
The proposed works under this tasks has motivated me to work for a proposal of a novel module. Moreover, the implementations of these models were inclined towards image classfication task on the traditional datasets like CIFAR10 & CIFAR100. Hence, taking this aspect under consideration, the proposed module has been is designed in such way that it not only supports in increasing efficiency of the image classification task whereas performs more accurately for the task of fine-grained visual recognition.  

### Model 
The proposed model is described as Streamlined attention module. This is built 
<p align="center">
<img src="https://user-images.githubusercontent.com/67636257/149652989-4a1389df-da08-4e91-b416-3d6d64d6b7cc.png" width="500" height="400">
</p>

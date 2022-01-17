# Streamlined Attention Module
This work was developed in order to propose a model for the task of fine-grained visual recognition under computer vision. This was implemented on CIFAR10, CIFAR100 and FGVC-Aircrafts dataset, where we could prove considerable gain in results for CIFAR10 &amp; CIFAR100 datasets and state-of-the-art results for FGVC-Aircrafts dataset with 99.7%. 

## Attention and its advancements
Attention mechanism in computer vision have been emerging vastly in application perspective. Even though this mechanism was introduced in a work inclined towards natural language processing, researchers were successful enough in inculcating the same in the vision perception as well. This mechanism can be interpreted by taking the prime focus on a context under consideration. In terms of vision, whenever you come across a scenario you get to focus on a certain partial of the whole scene in that scenario and paying attention to that particular partial is what your _attention_ is grabbed towards. 

### Global and Local attention
This mechanism can be approached in two different ways i.e., by considering the global and local attention. Global attention deals with the attention attained from a whole scene where in the attention of the whole scene is captured and the aligned weights and features with respect to the whole are averaged. This mean of weights is considered to be offering the global attention. Global attention can also be interpreted as soft attention, this is moreover considered to be low performing due to its tendency of taking attentive weighted features from the whole image patch. Later, researchers found that the results of attention attained from global attention were inaccurate due to the reason being more complex and computationally expensive.  
                                  Considering this factor, local attention have been introduced depicting the mechanism as it captures the attention over the sub-patches of the portrayed image i.e., learns those weights which are aligned with attentive sub-patches of the considered image. This is depicted as a combination of soft and hard attention which not only proves to capture more attention but also perform better than that of global attention.
<p align="center">
<img src="https://user-images.githubusercontent.com/67636257/149650420-a7b5ec51-eb82-4713-95b7-382ae4af7c70.png" width="600" height="300">
</p>

### Spatial and Channel attention                                 
Similarly, spatial and channel attention are considered to be the recent time adavancements dealing with the attention across the channels obtained via convolutional mechanism which is considered in a general neural network. Here, spatial attention can be interpreted as in the space based encapsulation of the features obtained from a specific spatial channel i.e., the overall refinement of the presented feature map (spatially) for better representation of the succeeding feature maps. Channel attention is something which deals with the attention gain of a specific channel from the group of channels obtained from the corresponding input in a neural network.
In a way we can interpret the combination of both channel and spatial attention as block based module, where in channel attention, the most attentive channel/ feature map is captured and the spatial attention is obtained from the selected feature map proceeded from the channel attention. This helps in gaining a greater level of attention based incentive due to which the produced feature maps are considered to be performing more efficiently.   


## Motivation
The proposed works under this tasks has motivated me to work for a proposal of a novel module. Moreover, the implementations of these models were inclined towards image classfication task on the traditional datasets like CIFAR10 & CIFAR100. Hence, taking this aspect under consideration, the proposed module has been is designed in such way that it not only supports in increasing efficiency of the image classification task whereas performs more accurately for the task of fine-grained visual recognition.  

## Model and its architecture
The proposed model is described as Streamlined attention module due to its tendency of proving more accuracy with less number of additional layers. This model was built under a task specification of fine-grained visual recognition and image classification. The utilized encoder for this model was Residual neural network i.e., ResNet. The basic convention of this residual neural network is to process the given input via a skip connection based mechanism where in the intial hypothesis of the work stood to be that this mechanism would help in proving better longer range denpendency based features which also helps in eliminating the concept of proceeding for depth in number of convolutions.
<p align="center">
<img src="https://user-images.githubusercontent.com/67636257/149652989-4a1389df-da08-4e91-b416-3d6d64d6b7cc.png" width="500" height="400">
</p>
The traditional ResNet have been proved in the above figure illustrated. In the proposed model, through an applied research approach, there has been a proposal of few additional layers right before the skip connection concatenates with the latest convolution layer, as illustrated above. The value added by these addiional layers have been described below.

### Architecture and methodology
The proposed module consists of 2 layers with an additional functional mechanism. The 2 additional layers are GAP-global average pooling layer and FC/Linear layer attached one after the other. After this layer, the illustrated functional mechanism deals with second order pooling which results in processing an element-wise multiplication operation  producing higher order features due to its tendency of collecting the additional valuable information through an outer-product mathematical operation as a function. Whenever the higher order features are captured they have the tendency to be more effective their capability of enriching the non-linear modelling. The below presented image illustrates the proposed module in detail. 
<p align="center">
<img src="https://user-images.githubusercontent.com/67636257/149658770-6761d894-7acc-4e4e-997a-1e272f34fd8b.png" width="450" height="250">
</p>

The reason for allotting GAP layer right after the previous convolution is because global average pooling layer has the efficienvy of providing convienent feature refinement through its procedure of pooling the weights globally and applying the mean operation, bringing down to an efficient 1D-convolution. This layer is followed by a FC/Linear layer which deals with compression of the produced 1D convolution from its preceding GAP layer. This ensures to provide channel attention. Further, the feature obtained from the linear layer go through an element-wise pooling operation with the features extracted from the initial input. This process ensures to provide spatial attention as this operation deals with the production of covariance matrices embedded with essential attentive features. This overall module is injected right before teh skip connection in the traditional ResNet architecture as illutrated in the below figure as embedded feature vector **_f<sub>v</sub>_**.


<p align="center">
<img src="https://user-images.githubusercontent.com/67636257/149658786-f7054a32-141b-4b88-bfc6-ffdde04dbc72.png" width="650" height="275">
</p>

Training of this model has been done wisely considering multiple factors. As the proposed model is inclined likely towards production of truthful results under the fine-grained visual recognition task i.e., pertained to use FGVC-Aircrafts dataset (dataset description presented below), an image size od 192x192 had been set with a batch size of 128 and while considering image classification for CIFAR-10 and CIFAR-100, an image size of 64x64 has been considered with a consistent number of epochs to be 45 and early stopping patience preset to 12 likely reproduced iterations.

## Training parameters
**Optimizer** - Adam with learning rate of 10<sup>-4</sup><br/>
**Loss Function** - Categorical Cross Entropy<br/>
**Early Stopping** - Patience 12<br/>
**Number of Epochs** - 45<br/>

## Dataset Description
The model was trained and tested on CIFAR-10, CIFAR-100 & FGVC-Aircrafts dataset.
| Datasets  | No.of classes | Train Samples | Test Samples |
| ------------- | ------------- | ------------- | ------------- |
| CIFAR-10  | 10  | 50k  | 10k  |
| CIFAR-100  | 100  | 50k  | 10k |
| FGVC-Aircrafts  | 100 | 6.6k  | 3.3k  |


## Training requirements
This model was trained on NVIDIA RTX A4000 16GB vRAM GPU.<br/>

Python >= *3.0*<br/>
PyTorch Version >= *0.4.0*<br/>
torchvision >= *0.2.1*<br/>
NumPy <br/>
Matplotlib<br/>
Pandas<br/>
Math<br/>
PIL for Image processing for CAM visualization<br/>
Model_zoo for pipelining weights of the required encoder<br/>

## Results
After an 10x iterative procedure of training the model, the obtained results were efficient compared to that of the previously exisiting models out there. The module was inculcated with pretrained ResNet-50, ResNet-101 and ResNet-152 on Imagenet. The obtained results under each task with respect to each dataset are illustrated below for the model encoder ResNet-50.

| Datasets  | Task |Accuracy | 
| ------------- | ------------- | ------------- |
| CIFAR-10  | Image Classification | 89.45  | 
| CIFAR-100  | Image Classification | 70.19 |
| FGVC-Aircrafts  | Fine-grained Visual Recognition | **97.27** | 

The FGVC-Aircrafts dataset obtained state-of-the-art results with an accuracy of **97.27%**.<br/> Similarly, the model outperformed other existing models under datasets of CIFAR-10 & CIFAR-100.<br/>
The visualizations for the obtained results have been proved through Class activation maps as illustrated below. These maps prove the features on the given image patch through a heatmap perception i.e., the most concentrated, here, the highly attentive region in an image is visualized (as red) to be most attentive and the image partial with less intensity in attention are seen to be less concentrated. As the features are also considered to the outputs of the activations generated in a neural network pertaining to predict a required class in a presented image, the mechanism is named after "class activation" maps.

<p align='center'>
<img src="https://user-images.githubusercontent.com/67636257/149658646-b4bb350a-bcbf-406c-974d-f63007e18cff.png" height="600">
</p>

## Future Scope
There is a hypothesis that this model would equally work well on Medical imagery as well. This hypothesis has been taken under consideration and the experimentation has begun. The work might be proved as soon as the expected is obtained.


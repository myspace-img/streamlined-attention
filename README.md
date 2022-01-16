# Streamlined Attention Module
This work was developed in order to propose a model for the task of fine-grained visual recognition under computer vision. This was implemented on CIFAR10, CIFAR100 and FGVC-Aircrafts dataset, where we could prove considerable gain in results for CIFAR10 &amp; CIFAR100 datasets and state-of-the-art results for FGVC-Aircrafts dataset with 99.7%. 

## Attention and its advancements
Attention mechanism in computer vision have been emerging vastly in application perspective. Even though this mechanism was introduced in a work inclined towards natural language processing, researchers were successful enough in inculcating the same in the vision perception as well. This mechanism can be interpreted by taking the prime focus on a context under consideration. In terms of vision, whenever you come across a scenario you get to focus on a certain partial of the whole scene in that scenario and paying attention to that particular partial is what your attention is grabbed towards. 

### Global and local attention
This mechanism can be approached in two different ways i.e., by considering the global and local attention. Global attention deals with the attention attained from a whole scene where in the attention of the whole scene is captured and the aligned weights and features with respect to the whole are averaged. This mean of weights is considered to be offering the global attention. Global attention can also be interpreted as soft attention, this is moreover considered to be low performing due to its tendency of taking attentive weighted features from the whole image patch. Later, researchers found that the results of attention attained from global attention were inaccurate due to the reason being more complex and computationally expensive.  
                                  Considering this factor, local attention have been introduced depicting the mechanism as it captures the attention over the sub-patches of the portrayed image i.e., learns those weights which are aligned with attentive sub-patches of the considered image. This is depicted as a combination of soft and hard attention which not only proves to capture more attention but also perform better than that of global attention.
                                  
 

### Spatial and Channel Attention                                 
Similarly, spatial and channel attention are considered to be the recent time adavancements dealing with the attention across the channels obtained via convolutional mechanism which is considered in a general neural network. 

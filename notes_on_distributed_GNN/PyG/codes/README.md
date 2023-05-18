### Combine PyG with Orca

- prepare conda environment
Follow [here](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#quick-start) and install CPU version of `PyG`.
![image](https://user-images.githubusercontent.com/42887453/225354550-25014f5e-7b37-4bb0-b5e4-f60228813f34.png)

command:
```
conda create -n GNN_env python=3.7
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
pip install --pre --upgrade bigdl-orca[ray] 
pip install torch==1.13.1 torchvision torchmetrics==0.10.0 tqdm pandas pyarrow
```

- train GNN on Orca:
python command:
```
python GNN_train.py
```

- results:

```
num_samples: 100                                                                                                        
epoch: 8.0                                                                                                              
batch_count: 4.0                                                                                                        
train_loss: 0.7794397179037332                                                                                          
last_train_loss: 0.017204800620675087                                                                                   
val_accuracy: 0.7599999904632568                                                                                        
val_loss: 0.638522135913372                                                                                             
val_num_samples: 100.0     
                                                                                                                                                                                                                     
num_samples: 100                                                                                                        
epoch: 9.0                                                                                                              
batch_count: 4.0                                                                                                        
train_loss: 0.6173351418972015                                                                                          
last_train_loss: 0.03387674689292908                                                                                    
val_accuracy: 0.8199999928474426                                                                                        
val_loss: 0.49365023508667943                                                                                           
val_num_samples: 100.0       
                                                                                                                                                                                                                   
num_samples: 100                                                                                                        
epoch: 10.0                                                                                                             
batch_count: 4.0                                                                                                        
train_loss: 0.48190005093812943                                                                                         
last_train_loss: 0.027375169098377228                                                                                   
val_accuracy: 0.8700000047683716                                                                                        
val_loss: 0.39040984891355035                                                                                           
val_num_samples: 100.0                                                                                                                                                                                                                          

(PytorchRayWorker pid=25618) [2023-03-15 23:06:43] INFO     
Finished training epoch 10, stats on rank 0: {'epoch': 10, 'batch_count': 4, 'num_samples': 100, 'train_loss': 0.48190005093812943, 'last_train_loss': 0.027375169098377228, 'val_accuracy': tensor(0.8700), 'val_loss': 0.39040984891355035, 'val_num_samples': 100}                                       

Evaluation results:                                                                                                     
num_samples: 100                                                                                                        
Accuracy: 0.8700000047683716                                                                                            
val_loss: 0.39040985584259036 
```


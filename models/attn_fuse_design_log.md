10/7/2018 11:48 AM: tried simple layer: only one hidden layer = 32 units, learning rate = 0.001, after around 100 epochs, training loss converge around 4.2730, validation acc converge around 0.125.

10/7/2018 11:52 AM: tried more hidden layers: [32,64,128], learnig rate = 0.001, after around 100 epochs, training loss converge around 4.2730, validation acc converge around 0.125

10/7/2018 12:21 PM: add batch norm for fc1, fc2, fc3 and attn_weight, loss becomes very large, after 20 epochs, terminated. loss fluctuates around 60000.00 and validation acc oscillate between 0.05 to 0.09.

10/7/2018 12:28 PM: remove batch norm after attn_weight & add softmax to self.out to make self.out in [0,1]. loss returns to normal, reaches 5.025 after 20 epochs, validation acc improved to 0.133. after 50 epochs loss converges to 5.025, validation acc converges to 0.135.

10/7/2018 1:41 PM: change all fc layers' bias initializers to tf.contrib.layers.xavier_initializer(). 100 epochs. training loss converges to 5.023, validation acc converges to 0.139.

10/7/2018 3:21 PM: since output dimension is 172, hidden layer better not to smaller than 172. so increase all hidden layer units to 256, that is hidden layers = [256,256,256]. 50 epochs. train loss 5.024. validation acc 0.14.

10/7/2018 4:58 PM: train loss drops very slow. try increase learning rate to 0.01. train loss increase dramatically after a few epochs.

10/7/2018 5:22 PM: suspect lr too large previously. now learning rate = 0.0001. no significant improvement.

10/7/2018 5:22 PM: replace the matmul operation with tf.nn.2dconv, set the kernel as the output of the attn mechanism. train loss no change after several iterations. acc no change... what's wrong???

10/7/2018 9:01 PM: change learning rate after i saw train loss cannot drop any more. change between 0.001 and 0.0001. after 125 epochs, train loss is still slowly droping, but it seems the validation acc reaches to 0.137

10/8/2018 12:47 AM: epoch 133 train loss suddenly drops to 5-, validation acc increase to 0.17+. what happened??????

10/8/2018 1:20 AM: epoch 161, train loss ossilates, decrease learning rate to 0.00001.

10/8/2018 10:36 AM: epoch 545, train loss 5.0081, validation acc 0.1358. 

10/8/2018 10:36 AM: no improvement on validation acc. go back to retrain with lr=1e-4. capture best perform: validation acc = 0.17

10/8/2018 1:13 PM: try normalize x_base before feed into model (in data generator). "nan" in loss. val acc around 0.4.

10/8/2018 2:37 PM: try replace activation fn relu with tanh. val acc in first a few epoch only 0.045. it means tanh initially is good at predicting the most likely but good at prediction less likly.

10/8/2018 2:50 PM: change conv_kernel's activation fn to sigmoid because it may be bad to give negative weight to any recommender.

10/8/2018 2:50 PM: acc loss back to normal. conclusion is that bactch norm and tanh will set the outweight to negative, negative will badly affect acc.

10/8/2018 6:16 PM: epoch 148. train loss 4.9819. val acc 0.1822.

10/8/2018 9:23 PM: epoch 275. train loss 4.9792. val acc 0.1826.

10/9/2018 10:44 PM: acc 18.26 cannot persist in different runs of the experiments. every time I get different acc all smaller than this. now change model to double_attn_fuse_model.

10/10/2018 11:44 PM: double_attn_fuse_model doesn't give much improvement. Surprisingly, normalize the data give us a lot of improvement. 

10/10/2018 11:44 PM: two models are tested. attn_train and double_attn_train. attn_train converges fast. after 735 epochs, results are as follow:
                    
                    Epoch-735  loss:4.03917925 -- acc:0.1830 -- lr:0.00000010 -- exp_name:attn_train
                    Val-735  loss:4.02525157 -- acc:0.1896 -- acc2:0.3079 -- acc3:0.3928 -- acc4:0.4552 -- acc5:0.5081 -- acc6:0.5508 -- acc7:0.5844 -- acc8:0.6138 -- acc9:0.6400 -- acc10:0.6634

                    attn_train : I can see the training loss cannot drop any more but val loss is still dropping slowly.
                    double_attn_train : train loss still dropping. wait for more time to see its results.

                    From the train process, i see time-decaying learning rate is necessary, the intial time is too slow while after time is to fast that it cannot converge.

                    Epoch-1273  loss:4.03053073 -- acc:0.1838 -- lr:0.00000010 -- exp_name:double_attn_train
                    Val-1273  loss:4.02300444 -- acc:0.1893 -- acc2:0.3068 -- acc3:0.3908 -- acc4:0.4540 -- acc5:0.5072 -- acc6:0.5495 -- acc7:0.5839 -- acc8:0.6139 -- acc9:0.6407 -- acc10:0.6639

                    double_attn_train: train loss converges. val loss is still dropping. there is almost no difference between these two model.



10/15/2018 2:01 PM: now the major questions are 1. why double attn is worse than the single attn?
												2. why adding title and nsw makes it worse??

												Epoch-1499  loss:3.98986938 -- acc:0.1813 -- lr:0.00000001 -- exp_name:double_attn_sw_title_train

												Val-1499  loss:3.99277245 -- acc:0.1863 -- acc2:0.3032 -- acc3:0.3873 -- acc4:0.4513 -- acc5:0.5030
												-- acc6:0.5443 -- acc7:0.5796 -- acc8:0.6088 -- acc9:0.6352 -- acc10:0.6584
												-- acc20:0.8140 -- acc30:0.8908 -- acc40:0.9338 -- acc50:0.9588

												Epoch-1702  loss:3.99739943 -- acc:0.1800 -- lr:0.00000001 -- exp_name:attn_sw_title_train

												Val-1702  loss:3.98842379 -- acc:0.1864 -- acc2:0.3048 -- acc3:0.3886 -- acc4:0.4530 -- acc5:0.5033
												-- acc6:0.5443 -- acc7:0.5786 -- acc8:0.6067 -- acc9:0.6316 -- acc10:0.6541
												-- acc20:0.8052 -- acc30:0.8872 -- acc40:0.9321 -- acc50:0.9578

10/15/2018 3:26 PM: try normalize nsw in attn_sw_title. try softmax the "out" in double_attn_sw_title.

10/15/2018 5:43 PM: normalize nsw not too much improve. softmax makes the topk acc improve but total train loss increase to 5.xx now try add softmax to channel mask and out and then the sum of the to. So 3 softmax added.

10/15/2018 6:16 PM: it seems to be softmax cannot be used to many times. and both base * conv and channel_mask should be sigmoid. 

10/15/2018 9:16 PM: reduce both batch size to 1. no improve.

10/15/2018 10:41 PM: keep title features but cut nsw feature. no improvement and all the same as the situation with nsw feature. acc1 seems to have upperbound 18.64. make me wonder title features is useless or have negtive effect.

10/16/2018 1:18 AM: title is proved to be the to bottle neck. the acc1 returns 18.9x when not using title.

10/16/2018 1:18 AM: but acc1 seems have upper bound 18.98? why??? thought nsw is dominated by hour but after seperate these two still upper bounded by 18.98.

10/16/2018 1:34 AM: change conv_kernel to tahn activ, train acc drop much to accc1=17.43, but val acc is still upper bounded by 18.96. ???

10/16/2018 1:41 AM: add tanh with title... no improve.

10/16/2018 5:32 PM: wonder attn really works, now i just add a conv2d to x_base and concatenate x_base x-hour and x_nsw then feed to fc...

10/16/2018 5:32 PM: suprise. inputall get very low train loss (4.96 at epoch 6) and high train acc: 20.16 at epoch 6. but at the beginning the val acc is not high (acc1<17 at epoch 7)

__________________________________________________________________

11/19/2018 3:52PM PROBLEM of "ALL ONE ATTENTION"
the reason the acc is so weird is that our attention weights are all 1(or -1) after 2 epochs. now trying to find why is that...

11/19/2018 4:15PM  change all weight initializers to truncated normal
no effect, same symptom. so weight initializers don't matter.

11/19/2018 5:44PM remove softmax
no effect, same symptom. so softmax doesn't matter.

11/19/2018 6:03PM change adam to gradient descent.
acc jump at epoch 7 but with more training the acc go back, and the attention weight go back to 1/-1.

11/19/2018 6:55PM change attention multiply to add
acc increase to 19.3x at epoch 15. attention weight seems right now.

11/19/2018 7:55PM add batch norm after conv_kernel_a/b add title.
acc around 18.31.

11/19/2018 8:55PM remove all batch norm.
batch norm doesn't matter. 5 epoch 19.48. but 19.4x is the best we can get.




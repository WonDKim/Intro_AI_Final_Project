# Project: 
## Dual-Module Memory based Convolutional Neural Network(DM-MCNN) with Recurrent Neural Filter(RNF) for sentiment analysis
### Member:
	20195427:Wondong Kim 김원동
	20195617:Jirou Feng
	20150350:Jinyong Park 박진용
	20194202:Yang Zidong

### Model: CNN + memory module + RNN Filter

- To train CNN with Linera Filters: `python main.py -linearFilter`
- To train with RNF Filter from scratch: `python main.py`
- To train with trained RNF Filter model: `python main.py -loadModel`


**Note**: Since the size of Glove word vector is too big(2.51GB), we do not include this in
the github. It may take some time to train for the first time(downloading glove word vector).

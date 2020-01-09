# Weibo-retweeting-scale-prediction
CNN+GBDT/SVM
本项目通过爬取10个国家级政务微博近一年的微博数据，训练转发规模分类器，对政务微博转发规模进行预测。特征组合为内容特征（CNN+TF-IDF）+用户特征+时间特征。  
代码部分包括微博爬虫代码、数据预处理代码、文本分类器cnn和转发规模分类器（svm、gbdt）代码。
其中，预处理文件夹包括：  
guiyihua.py: 对特征进行归一化  
period.py：对时间进行woe编码  
tfidf.py: 计算词tf-idf值  
cos_score: 计算新微博与高转发微博关键词的余弦相似度（对新微博进行cos打分）  
CNN文件夹包括：  
loader.py：文件预处理  
train_word2vec：word2vec训练、  
text_model: CNN模型定义  
text_train: CNN模型训练  
text_test:  CNN模型测试  
text_predict: CNN模型预测代码(对新微博进行文本打分)  
训练好的CNN模型存储在checkpoints文件夹中。  
转发规模分类器文件夹包括：  
svm.py: SVM的训练测试代码  
gbdt.py: GBDT的训练测试代码  
retweeting_scale_prediction.py: 转发规模预测代码  
训练好的转发规模分类器模型为gbdt_model.pkl。  
